import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def physics(model, t, x, lambda_phy=1.0):
    """
    1D equation
    u_t - 0.0001*u_xx + 5*u**3 - 5*u =0
    """
    u = model(t, x)

    dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    dxx = torch.autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0]

    # residual
    f = dt - 0.0001 * dxx + 5 * u**3 - 5 * u
    return f

"""
Demonstrate effectiveness of random Fourier features
"""
class FourierEmbedding(nn.Module):
    def __init__(self, in_features, out_features, scale):
        super().__init__()
        # B is frequency matrix
        self.B = nn.Parameter(torch.randn(in_features, out_features) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B # 2 for both sin and cos parts
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINN(nn.Module):
    def __init__(self, use_rff=False):
        super().__init__()

        self.use_rff = use_rff

        if use_rff:
            self.embedding = FourierEmbedding(in_features=2, out_features=20, scale=2.0)
            in_dim = 40
        else:
            in_dim = 2

        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=-1)
        if self.use_rff:
            inputs = self.embedding(inputs)
        return self.net(inputs)

def train(mode="baseline", epochs=5000):
    print(f"\n--- Training {mode.upper()} Model ---")
    use_rff = (mode == "improved")

    model = PINN(use_rff=use_rff).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    # Training Data (Collocation Points)
    N_f = 20000
    t_f = torch.rand(N_f, 1).to(device).requires_grad_(True)
    x_f = (torch.rand(N_f, 1) * 2 - 1).to(device).requires_grad_(True)

    # Mini-batching prevents overfitting
    dataset = TensorDataset(t_f, x_f)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)

    # Initial Condition
    N_u = 2000
    x_u = (torch.rand(N_u, 1) * 2 - 1).to(device)
    t_u = torch.zeros_like(x_u).to(device)
    u_exact = x_u ** 2 * torch.cos(np.pi * x_u)

    # Boundary Condition
    N_b = 200
    t_b = torch.rand(N_b, 1).to(device)
    x_b_left = -1 * torch.ones_like(t_b).to(device)
    x_b_right = 1 * torch.ones_like(t_b).to(device)

    loss_history = []
    pde_residual_history = []
    weight_history = {'data': [], 'phys': []}

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for batch_t, batch_x in loader:
            batch_t = batch_t.to(device).requires_grad_(True)
            batch_x = batch_x.to(device).requires_grad_(True)

            optimizer.zero_grad()

            # 1. Physics Loss (Computed on current MINI-BATCH)
            f = physics(model, batch_t, batch_x)
            loss_phys = torch.mean(f ** 2)

            # 2. Data Loss (Computed on FULL fixed set)
            # This anchors the model to the correct shape every single step
            u_pred_ic = model(t_u, x_u)
            loss_ic = torch.mean((u_pred_ic - u_exact) ** 2)

            u_pred_left = model(t_b, x_b_left)
            u_pred_right = model(t_b, x_b_right)
            loss_bc = torch.mean((u_pred_left - u_pred_right) ** 2)

            # Weighting: Heavy IC weight to prevent the trivial Zero Solution
            loss = 100.0 * loss_ic + loss_bc + loss_phys

            loss.backward()
            optimizer.step()

        if epoch % 100 == 0 and epoch > 0:
            scheduler.step()

        if epoch % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: Loss {loss.item():.5f} | Phys {loss_phys.item():.5f} |"
                  f" IC {loss_ic.item():.5f} | LR {current_lr:.1e}")
            loss_history.append(loss.item())
            pde_residual_history.append(loss_phys.item())

    print("\n--- Starting L-BFGS ---")

    # Get full batch
    t_batch = t_f.detach().requires_grad_(True)
    x_batch = x_f.detach().requires_grad_(True)

    optimizer_lbfgs = torch.optim.LBFGS(model.parameters(),
                                        lr=1.0,
                                        history_size=50,
                                        max_iter=1000,
                                        line_search_fn='strong_wolfe',)

    def closure():
        optimizer_lbfgs.zero_grad()

        f = physics(model, t_batch, x_batch)
        loss_phys = torch.mean(f ** 2)

        u_pred_ic = model(t_u, x_u)
        loss_ic = torch.mean((u_pred_ic - u_exact) ** 2)

        u_pred_left = model(t_b, x_b_left)
        u_pred_right = model(t_b, x_b_right)
        loss_bc = torch.mean((u_pred_left - u_pred_right) ** 2)

        loss = 100.0 * loss_ic + loss_bc + loss_phys

        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)
    final_loss = closure()

    print(f"Final L-BFGS Loss: {final_loss.item():.6f}")
    loss_history.append(final_loss.item())

    print(f"Time: {time.time() - start_time:.1f}s")
    print(f"Final PDE Residual: {pde_residual_history[-1]:.6f}")
    return model, loss_history, pde_residual_history, weight_history


if __name__ == "__main__":
    model_base, hist_base, pde_base, weights_base = train("baseline", epochs=2000)
    model_imp, hist_imp, pde_imp, weights_imp = train("improved", epochs=2000)

    # Save models
    torch.save(model_base.state_dict(), "ac_vanilla.pth")
    torch.save(model_imp.state_dict(), "ac_rff.pth")
    print("\nModels saved successfully.")

