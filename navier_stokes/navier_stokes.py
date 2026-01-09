import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NavierStokes(nn.Module):
    def __init__(self):
        super().__init__()

        # Input: (t, x, y) | Output: (psi, p)
        # MLP Network
        self.net = nn.Sequential(
            nn.Linear(3, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 2),
        )

        # lambda1 and lambda2 are learnable parameters
        self.lambda1 = nn.Parameter(torch.tensor([0.0]))
        self.lambda2 = nn.Parameter(torch.tensor([0.0]))

    def forward(self, t, x, y):
        return self.net(torch.cat([t, x, y], dim=1))

def physics(model, t, x, y, t_std, x_std, y_std):
    t.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True

    predictions = model(t, x, y)
    psi = predictions[:, 0:1]
    p = predictions[:, 1:2]

    # gradients of latent psi with respect to normalised variables
    psi_t = torch.autograd.grad(psi, t, torch.ones_like(psi), create_graph=True)[0]
    psi_x = torch.autograd.grad(psi, x, torch.ones_like(psi), create_graph=True)[0]
    psi_y = torch.autograd.grad(psi, y, torch.ones_like(psi), create_graph=True)[0]

    # Predict velocity from streaming
    # Impose u = psi_y, v = -psi_x
    # Solution space is divergence-free functions u_x + v_y = 0
    # u_x = psi_yx, v_y = -psi_xy => u_x + v_y = psi_yx - psi_xy = 0
    # Un-normalise features
    u = psi_y * (1.0 / y_std)
    v = -psi_x * (1.0 / x_std)

    # first derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0] * (1.0 / t_std)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0] * (1.0 / x_std)
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0] * (1.0 / y_std)

    v_t = torch.autograd.grad(v, t, torch.ones_like(v), create_graph=True)[0] * (1.0 / t_std)
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0] * (1.0 / x_std)
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0] * (1.0 / y_std)

    # Calculate pressure gradients
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0] * (1.0 / x_std)
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0] * (1.0 / y_std)

    # Viscosity
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0] * (1.0 / x_std)
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0] * (1.0 / y_std)

    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0] * (1.0 / x_std)
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0] * (1.0 / y_std)

    # Data residuals
    f_u = u_t + model.lambda1 * (u*u_x + v*u_y) + p_x - model.lambda2 * (u_xx + u_yy)
    f_v = v_t + model.lambda1 * (u*v_x + v*v_y) + p_y - model.lambda2 * (v_xx + v_yy)

    return u, v, p, f_u, f_v


if __name__ == "__main__":
    print("Loading training data...")
    try:
        df = pd.read_csv("cylinder_wake_train.csv")
    except FileNotFoundError:
        print("Error: 'cylinder_wake_train.csv' not found.")
        exit()
    # raw data
    t_raw = df['t'].values
    x_raw = df['x'].values
    y_raw = df['y'].values
    u_raw = df['u'].values
    v_raw = df['v'].values

    # statistics
    t_mean, t_std = t_raw.mean(), t_raw.std()
    x_mean, x_std = x_raw.mean(), x_raw.std()
    y_mean, y_std = y_raw.mean(), y_raw.std()

    np.savetxt('normalisation_stats.csv', [t_mean, t_std, x_mean, x_std, y_mean, y_std], delimiter=',')
    print(f"Stats saved: t_mean={t_mean:.2f}, x_std={x_std:.2f}...")

    # Normalised data
    t_norm = (t_raw - t_mean) / t_std
    x_norm = (x_raw - x_mean) / x_std
    y_norm = (y_raw - y_mean) / y_std

    t_train = torch.tensor(t_norm[:, None], dtype=torch.float32).to(device)
    x_train = torch.tensor(x_norm[:, None], dtype=torch.float32).to(device)
    y_train = torch.tensor(y_norm[:, None], dtype=torch.float32).to(device)
    u_train = torch.tensor(u_raw[:, None], dtype=torch.float32).to(device)
    v_train = torch.tensor(v_raw[:, None], dtype=torch.float32).to(device)
    # Learn the pressure field

    # Store standard deviation tensors for the loss function
    t_std_th = torch.tensor(t_std, dtype=torch.float32).to(device)
    x_std_th = torch.tensor(x_std, dtype=torch.float32).to(device)
    y_std_th = torch.tensor(y_std, dtype=torch.float32).to(device)

    # Training
    model = NavierStokes().to(device)
    # L-BFGS Optimiser
    optimiser = torch.optim.LBFGS(model.parameters(),
                                  lr=1.0,
                                  history_size=100,
                                  max_iter=5000,
                                  line_search_fn='strong_wolfe')
    print("Starting Training...")
    print(f"Initial lambda1: {model.lambda1.item()}")
    print(f"Initial lambda2: {model.lambda2.item()}")

    loss_history = []
    lambda1_history = []
    lambda2_history = []


    def closure():
        optimiser.zero_grad()

        # Calculate everything in one pass
        u_pred, v_pred, p_pred, f_u, f_v = physics(model, t_train, x_train, y_train,
                                                       t_std_th, x_std_th, y_std_th)

        # Data loss
        loss_u = torch.mean((u_pred - u_train) ** 2)
        loss_v = torch.mean((v_pred - v_train) ** 2)

        # Physics loss
        loss_f = torch.mean(f_u ** 2)
        loss_g = torch.mean(f_v ** 2)

        loss = loss_u + loss_v + loss_f + loss_g
        loss.backward()

        loss_history.append(loss.item())
        lambda1_history.append(model.lambda1.item())
        lambda2_history.append(model.lambda2.item())

        if len(loss_history) % 100 == 0:
            print(f"Iteration {len(loss_history)}: Loss {loss.item():.5f} | "
                  f"lambda_1 {model.lambda1.item():.4f} | lambda_2 {model.lambda2.item():.5f}")

        return loss

    optimiser.step(closure)

    # Results and visualisation
    print("\nTraining Complete.")
    print(f"Final lambda1 (Target 1.0): {model.lambda1.item():.5f}")
    print(f"Final lambda2 (Target 0.01): {model.lambda2.item():.5f}")

    err_l1 = abs(model.lambda1.item() - 1.0) / 1.0 * 100
    err_l2 = abs(model.lambda2.item() - 0.01) / 0.01 * 100
    print(f"Error lambda1: {err_l1:.2f}%")
    print(f"Error lambda2: {err_l2:.2f}%")

    torch.save(model.state_dict(), "ns_inverse_model.pth")
    print("Model saved to ns_inverse_model.pth")

    # Plot Convergence
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.title('Loss History')
    plt.xlabel('Iterations')

    plt.subplot(1, 2, 2)
    plt.plot(lambda1_history, label='lambda1 (Target 1.0)')
    plt.plot(lambda2_history, label='lambda2 (Target 0.01)')
    plt.axhline(1.0, color='grey', linestyle='--')
    plt.axhline(0.01, color='grey', linestyle='--')
    plt.title('Parameter Discovery')
    plt.legend()

    plt.tight_layout()
    plt.savefig("ns_convergence.png")
    print("Saved convergence plot.")