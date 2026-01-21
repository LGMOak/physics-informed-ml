import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 1000
N_COLLOCATION = 20000
N_IC = 2000
N_BC = 200

def physics(model, t, x):
    """
    Allen-Cahn equation
    """
    u = model(t, x)

    dt = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    dx = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    dxx = torch.autograd.grad(dx, x, torch.ones_like(dx), create_graph=True)[0]

    f = dt - 0.0001 * dxx + 5 * u**3 - 5 * u
    return f

class AC_SA_PINN(nn.Module):
    def __init__(self):
        super().__init__()

        # Vanilla MLP Architecture
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 1)
        )

        # Learnable weights
        self.sa_weights = nn.Parameter(torch.zeros(N_COLLOCATION, 1))

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

# Data generation
# Domain Collocation points
t_col = torch.rand(N_COLLOCATION, 1).to(device).requires_grad_(True)
x_col = (torch.rand(N_COLLOCATION, 1) * 2 - 1).to(device).requires_grad_(True)

# Initial Condition (t=0)
x_ic = (torch.rand(N_IC, 1) * 2 - 1).to(device)
t_ic = torch.zeros_like(x_ic).to(device)
u_ic_exact = x_ic ** 2 * torch.cos(np.pi * x_ic)

# Boundary Condition (x=-1, x=1) - Periodic
t_bc = torch.rand(N_BC, 1).to(device)
x_bc_left = -1 * torch.ones_like(t_bc).to(device)
x_bc_right = 1 * torch.ones_like(t_bc).to(device)

# Training
model = AC_SA_PINN().to(device)

# Optimiser minimises loss
optim_model = torch.optim.Adam(model.parameters(), lr=1e-3)
# Optimiser maximises weights
optim_weights = torch.optim.Adam([model.sa_weights], lr=1e-2)

print("Starting SA-PINN Training for Allen-Cahn...")
start_time = time.time()
loss_history = []

for epoch in range(EPOCHS):
    # Calculate residuals
    f = physics(model, t_col, x_col)

    # IC/BC errors
    u_pred_ic = model(t_ic, x_ic)
    loss_ic = torch.mean((u_pred_ic - u_ic_exact) ** 2)

    u_pred_left = model(t_bc, x_bc_left)
    u_pred_right = model(t_bc, x_bc_right)
    loss_bc = torch.mean((u_pred_left - u_pred_right) ** 2)

    # Update model
    # Apply mask to physics
    mask = torch.sigmoid(model.sa_weights) * 100.0 # Allows weight up to 100

    # Weighted Physics Loss
    loss_phys = torch.mean(mask.detach() * f ** 2)

    # Total Loss (Standard weights for BC/IC)
    loss = loss_phys + 100.0 * loss_ic + loss_bc

    optim_model.zero_grad()
    loss.backward()
    optim_model.step()

    # Update weights
    mask_train = torch.sigmoid(model.sa_weights) * 100.0
    loss_weights = torch.mean(mask_train * f.detach() ** 2)

    optim_weights.zero_grad()
    (-loss_weights).backward()  # Gradient Ascent
    optim_weights.step()

    if epoch % 500 == 0:
        w_mean = mask_train.mean().item()
        w_max = mask_train.max().item()
        print(
            f"Epoch {epoch}: Loss {loss.item():.4f} | Phys {loss_phys.item():.4f} | Weights [Mean: {w_mean:.1f}, Max: {w_max:.1f}]")
        loss_history.append(loss.item())

    print(f"Training Time: {time.time() - start_time:.1f}s")

# Visualisation
t_plot = torch.linspace(0, 1, 100).to(device)
x_plot = torch.linspace(-1, 1, 256).to(device)
T, X = torch.meshgrid(t_plot, x_plot, indexing='ij')

with torch.no_grad():
    t_flat = T.flatten()[:, None]
    x_flat = X.flatten()[:, None]

    # Predict
    u_pred = model(t_flat, x_flat)

    u_plot = u_pred.reshape(100, 256).cpu().numpy()


plt.figure(figsize=(8, 6))
plt.imshow(u_plot, extent=[-1, 1, 0, 1], origin='lower', aspect='auto', cmap='jet')
plt.colorbar(label='u(t,x)')
plt.xlabel('x')
plt.ylabel('t')
plt.title('SA-PINN Solution (Allen-Cahn)')
plt.tight_layout()
plt.savefig('sa_pinn_result.png')
plt.show()


