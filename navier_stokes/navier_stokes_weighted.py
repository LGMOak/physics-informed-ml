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
        # Simple MLP architecture
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

def load_data():
    print("Loading training data...")
    # Load data
    try:
        df = pd.read_csv("cylinder_wake_train.csv")
    except FileNotFoundError:
        print("Error: 'cylinder_wake_train.csv' not found.")
        return None

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

    return t_norm, x_norm, y_norm, u_raw, v_raw, t_std, x_std, y_std


def train(t_norm, x_norm, y_norm, u_raw, v_raw, t_std, x_std, y_std,
          adam_iters=5000, lbfgs_iters=10000, w_data=1.0, w_physics=1.0, save_name="navier_stokes.pth"):
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

    model = NavierStokes().to(device)

    history = {
        'loss': [], 'loss_data': [], 'loss_physics': [],
        'lambda1': [], 'lambda2': [], 'phase': []
    }

    # Begin with Adam optimisation as a first-pass parameter discovery
    print(f"Initial: lambda1={model.lambda1.item():.4f}, lambda2={model.lambda2.item():.5f}")

    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(adam_iters):
        optimizer_adam.zero_grad()

        u_pred, v_pred, p_pred, f_u, f_v = physics(
            model, t_train, x_train, y_train, t_std_th, x_std_th, y_std_th
        )

        loss_u = torch.mean((u_pred - u_train) ** 2)
        loss_v = torch.mean((v_pred - v_train) ** 2)
        loss_f = torch.mean(f_u ** 2)
        loss_g = torch.mean(f_v ** 2)

        loss_data = loss_u + loss_v
        loss_physics = loss_f + loss_g

        loss = w_data * loss_data + w_physics * loss_physics

        loss.backward()
        optimizer_adam.step()

        history['loss'].append(loss.item())
        history['loss_data'].append(loss_data.item())
        history['loss_physics'].append(loss_physics.item())
        history['lambda1'].append(model.lambda1.item())
        history['lambda2'].append(model.lambda2.item())
        history['phase'].append('Adam')

        if i % 500 == 0 or i == adam_iters - 1:
            print(f"Iter {i:4d}: Loss {loss.item():.6f} | "
                  f"Data {loss_data.item():.6f} | Phys {loss_physics.item():.6f} | "
                  f"lambda1 {model.lambda1.item():.4f} | lambda2 {model.lambda2.item():.5f}")

    print(f"\nAfter Adam: lambda1={model.lambda1.item():.4f}, lambda2={model.lambda2.item():.5f}")

    # Switch to L-BFGS optimiser for precise parameter refinement
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=100,
        history_size=50,
        line_search_fn='strong_wolfe'
    )

    def closure():
        optimizer_lbfgs.zero_grad()

        u_pred, v_pred, p_pred, f_u, f_v = physics(
            model, t_train, x_train, y_train, t_std_th, x_std_th, y_std_th
        )

        loss_u = torch.mean((u_pred - u_train) ** 2)
        loss_v = torch.mean((v_pred - v_train) ** 2)
        loss_f = torch.mean(f_u ** 2)
        loss_g = torch.mean(f_v ** 2)

        loss_data = loss_u + loss_v
        loss_physics = loss_f + loss_g

        loss = w_data * loss_data + w_physics * loss_physics

        loss.backward()

        history['loss'].append(loss.item())
        history['loss_data'].append(loss_data.item())
        history['loss_physics'].append(loss_physics.item())
        history['lambda1'].append(model.lambda1.item())
        history['lambda2'].append(model.lambda2.item())
        history['phase'].append('L-BFGS')

        if len(history['loss']) % 1000 == 0:
            print(f"Iter {len(history['loss']):5d}: Loss {loss.item():.6f} | "
                  f"lambda1 {model.lambda1.item():.5f} | lambda2 {model.lambda2.item():.6f}")

        return loss

    for epoch in range(lbfgs_iters // 20):
        optimizer_lbfgs.step(closure)

    # Results
    final_lambda1 = model.lambda1.item()
    final_lambda2 = model.lambda2.item()

    err_l1 = abs(final_lambda1 - 1.0) / 1.0 * 100
    err_l2 = abs(final_lambda2 - 0.01) / 0.01 * 100

    print(f"\nDiscovered Parameters:")
    print(f"  lambda1 = {final_lambda1:.6f} (target: 1.0, error: {err_l1:.2f}%)")
    print(f"  lambda2 = {final_lambda2:.6f} (target: 0.01, error: {err_l2:.2f}%)")

    torch.save({
        'model_state_dict': model.state_dict(),
        'lambda1': final_lambda1,
        'lambda2': final_lambda2,
        'history': history
    }, save_name)

    print(f"\nModel saved to '{save_name}'")

    return model, history


def plot_comparison(results):
    """
    Plots the convergence of lambda1 and lambda2 for multiple training runs.
    'results' is a dictionary: { 'Run Label': history_dict, ... }
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    colours = ['b', 'g', 'r', 'c', 'm']

    for idx, (label, history) in enumerate(results.items()):
        colour = colours[idx % len(colours)]
        iters = range(len(history['lambda1']))

        # Determine where L-BFGS started
        try:
            phase_change = next(i for i, p in enumerate(history['phase']) if p == 'L-BFGS')
        except StopIteration:
            phase_change = None

        # --- Plot Lambda 1 ---
        axes[0].plot(iters, history['lambda1'], label=f"{label}", color=colour, linewidth=1.5, alpha=0.8)
        if phase_change:
            axes[0].scatter(phase_change, history['lambda1'][phase_change], color=colour, marker='x', s=50)

        # --- Plot Lambda 2 ---
        axes[1].plot(iters, history['lambda2'], label=f"{label}", color=colour, linewidth=1.5, alpha=0.8)
        if phase_change:
            axes[1].scatter(phase_change, history['lambda2'][phase_change], color=colour, marker='x', s=50)

    # Formatting Lambda 1 Plot
    axes[0].axhline(1.0, color='k', linestyle='--', linewidth=2, label='Target (1.0)')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel('$\lambda_1$ Value')
    axes[0].set_title('Discovery of Advection ($\lambda_1$)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Formatting Lambda 2 Plot
    axes[1].axhline(0.01, color='k', linestyle='--', linewidth=2, label='Target (0.01)')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel('$\lambda_2$ Value')
    axes[1].set_title('Discovery of Viscosity ($\lambda_2$)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('parameter_comparison.png', dpi=150)
    print("\nSaved comparison plot to 'parameter_comparison.png'")
    plt.show()


if __name__ == "__main__":
    t_norm, x_norm, y_norm, u_raw, v_raw, t_std, x_std, y_std = load_data()
    configs = [
        (1.0, 1.0, "Balanced (1:1)", "navier_stokes_balanced.pth"),
        (10.0, 1.0, "Data Weighted (10:1)", "navier_stokes_mid.pth"),
        (100.0, 1.0, "Strong Data (100:1)", "navier_stokes_strong.pth"),
    ]

    # Dictionary to store the history of each run
    all_results = {}

    for w_data, w_physics, desc, filename in configs:
        # Train the model
        model, history = train(
            t_norm, x_norm, y_norm, u_raw, v_raw, t_std, x_std, y_std,
            w_data=w_data,
            w_physics=w_physics,
            save_name=filename
        )

        # Store results
        if history is not None:
            all_results[desc] = history

    # Plot the comparison of all runs together
    if all_results:
        plot_comparison(all_results)