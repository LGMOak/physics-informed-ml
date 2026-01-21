import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from navier_stokes import NavierStokes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualise():
    try:
        # Load stats
        stats = np.loadtxt('normalisation_stats.csv', delimiter=',')
        t_mean, t_std, x_mean, x_std, y_mean, y_std = stats
    except OSError:
        print("Error: normalisation_stats.csv not found. Run training first.")
        return

    # Initialize Model
    model = NavierStokes().to(device)

    try:
        checkpoint = torch.load("navier_stokes.pth", map_location=device)

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(
                f"Loaded checkpoint. Lambda1: {checkpoint.get('lambda1', '?')}, Lambda2: {checkpoint.get('lambda2', '?')}")
        else:
            # Fallback for older saves
            model.load_state_dict(checkpoint)

    except FileNotFoundError:
        print("Error: navier_stokes.pth not found.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Load Ground Truth Data
    try:
        data = scipy.io.loadmat('cylinder_wake.mat')
    except FileNotFoundError:
        print("Error: cylinder_wake.mat not found.")
        return

    # Extract coordinates and pressure
    X_star = data['X_star']  # N x 2
    t_star = data['t']  # T x 1
    p_star = data['p_star']  # N x T

    snap_idx = 100  # Time index to visualise
    t_snap = t_star[snap_idx, 0]
    p_true = p_star[:, snap_idx]

    # Grid coordinates
    x_grid = X_star[:, 0]
    y_grid = X_star[:, 1]

    # Normalise Inputs for the PINN
    t_norm_val = (t_snap - t_mean) / t_std
    x_norm_vals = (x_grid - x_mean) / x_std
    y_norm_vals = (y_grid - y_mean) / y_std

    # Create tensors
    t_tensor = torch.full((x_grid.shape[0], 1), t_norm_val, dtype=torch.float32).to(device)
    x_tensor = torch.tensor(x_norm_vals[:, None], dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_norm_vals[:, None], dtype=torch.float32).to(device)

    # Predict Pressure
    with torch.no_grad():
        predictions = model(t_tensor, x_tensor, y_tensor)
        p_pred_tensor = predictions[:, 1:2]
        p_pred = p_pred_tensor.cpu().numpy().flatten()

    # Post-Processing: Center pressure (relative value)
    p_true_centered = p_true - np.mean(p_true)
    p_pred_centered = p_pred - np.mean(p_pred)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    # Truth Heatmap
    sc1 = ax[0].scatter(x_grid, y_grid, c=p_true_centered, cmap='RdBu', s=20)
    ax[0].set_title(f'Ground Truth Pressure (t={t_snap:.2f})')
    plt.colorbar(sc1, ax=ax[0])
    ax[0].axis('equal')

    # Prediction Heatmap
    sc2 = ax[1].scatter(x_grid, y_grid, c=p_pred_centered, cmap='RdBu', s=20)
    ax[1].set_title(f'PINN Prediction Pressure (t={t_snap:.2f})')
    plt.colorbar(sc2, ax=ax[1])
    ax[1].axis('equal')

    plt.suptitle(f"Parameters: $\lambda_1$={model.lambda1.item():.4f}, $\lambda_2$={model.lambda2.item():.5f}")
    plt.tight_layout()
    plt.savefig('pressure_comparison.png')
    print("Saved pressure_comparison.png")
    plt.show()


if __name__ == "__main__":
    visualise()