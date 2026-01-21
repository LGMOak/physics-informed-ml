import torch
import numpy as np
import matplotlib.pyplot as plt
import allen_cahn as ac  # Import your training file
from scipy.integrate import odeint

device = torch.device("cpu")


# --- 1. Spectral Solver (Ground Truth) ---
def get_exact_solution():
    print("Generating Ground Truth (Spectral Method)...")
    L = 2.0
    Nx = 512
    Nt = 201
    x = np.linspace(-1, 1, Nx, endpoint=False)
    t = np.linspace(0, 1.0, Nt)
    u0 = x ** 2 * np.cos(np.pi * x)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)

    def reaction_diffusion_wrapper(y_packed, t, k):
        N = len(y_packed) // 2
        u_hat = y_packed[:N] + 1j * y_packed[N:]
        u = np.fft.ifft(u_hat).real
        du_hat = -0.0001 * (k ** 2) * u_hat + np.fft.fft(-5 * u ** 3 + 5 * u)
        return np.concatenate([du_hat.real, du_hat.imag])

    u0_hat = np.fft.fft(u0)
    y0_packed = np.concatenate([u0_hat.real, u0_hat.imag])

    sol_packed = odeint(reaction_diffusion_wrapper, y0_packed, t, args=(k,))
    u_hat_sol = sol_packed[:, :Nx] + 1j * sol_packed[:, Nx:]

    u_sol = np.zeros((Nt, Nx))
    for i in range(Nt):
        u_sol[i, :] = np.fft.ifft(u_hat_sol[i, :]).real
    return x, t, u_sol


if __name__ == "__main__":
    # 1. Load Data & Models
    x_true, t_true, u_true = get_exact_solution()

    print("Loading models...")
    model_base = ac.PINN(use_rff=False).to(device)
    model_imp = ac.PINN(use_rff=True).to(device)

    try:
        # UPDATED FILENAMES HERE
        model_base.load_state_dict(torch.load("ac_vanilla.pth", map_location=device))
        model_imp.load_state_dict(torch.load("ac_rff.pth", map_location=device))
    except FileNotFoundError:
        print("Error: Could not find .pth files (ac_vanilla.pth or ac_rff.pth).")
        exit()

    # 2. Predict Full Field
    print("Predicting full domain...")
    X_mesh, T_mesh = np.meshgrid(x_true, t_true)
    X_torch = torch.tensor(X_mesh.flatten()[:, None], dtype=torch.float32).to(device)
    T_torch = torch.tensor(T_mesh.flatten()[:, None], dtype=torch.float32).to(device)

    with torch.no_grad():
        u_base = model_base(T_torch, X_torch).numpy().reshape(len(t_true), len(x_true))
        u_imp = model_imp(T_torch, X_torch).numpy().reshape(len(t_true), len(x_true))

    # 3. Calculate Errors
    err_base = np.abs(u_true - u_base)
    err_imp = np.abs(u_true - u_imp)

    # 4. PLOT HEATMAPS (Transposed: Time on X-axis)
    print("Plotting heatmaps...")
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # Transpose for [Space x Time] view
    # Origin='lower' means x=-1 is bottom, x=1 is top. t=0 is left, t=1 is right.
    u_true_T = u_true.T
    u_base_T = u_base.T
    u_imp_T = u_imp.T
    err_base_T = err_base.T
    err_imp_T = err_imp.T

    extent = [0, 1, -1, 1]  # [t_min, t_max, x_min, x_max]

    # --- ROW 1: BASELINE ---
    # Exact
    im1 = ax[0, 0].imshow(u_true_T, aspect='auto', extent=extent, cmap='jet', vmin=-1, vmax=1, origin='lower')
    ax[0, 0].set_title("Exact Solution")
    ax[0, 0].set_ylabel("Space (x)")
    plt.colorbar(im1, ax=ax[0, 0])

    # Prediction
    im2 = ax[0, 1].imshow(u_base_T, aspect='auto', extent=extent, cmap='jet', vmin=-1, vmax=1, origin='lower')
    ax[0, 1].set_title("Baseline Prediction (Standard PINN)")
    plt.colorbar(im2, ax=ax[0, 1])

    # Error
    im3 = ax[0, 2].imshow(err_base_T, aspect='auto', extent=extent, cmap='inferno', origin='lower')
    ax[0, 2].set_title("Baseline Error")
    plt.colorbar(im3, ax=ax[0, 2])

    # --- ROW 2: IMPROVED ---
    # Exact (Repeated)
    im4 = ax[1, 0].imshow(u_true_T, aspect='auto', extent=extent, cmap='jet', vmin=-1, vmax=1, origin='lower')
    ax[1, 0].set_title("Exact Solution")
    ax[1, 0].set_ylabel("Space (x)")
    ax[1, 0].set_xlabel("Time (t)")
    plt.colorbar(im4, ax=ax[1, 0])

    # Prediction
    im5 = ax[1, 1].imshow(u_imp_T, aspect='auto', extent=extent, cmap='jet', vmin=-1, vmax=1, origin='lower')
    ax[1, 1].set_title("Improved Prediction (RFF + Hybrid)")
    ax[1, 1].set_xlabel("Time (t)")
    plt.colorbar(im5, ax=ax[1, 1])

    # Error
    im6 = ax[1, 2].imshow(err_imp_T, aspect='auto', extent=extent, cmap='inferno', origin='lower')
    ax[1, 2].set_title("Improved Error")
    ax[1, 2].set_xlabel("Time (t)")
    plt.colorbar(im6, ax=ax[1, 2])

    plt.suptitle("Combating Spectral Bias: Standard vs. RFF-PINN", fontsize=16)
    plt.tight_layout()
    plt.savefig("allen_cahn_final_comparison.png", dpi=150)
    print("Saved 'allen_cahn_final_comparison.png'")
    plt.show()