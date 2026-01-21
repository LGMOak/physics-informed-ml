import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import allen_cahn as ac  # Import the file you just created

device = torch.device("cpu")


# --- 1. Ground Truth Solver ---
def get_exact_solution(t_val, Nx=512):
    L = 2.0
    x = np.linspace(-1, 1, Nx, endpoint=False)
    # Solve specifically up to t_val
    t = np.linspace(0, t_val, 100)
    u0 = x ** 2 * np.cos(np.pi * x)
    k = 2 * np.pi * np.fft.fftfreq(Nx, d=L / Nx)

    def reaction_diffusion(y, t, k):
        N = len(y) // 2
        u = np.fft.ifft(y[:N] + 1j * y[N:]).real
        du = -0.0001 * (k ** 2) * (y[:N] + 1j * y[N:]) + np.fft.fft(-5 * u ** 3 + 5 * u)
        return np.concatenate([du.real, du.imag])

    y0 = np.concatenate([np.fft.fft(u0).real, np.fft.fft(u0).imag])
    sol = odeint(reaction_diffusion, y0, t, args=(k,))

    # Reconstruct final step
    u_final = np.fft.ifft(sol[-1, :Nx] + 1j * sol[-1, Nx:]).real
    return x, u_final


# --- 2. Spectral Error Function ---
def get_error_spectrum(model, t_val, x_grid, u_true):
    # Predict
    t_ten = torch.ones(len(x_grid), 1) * t_val
    x_ten = torch.tensor(x_grid, dtype=torch.float32).view(-1, 1)

    with torch.no_grad():
        u_pred = model(t_ten, x_ten).numpy().flatten()

    # Error
    error = u_true - u_pred

    # FFT
    fft_err = np.fft.fft(error)
    freqs = np.fft.fftfreq(len(error))

    # Shift so 0 frequency is in center
    return np.fft.fftshift(freqs), np.fft.fftshift(np.abs(fft_err)), u_pred


# --- 3. Main Plotting ---
if __name__ == "__main__":
    t_eval = 0.5  # Snapshot time

    # Load Models
    model_vanilla = ac.PINN(use_rff=False).to(device)
    model_rff = ac.PINN(use_rff=True).to(device)
    try:
        model_vanilla.load_state_dict(torch.load("ac_vanilla.pth", map_location=device))
        model_rff.load_state_dict(torch.load("ac_rff.pth", map_location=device))
    except:
        print("Run training script first!")
        exit()

    # Get Data
    x, u_true = get_exact_solution(t_eval)
    f_v, err_v, u_v = get_error_spectrum(model_vanilla, t_eval, x, u_true)
    f_r, err_r, u_r = get_error_spectrum(model_rff, t_eval, x, u_true)

    # --- PLOT ---
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: The Function Space (What it looks like)
    ax[0].plot(x, u_true, 'k-', linewidth=2, label="Exact")
    ax[0].plot(x, u_v, 'r--', label="Vanilla")
    ax[0].plot(x, u_r, 'b--', label="RFF")
    ax[0].set_title(f"Solution at t={t_eval}")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Plot 2: The Frequency Space (The Spectral Bias)
    # We plot on Log Scale to see the differences
    ax[1].semilogy(f_v, err_v, 'r-', alpha=0.6, label="Vanilla Error Spectrum")
    ax[1].semilogy(f_r, err_r, 'b-', alpha=0.6, label="RFF Error Spectrum")

    ax[1].set_title("Spectral Bias Quantification (FFT of Error)")
    ax[1].set_xlabel("Frequency (k)")
    ax[1].set_ylabel("Error Magnitude (Log Scale)")
    ax[1].set_ylim(1e-6, 100)  # Fix scale for clarity
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("spectral_bias_quantification.png")
    print("Saved 'spectral_bias_quantification.png'")
    plt.show()