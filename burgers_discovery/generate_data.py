import torch
import numpy as np
import Burgers as fwd_pinn

# Load truth data
model_true = fwd_pinn.PINN().to("cpu")
model_true.load_state_dict(torch.load("burgers_pinn.pth", map_location="cpu"))
model_true.eval()

# Generate random points simulating sensor data
N_SAMPLES = 2000
t_vals = np.random.uniform(0, 1, (N_SAMPLES, 1))
x_vals = np.random.uniform(-1, 1, (N_SAMPLES, 1))

t_tensor = torch.tensor(t_vals, dtype=torch.float32)
x_tensor = torch.tensor(x_vals, dtype=torch.float32)

# Add noise once tru u(t,x) found
with torch.no_grad():
    u_true = model_true(t_tensor, x_tensor).numpy()

noise_level = 0.025
noise = noise_level * np.std(u_true) * np.random.randn(N_SAMPLES, 1)
u_noisy = u_true + noise

# Save data
data = np.hstack((t_vals, x_vals, u_noisy))
np.savetxt("burgers_inverse_data.csv", data, delimiter=",", header="t,x,u", comments="")
print("Dataset generated: burgers_inverse_data.csv")