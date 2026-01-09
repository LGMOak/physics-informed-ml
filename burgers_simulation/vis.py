import matplotlib.pyplot as plt
import Burgers as pinn
import torch
import numpy as np

model = pinn.PINN().to("cpu")

model.load_state_dict(torch.load("burgers_pinn.pth", map_location=torch.device('cpu')))
model.eval()

t_vals = np.linspace(0, 1, 100)
x_vals = np.linspace(-1, 1, 256)
T, X = np.meshgrid(t_vals, x_vals)

# Flatten network
t_flat = torch.tensor(T.flatten()[:, None], dtype=torch.float32).to("cpu")
x_flat = torch.tensor(X.flatten()[:, None], dtype=torch.float32).to("cpu")

# Make predictions
with torch.no_grad():
    u_pred = model(t_flat, x_flat).cpu().numpy()

# Reshape back to grid
U_pred = u_pred.reshape(T.shape)


# 3. Plotting
fig, ax = plt.subplots(figsize=(10, 5))
# Heatmap of the solution u(t,x)
c = ax.pcolormesh(T, X, U_pred, cmap='jet', shading='auto')
fig.colorbar(c, ax=ax)
ax.set_title("PINN Solution to Burgers' Equation")
ax.set_xlabel("Time (t)")
ax.set_ylabel("Space (x)")
plt.savefig("burgers_heatmap.png")
print("Saved burgers_heatmap.png")

# 4. Cross-section snapshots (to see the shock)
plt.figure(figsize=(10, 6))
times = [0.0, 0.25, 0.5, 0.75]
for t in times:
    # Extract the column corresponding to time t
    t_idx = int(t * 100) # approximate index
    if t_idx >= 100: t_idx = 99
    plt.plot(x_vals, U_pred[:, t_idx], label=f"t={t}")

plt.title("Shock Formation over Time")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.legend()
plt.grid()
plt.savefig("burgers_shocks.png")
print("Saved burgers_shocks.png")
