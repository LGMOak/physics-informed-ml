import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

'''
Goal: learn nu parameter from data set
'''
class InversePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Multi-Layer Perceptron
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

        # Physics parameter initialised to 0
        self.nu_parameter = nn.Parameter(torch.tensor([0.0]))

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

'''
Calculates physics loss using automatic differentiation trick
'''
def inverse_physics_loss(model, t, x):
    t.requires_grad = True
    x.requires_grad = True
    u = model(t, x)

    # Derivatives
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # Minimise the residual by finding the nu parameter
    residual = u_t + (u * u_x) - (model.nu_parameter * u_xx)

    return torch.mean(residual**2)

try:
    data = np.loadtxt("burgers_inverse_data.csv", delimiter=",", skiprows=1)
except OSError:
    print("Error: 'burgers_inverse_data.csv' not found. Please run the data generation script first.")
    exit()

t_data = torch.tensor(data[:, 0:1], dtype=torch.float32).to(device)
x_data = torch.tensor(data[:, 1:2], dtype=torch.float32).to(device)
u_data = torch.tensor(data[:, 2:3], dtype=torch.float32).to(device)

model = InversePINN().to(device)

# Optimiser learns weights and nu_parameter automatically
# L-BFGS over Adam here
optimiser = torch.optim.LBFGS(model.parameters(), lr=1.0, history_size=100, max_iter=5000, line_search_fn='strong_wolfe')

loss_history = []
nu_history = []

def closure():
    optimiser.zero_grad()

    u_pred = model(t_data, x_data)
    loss_u = torch.mean((u_data - u_pred) ** 2)
    loss_f = inverse_physics_loss(model, t_data, x_data)
    loss = loss_u + loss_f
    loss.backward()

    current_nu = model.nu_parameter.item()
    nu_history.append(current_nu)
    loss_history.append(loss.item())

    if len(loss_history) % 100 == 0:
        error = abs(current_nu - true_nu) / true_nu * 100
        print(f"Step {len(loss_history)}: Loss {loss.item():.5f} | Nu {current_nu:.6f} | Err {error:.2f}%")

    return loss

true_nu = 0.01 / np.pi
print(f" Target Nu: {true_nu:.6f}")

optimiser.step(closure)

print(f"\nFinal discovered nu: {model.nu_parameter.item():.6f}")
print(f"True nu: {true_nu:.6f}")
print(f"Final Error: {abs(model.nu_parameter.item() - true_nu)/true_nu*100:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(nu_history, label='Discovered nu')
plt.axhline(y=true_nu, color='r', linestyle='--', label='True nu')
plt.xlabel('L-BFGS Iterations')
plt.ylabel('Viscosity Parameter')
plt.title('Parameter Discovery Convergence')
plt.legend()
plt.grid()
plt.savefig("inverse_convergence.png")
print("Saved convergence plot to inverse_convergence.png")