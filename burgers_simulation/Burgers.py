import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

nu = 0.01 / np.pi

class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (t,x) | Output u(t,x)
        # Tanh activation needed for calculating second derivatives
        self.net = nn.Sequential(
            nn.Linear(2, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 20), nn.Tanh(),
            nn.Linear(20, 1)
        )

    def forward(self, t, x):
        return self.net(torch.cat([t, x], dim=1))

'''
Physics loss utilising automatic differentiation
'''
def physics_loss(model, t, x):
    t.requires_grad = True
    x.requires_grad = True

    u = model(t, x)

    # first derivative with autograd
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    # residual: f=u_t + u*u_x - nu*u_xx
    residual = u_t + u*u_x - nu*u_xx

    return torch.mean(residual**2)


if __name__ == "__main__":
    '''
    Data generation
    Initial and Boundary Conditions
    '''
    # t=0, between x=-1 and x=1
    x_ic = np.random.uniform(-1, 1, 200).reshape(-1, 1)
    t_ic = np.zeros_like(x_ic)
    u_ic = -np.sin(np.pi * x_ic)

    # x=-1, between t=0 and t=1
    t_bc1 = np.random.uniform(0, 1, 200).reshape(-1, 1)
    x_bc1 = -np.ones_like(t_bc1)
    u_bc1 = np.zeros_like(t_bc1) # u(t, -1)=0

    # x=1, between t=0 and t=1
    t_bc2 = np.random.uniform(0, 1, 200).reshape(-1, 1)
    x_bc2 = np.ones_like(t_bc2)
    u_bc2 = np.zeros_like(t_bc2) # u(t, 1)=0

    # Combine Data
    t_data = torch.tensor(np.vstack([t_ic, t_bc1, t_bc2]), dtype=torch.float32).to(device)
    x_data = torch.tensor(np.vstack([x_ic, x_bc1, x_bc2]), dtype=torch.float32).to(device)
    u_data = torch.tensor(np.vstack([u_ic, u_bc1, u_bc2]), dtype=torch.float32).to(device)

    # Collocation points. Data unknown, we want to calculate
    # randon points in the domain
    t_calc = torch.tensor(np.random.uniform(0, 1, (2000, 1)), dtype=torch.float32).to(device)
    x_calc = torch.tensor(np.random.uniform(-1, 1, (2000, 1)), dtype=torch.float32).to(device)

    '''
    Training Loop
    '''
    model = PINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5000):
        optimizer.zero_grad()

        # initial and boundary loss
        u_pred = model(t_data, x_data)
        loss_u = torch.mean((u_pred - u_data)**2)

        # equation loss
        loss_f = physics_loss(model, t_calc, x_calc)

        # total loss
        loss = loss_u + loss_f

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss_u = {loss_u.item():.4f}, Loss_f = {loss_f.item():.4f}, Total Loss = {loss.item():.4f}")

    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "burgers_pinn.pth")
