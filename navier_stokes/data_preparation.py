import scipy.io
import numpy as np

data = scipy.io.loadmat('cylinder_wake.mat')

U_star = data['U_star']
P_star = data['p_star']
t_star = data['t']
X_star = data['X_star']

N = X_star.shape[0]
T = t_star.shape[0]

# Flatten data do every row is an observation
XX = np.tile(X_star[:,0:1], (1, T))
YY = np.tile(X_star[:,1:2], (1, T))
TT = np.tile(t_star, (N, 1)).T

x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]

u = U_star[:,0,:].flatten()[:, None]
v = U_star[:,1,:].flatten()[:, None]
p = P_star.flatten()[:, None]

# Training data creation
idx = np.random.choice(N*T, 5000, replace=False)

x_train = x[idx,:]
y_train = y[idx,:]
t_train = t[idx, :]
u_train = u[idx,:]
v_train = v[idx,:]
p_train = p[idx,:]

training_data = np.hstack((t_train, x_train, y_train, u_train, v_train, p_train))
np.savetxt('cylinder_wake_train.csv', training_data, delimiter=',',
           header='t,x,y,u,v,p', comments='')

print(f"Saved {training_data.shape[0]} training points to cylinder_wake_train.csv")