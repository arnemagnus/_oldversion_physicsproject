from numerical_integrators.single_step import rk4
import numpy as np
import multiprocessing as mp
from velocity_field import double_gyre

integrator = rk4
hs = np.logspace(-5,0,12)
evals_step = 4

x_min, x_max = 0, 2
y_min, y_max = 0, 1

Ny = 200
Nx = int(Ny * (x_max - x_min)/(y_max - y_min))

# Grid spacing:
delta = (x_max - x_min)/Nx

# "Basis vectors along either direction
x_0 = (np.arange(Nx) + 1/2) * delta
y_0 = (np.arange(Ny) + 1/2) * delta

x = np.zeros(Nx * Ny)
y = np.copy(x)

for j in range(Ny):
    x[j*Nx : (j+1)*Nx] = x_0
    y[j*Nx : (j+1)*Nx] = y_0[j]

# Initial positions:
pos_init = np.array([x, y])
pos = np.copy(pos_init)

t_min = 0
t_max = 5

partition = np.floor(np.size(pos,1)/4).astype(int)

def transport_fixed_slice(t_min, t_max, pos, h, integrator, deriv, q):
    t = t_min
    for j in range(np.ceil((t_max - t_min)/h).astype(int)):
        t, pos, h = integrator(t, pos, h, deriv)
    q.put(pos)

for i, h in enumerate(hs):
    queuelist = [mp.Queue() for j in range(4)]
    steps = np.ones(Nx*Ny) * np.ceil((t_max - t_min)/h).astype(int) * evals_step
    processlist = [mp.Process(target = transport_fixed_slice,
                                args = (t_min, t_max,
                                        pos_init[:, j*partition:Nx*Ny if j + 1 == 4 else (j+1)*partition],
                                        h, integrator, double_gyre,
                                        queuelist[j]
                                        )
                                ) for j in range(4)]

    for process in processlist:
        process.start()
    for j in range(4):
        pos[:, j*partition:Nx*Ny if j + 1 == 4 else (j+1)*partition] = queuelist[j].get()
    for process in processlist:
        process.join()
    np.save('runs/endpos_{}_h={}.npy'.format(integrator.__name__, h), pos)
    np.save('runs/evaluations_{}_h={}.npy'.format(integrator.__name__, h), steps)



