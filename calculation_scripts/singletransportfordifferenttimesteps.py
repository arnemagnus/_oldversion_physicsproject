from numerical_integrators.single_step import euler, rk2, rk3, rk4

import multiprocessing as mp

import numpy as np
from velocity_field import vel

t_min, t_max = 0, 5

x_min, x_max = 0, 2
y_min, y_max = 0, 1

Ny = 201
Nx = 1 + np.floor((Ny-1)*(x_max-x_min)/(y_max-y_min)).astype(int)

dx = (x_max - x_min)/(Nx-1)
dy = (y_max - y_min)/(Ny-1)

x0 = np.linspace(x_min, x_max, Nx)
y0 = np.linspace(y_min, y_max, Ny)

x = np.zeros(Nx*Ny)
y = np.copy(x)

for j in range(Ny):
    x[j*Nx:(j+1)*Nx] = x0
    y[j*Nx:(j+1)*Nx] = y0[j]

pos = np.array([x, y])
initpos = np.copy(pos)

def timestep(t,pos,h,deriv,integrator):
    t, pos, h = integrator(t, pos, h, deriv)
    return t, pos, h

integrators = np.array([euler, rk2, rk3, rk4])
tol = np.logspace(-4, 0, num = 10)

def step_all_the_way(t, pos, h, deriv, integrator, t_max, q):
    while np.any(np.less(t, t_max)):
        h = np.minimum(h, t_max - t)

        t, pos, h = timestep(t, pos, h, deriv, integrator)

    q.put(pos)

N = 4

part = np.floor((Nx*Ny)/N).astype(int)

for integrator in integrators:

    for i in range(len(tol)):
        print('{}: {} of {}'.format(integrator.__name__, i, len(tol)))
        pos = np.copy(initpos)
        t = np.ones(Nx*Ny)*t_min
        h = np.ones(Nx*Ny)*tol[i]



        queuelist = [mp.Queue() for j in range(N)]
        processlist = [mp.Process(target=step_all_the_way,
            args = (t[j*part:(Nx*Ny) if j + 1 == N else (j + 1)*part],
                pos[:, j*part:(Nx*Ny) if j + 1 == N else (j + 1)*part],
                h[j*part:(Nx*Ny) if j + 1 == N else (j + 1)*part],
                vel,
                integrator,
                t_max,
                queuelist[j])) for j in range(N)]

        for process in processlist:
            process.start()

        for j in range(N):
            pos[:, j*part:(Nx*Ny) if j + 1 == N else (j+1)*part] = queuelist[j].get()

        for process in processlist:
            process.join()

        np.save('particlepositions_errorestimation/{}_h={}'.format(integrator.__name__,tol[i]), pos)


