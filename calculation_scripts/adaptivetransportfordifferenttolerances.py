from numerical_integrators.adaptive_step import rkbs32, rkbs54, rkck45, rkdp54, rkdp87, rkf12, rkf45, rkhe21

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

def timestep(t,pos,h,deriv,integrator,tol):
    t, pos, h = integrator(t, pos, h, deriv, atol = tol, rtol = tol)
    return t, pos, h

integrators = np.array([rkbs54, rkck45, rkdp54, rkdp87, rkf45])
tol = np.logspace(-13, 0, num = 14)

def step_all_the_way(t, pos, h, deriv, integrator, tol, t_max, q):
    while np.any(np.less(t, t_max)):
        h = np.minimum(h, t_max - t)

        t, pos, h = timestep (t, pos, h, deriv, integrator, tol)

    q.put(pos)

N = 4

part = np.floor((Nx*Ny)/N).astype(int)

for integrator in integrators:

    for i in range(len(tol)):
        print('{}: {} of {}'.format(integrator.__name__, i, len(tol)))
        pos = np.copy(initpos)
        ts = np.ones(Nx*Ny)*t_min
        hs = np.ones(Nx*Ny)*0.01



        queuelist = [mp.Queue() for j in range(N)]
        processlist = [mp.Process(target=step_all_the_way,
            args = (ts[j*part:(Nx*Ny) if j + 1 == N else (j+1)*part],
                pos[:, j*part:(Nx*Ny) if j + 1 == N else (j + 1)*part],
                hs[j*part:(Nx*Ny) if j + 1 == N else (j + 1)*part],
                vel,
                integrator,
                tol[i],
                t_max,
                queuelist[j])) for j in range(N)]

        for process in processlist:
            process.start()

        for j in range(N):
            pos[:, j*part:(Nx*Ny) if j + 1 == N else (j+1)*part] = queuelist[j].get()

        for process in processlist:
            process.join()

        np.save('particlepositions_errorestimation/{}_tol={}'.format(integrator.__name__,tol[i]), pos)


#for i in range(len(tol)):
#    pos = np.copy(initpos)
#    ts = np.ones(Nx*Ny)*t_min
#    hs = np.ones(Nx*Ny)*0.01
#
#    while np.any(np.less(ts, t_max)):
#        hs = np.minimum(hs, t_max - ts)
#
#        ts, pos, hs = timestep(ts, pos, hs, vel, integrator, tol[i])
#
#    np.save('particlepositions_errorestimation/{}_tol={}'.format(integrator.__name__,tol[i]), pos)

