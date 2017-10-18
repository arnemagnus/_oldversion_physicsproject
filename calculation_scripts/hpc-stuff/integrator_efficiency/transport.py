from velocity_field import double_gyre
import numpy as np
import multiprocessing as mp
from numerical_integrators.single_step import euler, rk2, rk3, rk4
from numerical_integrators.adaptive_step import rkbs32, rkbs54, rkck45, rkdp54,\
                                                rkdp87, rkf45, rkf78
import sys

def transport(chosen_integrator, n_proc):
    if chosen_integrator == 'e':
        integrator = euler
        h = np.logspace(-5, 0, 12)
        evals_step = 1
    elif chosen_integrator == 'h':
        integrator = rk2
        h = np.logspace(-5, 0, 12)
        evals_step = 2
    elif chosen_integrator == 'k':
        integrator = rk3
        h = np.logspace(-5, 0, 12)
        evals_step = 3
    elif chosen_integrator == 'r':
        integrator = rk4
        h = np.logspace(-5, 0, 12)
        evals_step = 4
    elif chosen_integrator == 'rkbs32':
        integrator = rkbs32
        tols = np.logspace(-10, 0, 15)
        evals_step = 4
    elif chosen_integrator == 'rkbs54':
        integrator = rkbs54
        tols = np.logspace(-10, 0, 15)
        evals_step = 8
    elif chosen_integrator == 'rkck45':
        integrator = rkck45
        tols = np.logspace(-10, 0, 15)
        evals_step = 6
    elif chosen_integrator == 'rkdp54':
        integrator = rkdp54
        tols = np.logspace(-10, 0, 15)
        evals_step = 7
    elif chosen_integrator == 'rkdp87':
        integrator = rkdp87
        tols = np.logspace(-10, 0, 15)
        np.save('runs/tolerances_adaptive.npy', tols)
        tols = np.array([1e-12])
        z = np.logspace(-5, 0, 12)
        np.save('runs/steplengths_single.npy', z)
        evals_step = 13
    elif chosen_integrator == 'rkf45':
        integrator = rkf45
        tols = np.logspace(-10, 0, 15)
        evals_step = 6
    elif chosen_integrator == 'rkf78':
        integrator = rkf78
        tols = np.logspace(-10, 0, 15)
        evals_step = 13
    else:
        print('Integrator choice {} not valid. Try a different integrator.'.format(chosen_integrator))
        sys.exit()

    # Defining the domain of interest:
    x_min, x_max = 0, 2
    y_min, y_max = 0, 1

    # Number of points in the y-direction chosen as the independent variable:
    Ny = 200
    Nx = int(Ny * (x_max - x_min) / (y_max - y_min))

    # Grid spacing, equidistant grid:
    delta = (x_max - x_min) / Nx

    # "Basis" vectors along either direction:
    x_0 = (np.arange(Nx) + 1/2) * delta
    y_0 = (np.arange(Ny) + 1/2) * delta

    x = np.zeros(Nx*Ny)
    y = np.copy(x)

    for j in range(Ny):
        x[j*Nx:(j+1)*Nx] = x_0
        y[j*Nx:(j+1)*Nx] = y_0[j]

    # Initial positions:
    pos_init = np.array([x, y])
    pos = np.copy(pos_init)

    # Time interval of interest:
    t_min = 0
    t_max = 5

    partition = np.floor(np.size(pos, 1)/n_proc).astype(int)

    if 'h' in locals():
        for i in range(np.size(h)):
            queuelist = [mp.Queue() for j in range(n_proc)]
            steps = np.ones(Nx*Ny) * np.ceil((t_max - t_min)/h[i]).astype(int) * evals_step
            processlist = [mp.Process(target = transport_fixed_slice,
                                      args = (t_min, t_max,
                                      pos_init[:, j*partition:Nx*Ny if j + 1 == n_proc else (j+1)*partition],
                                      h[i], integrator, double_gyre,
                                      queuelist[j])) for j in range(n_proc)]
            for process in processlist:
                process.start()
            for j in range(n_proc):
                pos[:, j*partition:Nx*Ny if j + 1 == n_proc else (j+1)*partition] = queuelist[j].get()
            for process in processlist:
                process.join()

            np.save('runs/endpos_{}_h={}.npy'.format(integrator.__name__, h[i]), pos)
            np.save('runs/evaluations_{}_h={}.npy'.format(integrator.__name__, h[i]), steps)

    else:
        for i in range(np.size(tols)):
            positionqueuelist = [mp.Queue() for j in range(n_proc)]
            steps = np.zeros(Nx*Ny, dtype = int)
            stepqueuelist = [mp.Queue() for j in range(n_proc)]
            processlist = [mp.Process(target = transport_adaptive_slice,
                                      args = (t_min, t_max,
                                      pos_init[:, j*partition:Nx*Ny if j + 1 == n_proc else (j+1)*partition],
                                      tols[i], integrator, evals_step, double_gyre,
                                      positionqueuelist[j], stepqueuelist[j])) for j in range(n_proc)]
            for process in processlist:
                process.start()
            for j in range(n_proc):
                pos[:, j*partition:Nx*Ny if j + 1 == n_proc else (j+1)*partition] = positionqueuelist[j].get()
                steps[j*partition:Nx*Ny if j + 1 == n_proc else (j+1)*partition] = stepqueuelist[j].get()
            for process in processlist:
                process.join()

            np.save('runs/endpos_{}_tol={}.npy'.format(integrator.__name__,tols[i]), pos)
            np.save('runs/evaluations_{}_tol={}.npy'.format(integrator.__name__,tols[i]), steps)


def transport_fixed_slice(t_min, t_max, pos, h, integrator, deriv, q):
    t = t_min
    for j in range(np.ceil((t_max - t_min)/h).astype(int)):
        t, pos, h = integrator(t, pos, h, deriv)
    q.put(pos)

def transport_adaptive_slice(t_min, t_max, pos, tol, integrator, evals_step, deriv, posqueue, stepqueue):
    h = np.ones(np.size(pos, 1)) * 0.01
    t = np.ones(np.size(pos, 1)) * t_min

    steps = np.zeros(np.size(pos, 1), dtype = 'int')

    unfinished_mask = np.less(t, t_max)

    while np.any(unfinished_mask):
        h[unfinished_mask] = np.minimum(h[unfinished_mask], t_max - t[unfinished_mask])
        t[unfinished_mask], tmp, h[unfinished_mask] = integrator(t[unfinished_mask],
                                                                pos[:, unfinished_mask],
                                                                h[unfinished_mask],
                                                                deriv,
                                                                atol = tol,
                                                                rtol = tol)
        pos[:, unfinished_mask] = tmp
        steps[unfinished_mask] += evals_step

        unfinished_mask = np.less(t, t_max)
    posqueue.put(pos)
    stepqueue.put(steps)




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=('Input arguments to the\
                                                    transport function.'))

    parser.add_argument('-m', type=str, help = 'Chosen integrator')
    parser.add_argument('-n', type=int, default = 1, help = 'No. Proc.')
    args = parser.parse_args()
    transport(chosen_integrator = args.m, n_proc = args.n)
