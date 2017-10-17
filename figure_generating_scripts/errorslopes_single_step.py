import numpy as np
from matplotlib import pyplot as plt

from itertools import cycle

marker_tols = cycle(('.', 'x', 'h', '*', 'o', '+'))
marker_evals = cycle(('.', 'x', 'h', '*', 'o', '+'))


plt.rc('text', usetex = True)

figwidth = 10

resolution = 300 # Forced dpi

path = '../calculation_scripts/hpc-stuff/integrator_efficiency/runs/'

reference = np.load(path + 'endpos_rkdp87_tol=1e-12.npy')

steplengths = np.load(path + 'steplengths_single.npy')

integrators = ['euler', 'rk2', 'rk3', 'rk4']

integrator_plotnames = [r'Euler', r'Heun', r'Kutta', r'RK4']

plt.figure('tolerance', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)

plt.figure('evaluations', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)



for i, name in enumerate(integrators):
    mean_evals = np.empty(np.size(steplengths))
    mean_errors = np.empty(np.size(steplengths))
    for j, h in enumerate(steplengths):
        endpos = np.load(path + '{}/endpos_{}_h={}.npy'.format(name, name, h))
        evaluations = np.load(path + '{}/evaluations_{}_h={}.npy'.format(name, name, h))

        mean_evals[j] = np.mean(evaluations)
        mean_errors[j] = np.mean(np.sqrt((reference[0,:] - endpos[0,:])**2 + (reference[1,:] - endpos[1,:])**2))

    plt.figure('tolerance')
    plt.loglog(steplengths, mean_errors, label = integrator_plotnames[i], marker = next(marker_tols), ms = 6, fillstyle = 'none')

    plt.figure('evaluations')
    plt.semilogy(mean_evals, mean_errors, label = integrator_plotnames[i], marker = next(marker_evals), ms = 6, fillstyle = 'none')


plt.figure('tolerance')
plt.xlabel('Steplength')
plt.ylabel('Mean error')
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('single1.png')

plt.figure('evaluations')
plt.xlabel('Function evaluations')
plt.ylabel('Mean error')
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('single2.png')

