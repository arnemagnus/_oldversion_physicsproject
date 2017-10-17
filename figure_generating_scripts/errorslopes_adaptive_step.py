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

tolerances = np.load(path + 'tolerances_adaptive.npy')

integrators = ['rkbs32', 'rkbs54', 'rkdp54', 'rkdp87', 'rkf45', 'rkf78']

integrator_plotnames = [r'B-S 3(2)', r'B-S 5(4)', r'D-P 5(4)', r'D-P 8(7)', r'F 4(5)', r'F 7(8)']

plt.figure('tolerance', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)

plt.figure('evaluations', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)



for i, name in enumerate(integrators):
    mean_evals = np.empty(np.size(tolerances))
    mean_errors = np.empty(np.size(tolerances))
    for j, tol in enumerate(tolerances):
        endpos = np.load(path + '{}/endpos_{}_tol={}.npy'.format(name, name, tol))
        evaluations = np.load(path + '{}/evaluations_{}_tol={}.npy'.format(name, name, tol))

        mean_evals[j] = np.mean(evaluations)
        mean_errors[j] = np.mean(np.sqrt((reference[0,:] - endpos[0,:])**2 + (reference[1,:] - endpos[1,:])**2))

    plt.figure('tolerance')
    plt.loglog(tolerances, mean_errors, label = integrator_plotnames[i], marker = next(marker_tols), ms = 6, fillstyle = 'none')

    plt.figure('evaluations')
    plt.semilogy(mean_evals, mean_errors, label = integrator_plotnames[i], marker = next(marker_evals), ms = 6, fillstyle = 'none')


plt.figure('tolerance')
plt.xlabel('Tolerance')
plt.ylabel('Mean error')
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('adaptive1.png')

plt.figure('evaluations')
plt.xlabel('Function evaluations')
plt.ylabel('Mean error')
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('adaptive2.png')

plt.figure('evaluations_sans_BS32', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)


markers_evals_sans_BS32 = cycle(('.', 'x', 'h', '*', 'o', '+'))
plt.plot([],[], label = '' )


#default_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

#plt.gca().set_prop_cycle(default_cycle[1::])

for i, name in enumerate(integrators[1::]):
    mean_evals = np.empty(np.size(tolerances))
    mean_errors = np.empty(np.size(tolerances))
    for j, tol in enumerate(tolerances):
        endpos = np.load(path + '{}/endpos_{}_tol={}.npy'.format(name, name, tol))
        evaluations = np.load(path + '{}/evaluations_{}_tol={}.npy'.format(name, name, tol))

        mean_evals[j] = np.mean(evaluations)
        mean_errors[j] = np.mean(np.sqrt((reference[0,:] - endpos[0,:])**2 + (reference[1,:] - endpos[1,:])**2))
    plt.semilogy(mean_evals, mean_errors, label = integrator_plotnames[1::][i], marker = next(markers_evals_sans_BS32), ms = 6, fillstyle = 'none')


plt.xlabel('Function evaluations')
plt.ylabel('Mean error')
#plt.xlim([min_evals, max_evals_sans_BS32])
plt.legend(loc = 'best')
plt.tight_layout()
plt.savefig('adaptive3.png')
