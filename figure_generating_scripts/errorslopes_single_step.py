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

integrator_plotnames = [r'Euler', r'Heun', r'Kutta', r'RK4']

euler_errs = np.zeros(np.size(steplengths))
rk2_errs = np.copy(euler_errs)
rk3_errs = np.copy(euler_errs)

euler_steps = np.zeros(np.size(steplengths), dtype=int)
rk2_steps = np.copy(euler_steps)
rk3_steps = np.copy(euler_steps)

for i, h in enumerate(steplengths):
    euler_sol = np.load(path + 'euler/endpos_euler_h={}.npy'.format(h))
    euler_stepcount = np.load(path + 'euler/evaluations_euler_h={}.npy'.format(h))

    euler_errs[i] = np.mean(np.sqrt((euler_sol[0,:] - reference[0,:])**2 + (euler_sol[1,:] - reference[1,:])**2))
    euler_steps[i] = np.mean(euler_stepcount).astype(int)

    rk2_sol = np.load(path + 'rk2/endpos_rk2_h={}.npy'.format(h))
    rk2_stepcount = np.load(path + 'rk2/evaluations_rk2_h={}.npy'.format(h))

    rk2_errs[i] = np.mean(np.sqrt((rk2_sol[0,:] - reference[0,:])**2 + (rk2_sol[1,:] - reference[1,:])**2))
    rk2_steps[i] = np.mean(rk2_stepcount).astype(int)

    rk3_sol = np.load(path + 'rk3/endpos_rk3_h={}.npy'.format(h))
    rk3_stepcount = np.load(path + 'rk3/evaluations_rk3_h={}.npy'.format(h))

    rk3_errs[i] = np.mean(np.sqrt((rk3_sol[0,:] - reference[0,:])**2 + (rk3_sol[1,:] - reference[1,:])**2))
    rk3_steps[i] = np.mean(rk3_stepcount).astype(int)

plt.figure(figsize = (figwidth, np.round(figwidth*2/3).astype(int)), dpi = resolution)
plt.loglog(steplengths, euler_errs, marker = next(marker_tols), label = integrator_plotnames[0], ms = 6, fillstyle = 'none')

plt.loglog(steplengths, rk2_errs, marker = next(marker_tols), label = integrator_plotnames[1], ms = 6, fillstyle = 'none')


plt.loglog(steplengths, rk3_errs, marker = next(marker_tols), label = integrator_plotnames[2], ms = 6, fillstyle = 'none')

plt.xlabel('Steplength')
plt.ylabel('Mean error')
plt.legend()
plt.tight_layout()
plt.savefig('jallah.png')
#integrators = ['euler', 'rk2', 'rk3', 'rk4']

#integrator_plotnames = [r'Euler', r'Heun', r'Kutta', r'RK4']

#plt.figure('tolerance', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)

#plt.figure('evaluations', figsize = (figwidth, np.round(figwidth * 2/3).astype(int)), dpi = resolution)



#for i, name in enumerate(integrators):
#    mean_evals = np.empty(np.size(steplengths))
#    mean_errors = np.empty(np.size(steplengths))
#    for j, h in enumerate(steplengths):
#        endpos = np.load(path + '{}/endpos_{}_h={}.npy'.format(name, name, h))
#        evaluations = np.load(path + '{}/evaluations_{}_h={}.npy'.format(name, name, h))
#
#        mean_evals[j] = np.mean(evaluations)
#        mean_errors[j] = np.mean(np.sqrt((reference[0,:] - endpos[0,:])**2 + (reference[1,:] - endpos[1,:])**2))
#
#    plt.figure('tolerance')
#    plt.loglog(steplengths, mean_errors, label = integrator_plotnames[i], marker = next(marker_tols), ms = 6, fillstyle = 'none')
#
#    plt.figure('evaluations')
#    plt.semilogy(mean_evals, mean_errors, label = integrator_plotnames[i], marker = next(marker_evals), ms = 6, fillstyle = 'none')
#
#
#plt.figure('tolerance')
#plt.xlabel('Steplength')
#plt.ylabel('Mean error')
#plt.legend(loc = 'best')
#plt.tight_layout()
#plt.savefig('single1.png')
#
#plt.figure('evaluations')
#plt.xlabel('Function evaluations')
#plt.ylabel('Mean error')
#plt.legend(loc = 'best')
#plt.tight_layout()
#plt.savefig('single2.png')

