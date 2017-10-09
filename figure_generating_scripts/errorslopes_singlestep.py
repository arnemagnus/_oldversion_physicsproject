import numpy as np
from matplotlib import pyplot as plt

path = '../calculation_scripts/particlepositions_errorestimation/'

ref = np.load(path + 'rkdp87_tol=1e-12.npy')

tols = np.logspace(-10, 0, num=11)

rkck45_errs = np.zeros(np.size(tols))
rkbs32_errs = np.copy(rkck45_errs)
rkbs54_errs = np.copy(rkck45_errs)
rkdp54_errs = np.copy(rkck45_errs)
rkdp87_errs = np.copy(rkck45_errs)
rkf45_errs = np.copy(rkck45_errs)

for j in range(np.size(tols)):
    rkck45_sol = np.load(path + 'rkck45_tol={}.npy'.format(tols[j]))
    rkck45_errs[j] = np.mean(np.sqrt((rkck45_sol[0,:] - ref[0,:])**2 + (rkck45_sol[1,:]-ref[1,:])**2))
    rkbs32_sol = np.load(path + 'rkbs32_tol={}.npy'.format(tols[j]))
    rkbs32_errs[j] = np.mean(np.sqrt((rkbs32_sol[0,:] - ref[0,:])**2 + (rkbs32_sol[1,:]-ref[1,:])**2))
    rkbs54_sol = np.load(path + 'rkbs54_tol={}.npy'.format(tols[j]))
    rkbs54_errs[j] = np.mean(np.sqrt((rkbs54_sol[0,:] - ref[0,:])**2 + (rkbs54_sol[1,:]-ref[1,:])**2))
    rkdp54_sol = np.load(path + 'rkdp54_tol={}.npy'.format(tols[j]))
    rkdp54_errs[j] = np.mean(np.sqrt((rkdp54_sol[0,:] - ref[0,:])**2 + (rkdp54_sol[1,:]-ref[1,:])**2))
    rkdp87_sol = np.load(path + 'rkdp87_tol={}.npy'.format(tols[j]))
    rkdp87_errs[j] = np.mean(np.sqrt((rkdp87_sol[0,:] - ref[0,:])**2 + (rkdp87_sol[1,:]-ref[1,:])**2))
    rkf45_sol = np.load(path + 'rkf45_tol={}.npy'.format(tols[j]))
    rkf45_errs[j] = np.mean(np.sqrt((rkf45_sol[0,:] - ref[0,:])**2 + (rkf45_sol[1,:]-ref[1,:])**2))

plt.rc('text', usetex = True)

plt.figure(figsize=(12,5), dpi = 300)
plt.loglog(tols, rkbs32_errs, 'o', fillstyle = 'none', label = 'B-S 3(2)')
plt.loglog(tols, rkbs54_errs, 'v', fillstyle = 'none', label = 'B-S 5(4)')
plt.loglog(tols, rkck45_errs, 's', fillstyle = 'none', label = 'C-K 4(5)')
plt.loglog(tols, rkdp54_errs, 'p', fillstyle = 'none', label = 'D-P 5(4)')
plt.loglog(tols, rkdp87_errs, '*', fillstyle = 'none', label = 'D-P 8(7)')
plt.loglog(tols, rkf45_errs, 'X', fillstyle = 'none', label = 'F 4(5)')
plt.xlabel('Tolerance level')
plt.ylabel('Mean error')
plt.legend(loc = 'best', fontsize = '8')
plt.show()

