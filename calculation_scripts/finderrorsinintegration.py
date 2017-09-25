import numpy as np

times = np.linspace(0.5,5,10,endpoint=True)

ref_integrator = 'rk4'

integrators = ['euler','rk2','rk3','rk4']

dt = [10**(-(i+1)) for i in range (3)]

ref_dt = dt[-1]

dx = 0.005

outfilepath = 'error_estimation_debug/'





for integrator in integrators:
    outfilename = 'lyap_{}_dx={}.txt'.format(integrator, dx)
    textbuf = ''
    for t in times:
        ref = np.loadtxt(fname=('datadump_debug/'
                       + 'lyapunov_{}_t={}_dt={}_dx={}'.format(ref_integrator,
                                                                t,
                                                                ref_dt,
                                                                dx)
                       + '.txt'
                       )
                )
        for i in range(len(dt)-(ref_integrator[:]==integrator[:])):
            foo = np.loadtxt(fname=('datadump_debug/'
                                + 'lyapunov_{}_t={}_dt={}_dx={}'.format(integrator,
                                                                        t,
                                                                        dt[i],
                                                                        dx)
                                + '.txt'
                                )
                        )
            textbuf += (('Error of {} with dt = {} with {} solution for '
               'dt = {} as reference, dx = {}, at t = {}:\n'.format(integrator,
                                                                    dt[i],
                                                                    ref_integrator,
                                                                    dt[-1],
                                                                    dx,
                                                                    t)
              )
             )
            textbuf +=  (('\t\tAverage abs. error: {}\n'.format(np.mean(np.absolute(foo-ref)))))
            textbuf +=  (('\t\tMaximum abs. error: {}\n\n'.format(np.amax(np.absolute(foo-ref)))))

            textbuf += '-'*100 + '\n\n'
        np.savetxt(fname=(outfilepath+outfilename), X=[textbuf], fmt='%s')

