# -*- coding=utf-8 -*-
from __future__ import division
def simulation(N):



    from velocity_field import vel


    from numerical_integrators.single_step import euler, rk2, rk3, rk4


    from matplotlib import pyplot as plt


    import numpy as np

    import multiprocessing as mp
    import os
    os.system("taskset -p 0xff %d"% os.getpid())
    t_min, t_max = 0, 5


    h = 0.1


    x_min, x_max = 0, 2

    y_min, y_max = 0, 1




    Ny = 201


    Nx = 1 + int(np.floor(Ny-1) * (x_max - x_min) / (y_max - y_min))


    x = np.linspace(x_min, x_max, Nx)

    y = np.linspace(y_min, y_max, Ny)


    xy, yx = np.meshgrid(x, y)
    xy = np.rot90(xy, 2)
    yx = np.rot90(yx, 2)
    dx = (x_max - x_min) / (Nx - 1)

    dy = (y_max - y_min) / (Ny - 1)


    left_off = np.ones(np.shape(xy))

    right_off = np.ones(np.shape(xy))


    top_off = np.ones(np.shape(xy))

    bott_off = np.ones(np.shape(xy))



    lyap = np.zeros(np.shape(xy))


    n_snaps = 1


    t_tot = t_max - t_min


    t_incr = t_tot / n_snaps


    def timestep(t, xy, yx, h, deriv, integrator):

        t, (xy, yx), h = integrator(t, np.array([xy, yx]), h, deriv)


        return t, xy, yx, h


    t = t_min

    def simulate_all_the_way(t, t_end, xy, yx, h, deriv, integrator, q):
        while t < t_end:

            t, xy, yx, h = timestep(t, xy, yx, h, deriv, integrator)

            h = np.minimum(h, t_end - t)

        q.put(np.array((xy, yx)))


    integrator = rk3

    h_ref = np.copy(h)

    plt.figure(figsize = (10, int(10 * (Ny - 1)/(Nx - 1))), dpi = 300)

    part_x = np.floor(Nx/N).astype(int)
    
    for j in xrange(N):
        print(j*part_x, Nx if j + 1 == N else (j + 1)*part_x - 1)

    for i in xrange(n_snaps):

        queuelist = [mp.Queue() for j in xrange(N)]
        processlist = [mp.Process(target = simulate_all_the_way,
                                  args = (t_min + i * t_incr,
                                          t_min + (i+1)*t_incr,
                                          xy[j*part_x:Nx if j + 1 == N else (j+1)*part_x,:],
                                          yx[j*part_x:Nx if j + 1 == N else (j+1)*part_x,:],
                                          h,
                                          vel,
                                          integrator,
                                          queuelist[j])) for j in xrange(N)]

        for process in processlist:
            process.start()


        for j in xrange(N):
            xy[j*part_x:Nx if j + 1 == N else (j + 1)*part_x,:], yx[j*part_x:Nx if j + 1 == N else (j + 1)*part_x,:] = queuelist[j].get()

        for process in processlist:
            process.join()
       
      


        left_off[1:-2,1:-2] = np.sqrt((xy[0:-3,1:-2]-xy[1:-2,1:-2])**2
                                    +(yx[0:-3,1:-2]-yx[1:-2,1:-2])**2)

        right_off[1:-2,1:-2] = np.sqrt((xy[2:-1,1:-2]-xy[1:-2,1:-2])**2
                                     +(yx[2:-1,1:-2]-yx[1:-2,1:-2])**2)


        top_off[1:-2,1:-2] = np.sqrt((xy[1:-2,0:-3]-xy[1:-2,1:-2])**2
                                   +(yx[1:-2,0:-3]-yx[1:-2,1:-2])**2)

        bott_off[1:-2,1:-2] = np.sqrt((xy[1:-2,2:-1]-xy[1:-2,1:-2])**2
                                      +(yx[1:-2,2:-1]-yx[1:-2,1:-2])**2)


        lyap = np.fmax(np.log(np.fmax(left_off,right_off)
                  /dx)/(t_min+(i+1)*t_incr),
              np.log(np.fmax(top_off,bott_off)
                  /dy)/(t_min+(i+1)*t_incr)
                  )


        plt.pcolormesh(xy[1:-2,1:-2], 1- yx[1:-2,1:-2], lyap[1:-2,1:-2], cmap = 'RdBu_r')

        plt.colorbar()
#    plt.title(r'$t =$ {}, {},  $\Delta{t} =$ {}, $\Delta{x} =$ {}'.format(t,
 #                                                                         integrator.__name__,
  #                                                                        h,
   #                                                                       dx)
#         )
        #plt.gca().invert_xaxis()
   #     plt.gca().invert_yaxis()
        plt.savefig('figure_debug/' +
                'lyapunov_{}_N={}_t={}_dt={}_dx={}'.format(integrator.__name__,
                                                      N,
                                                           t_min+(i+1)*t_incr, h_ref, dx)
                + '.png')

        plt.clf()
                                      
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=('Kake?'))
    parser.add_argument('-N', type=int, default=1, help = 'No. Proc.')
    args = parser.parse_args()
    simulation(N = args.N)
