# Changelog:
#     2017-08-30: File created. Calculation not yet fully implemented.
#
#     2017-08-31: Calculation yields what appear to be reasonable
#                 results, although as of this moment I have yet
#                 to find a way of verifying the numerical solutions.
#
#                 Attempts to speed up the calculations by use of
#                 numba have, as of yet, not resulted in significant
#                 speedup, if at all.
#
#     2017-09-01: Changed function calls to the numerical integrator,
#                 in accordance with the recent change in function call
#                 signature [f(x,t) --> f(t,x)]
#
#     2017-09-06: Altered both the call signature and return variables
#                 from the timestep(...) function, in accordance with
#                 the recent change in function call signatures for
#                 the numerical integrators, cf. 2017-09-01.
#
#                 Changed the way the simulation procedure handles
#                 both the timestep procedure (for[for] --> for[while])
#                 and the snapshot process. The new way of stepping
#                 forwards in time was designed with adaptive timestep
#                 integrators in mind. Nevertheless, such integrators
#                 don't work with the current iteration.
#
#                 The challenge is that each trajectory will need
#                 different timesteps, so we probably need time and
#                 timestep variables for each and every one of them.
#                 There's also the issue of the quickly simulated
#                 trajectories potentially having long idle times,
#                 waiting for the slower ones - nevertheless,
#                 as far as adaptive timestep integrators go, the
#                 way forward seems to be parallelization.

# Written by Arne M. T. Løken as part of a specialization project in
# physics at NTNU, fall 2017.

#---------------------------------------------------------------------#

# First of all, I import the function definition for the velocity
# field, and the numerical integrator.
#
# To begin with, I use the RK4 integration scheme; a higher-order
# numerical method, and perhaps the one we need to use as a reference
# when analyzing the dependence of the transport calculations on
# integration scheme.

from velocity_field import vel
from numerical_integrators.single_step import euler, rk2, rk3, rk4
from numerical_integrators.adaptive_step import rkhe21, rkdp54

# To speed up the timestep loop:
#from numba import jit

# For plotting purposes:
import matplotlib.pyplot as plt

# For mathematical ease-of-use:
import numpy as np

# The goal here is to find an estimate of the finite-time Lyapunov
# exponents for the velocity field. To this end, we generate a
# quadratic grid of fluid elements at t = 0, covering the domain of
# the velocity field, and transporting all of the fluid elements with
# the velocity field until t = t_max (TBD)

# We consider the 2-norm of the distance between each fluid element and
# its initial nearest neighbors. Per definition, the neighbors whose
# trajectories diverge the quickest have the largest Lyapunov exponent.
# For our purposes, the largest local Lyapunov exponent is of interest.
# Thus, for each internal fluid element, we assign the largest
# Lyapunov exponent.

# First, let's define the time interval of interest:
t_min, t_max = 0, 5

# We need to define the timestep for the transport calculation:
h = 0.1

# In the event that we want to use adaptive timestep integrators,
# or, as is the case in the current edition of the program, we
# want to generate snapshots of the Lyapunov field at fixed time
# levels, we will end up overwriting the variable h.
# To ensure predictable behaviour, we keep a reference copy of the
# initial timestep:
h_ref = np.copy(h)

# Let's define the domain of the velocity field:
x_min, x_max = 0, 2
y_min, y_max = 0, 1

# Seeing as we desire a quadratic grid, the number of grid points
# in either directions will be codependent. I choose the number of
# grid points in the x-direction to be the dependent variable. Hence:
Ny = 201
Nx = 1 + int(np.floor(Ny-1)*(x_max-x_min)/(y_max-y_min))

# With the above definition, we will get a quadratic grid.
# Regardless of the integration scheme, we will need to keep hold of
# the x- and y-coordinates of each fluid element at each timestep.
# For this purpose, I choose to use meshgrids:

x0 = np.linspace(x_min, x_max, Nx)
y0 = np.linspace(y_min, y_max, Ny)

x = np.zeros(Nx*Ny)
y = np.copy(x)

for j in range(Ny):
    x[j*Nx:(j+1)*Nx] = x0
    y[j*Nx:(j+1)*Nx] = y0[j]

pos = np.array([x, y])

# To begin with, we're interested in finding the FTLE, so finding
# an estimate at each time instance seems like a reasonable approach.
# Seeing as we're using an equidistant grid, there's no need to keep
# hold of the coordinates of the fluid elements. By the definition
# of the FTLE, we need the initial separation between neighboring
# fluid elements, and their separation at time instant t, to find
# the FTLE at time instant t.

dx = (x_max - x_min)/(Nx - 1)
dy = (y_max - y_min)/(Ny - 1)

# Estimating the FTLE by finding the largest divergence in trajectories
# between a fluid element and its nearest neighbors will not work.
# As a first order approximation, we assume that the velocity field
# is zero along the domain edges, so that the FTLE is assigned the
# value zero. We take care of the coupling with the internal grid
# points in estimating the maximal FTLE for those points.

# We preallocate containers for the comparison of trajectory offsets,
# for fluid elements that start out as nearest neighbors. We
# initialize with ones(:) rather than zeros(:) to avoid headaches when
# we take the logarithm later.

left_offset = np.zeros([Nx, Ny])
right_offset = np.copy(left_offset)

top_offset = np.copy(left_offset)
bottom_offset = np.copy(left_offset)

# We also preallocate a container for the FTLE estimate:
lyap = np.copy(left_offset)

# Now, we step forward in time. At each time instant, we must find the
# largest FTLE estimate, based on trajectory divergence between each
# fluid element and its initial nearest neighbors.

# As a first approximation, we take a set number of snapshots of the
# Lyapunov field:
n_snaps = 1

# A rather straightforward way to do this, is determining the total
# time increment we want to simulate;
t_tot = t_max - t_min

# ... then, dividing the total time increment by the number of
# snapshots we want to generate:
t_incr = t_tot / n_snaps

# Defining a function to move the calculation process from inside
# the timeloop seems to be a good idea, as it opens up for use
# of just-in-time compilation, among other things.

def timestep(t,            # Current time level
             pos,
             h,            # Timestep
             deriv,        # Function handle for the derivatives,
                           # in our case, the velocity field
             integrator    # Function handle for the numerical
                           # integrator to use, e.g., 'euler' or 'rk4'
             ):

   # All numerical integrators return the following variables:
   #    t:    New time level       (for adaptive timestep integrators,
   #                                the time level is only updated if
   #                                the trial step is accepted)
   #    x:    New coordinate array (for adaptive timestep integrators,
   #                                the coordinates are only updated
   #                                if the trial step(s) is accepted)
   #    h:    Timestep             (subject to change in adaptive
   #                                timestep integrators, otherwise
   #                                unaltered)
   t, pos, h = integrator(t,
                              pos,
                              h,
                              deriv, atol = 1e-2, rtol = 1e-2
                              )
   # We return the new time level, the updated coordinates and
   # the (updated) timestep.
   return t, pos, h


# We need a container for the current simulation time:
t = t_min
ts = np.ones(Nx*Ny)*t_min

# We need a container for the current time step:
hs = np.ones(np.shape(ts))*h_ref

# We need to choose a numerical integrator:
integrator = rkhe21

# Lastly, we need a canvas:
plt.figure(figsize=(10, 5), dpi = 300)


# Now, we're ready to step forwards in time:

# First, we loop over the number of snapshots we want to generate:
for i in range(n_snaps):
    # We step forwards in time from one snapshot to the next:
    counter = 0
    while np.any(ts < t_min + (i+1)*t_incr):
       counter+=1
       hs = np.minimum(hs, t_min + (i+1)*t_incr - ts)
       ts, pos, hs = timestep(ts, pos, hs, vel, integrator)

    print(counter)
    # Because we force the snapshots to be generated at fixed time
    # levels, we must perform an explicit check as to whether each
    # timestep will result in overshootiing.
    # Seeing as the timestep will be modified, catering to our
    # predetermined snapshot times, we have to reinitialize the
    # timestep after every snapshot. Hence:
    hs = np.ones(np.shape(hs))*h_ref

    xy = pos[0].reshape(Nx, Ny)
    yx = pos[1].reshape(Nx, Ny)

    left_offset[1:-2,1:-2] = np.sqrt((xy[0:-3,1:-2]-xy[1:-2,1:-2])**2
                                    +(yx[0:-3,1:-2]-yx[1:-2,1:-2])**2)

    right_offset[1:-2,1:-2] = np.sqrt((xy[2:-1,1:-2]-xy[1:-2,1:-2])**2
                                     +(yx[2:-1,1:-2]-yx[1:-2,1:-2])**2)


    top_offset[1:-2,1:-2] = np.sqrt((xy[1:-2,0:-3]-xy[1:-2,1:-2])**2
                                   +(yx[1:-2,0:-3]-yx[1:-2,1:-2])**2)

    bottom_offset[1:-2,1:-2] = np.sqrt((xy[1:-2,2:-1]-xy[1:-2,1:-2])**2
                                     +(yx[1:-2,2:-1]-yx[1:-2,1:-2])**2)

    lyap[1:-2,1:-2] = np.fmax(np.log(np.fmax(left_offset[1:-2,1:-2],right_offset[1:-2,1:-2])
                  /dx)/(t_min+(i+1)*t_incr),
                  np.log(np.fmax(top_offset[1:-2,1:-2],bottom_offset[1:-2,1:-2])
                  /dy)/(t_min+(i+1)*t_incr)
                  )


    # We plot the calculated FTLE field:
    #plt.pcolormesh(xy_ref,yx_ref,lyap,cmap='RdBu_r')
    plt.pcolormesh(xy[1:-2,1:-2],yx[1:-2,1:-2],  lyap[1:-2,1:-2], cmap='RdBu_r')

    plt.colorbar()
    plt.title(r'$t=$ {}, {}, dt = {}, dx = {}'.format(t_min + (i+1)*t_incr,
                                                      integrator.__name__,
                                                      h,
                                                      dx
                                                      )
             )
    plt.savefig('figure_debug/' +
                'lyapunov_{}_t={}_dt={}_dx={}'.format(integrator.__name__,
                                                      t_min + (i+1)*t_incr,
                                                      h,
                                                      dx
                                                      )
                 + '.png'
               )

    # We clear the canvas, preparing for the next snapshot:
    plt.clf()




    # Dump current Lyapunov field to text file, enabling error
    # estimation wrt. a reference (i.e. higher order) solution;
    np.savetxt(fname=('datadump_debug/'
                     + 'lyapunov_{}_t={}_dt={}_dx={}'.format(integrator.__name__,
                                                             t,
                                                             h,
                                                             dx
                                                            )
                     +'.txt'
                    ),
                X=lyap
              )


