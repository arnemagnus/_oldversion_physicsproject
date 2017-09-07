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
from numerical_integrators import *

# To speed up the timestep loop:
from numba import jit

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
xmin, xmax = 0, 2
ymin, ymax = 0, 1

# Seeing as we desire a quadratic grid, the number of grid points
# in either directions will be codependent. I choose the number of
# grid points in the x-direction to be the dependent variable. Hence:
Ny = 201
Nx = 1 + int(np.floor(Ny-1)*(xmax-xmin)/(ymax-ymin))

# With the above definition, we will get a quadratic grid.
# Regardless of the integration scheme, we will need to keep hold of 
# the x- and y-coordinates of each fluid element at each timestep.
# For this purpose, I choose to use meshgrids:

xy, yx = np.meshgrid(np.linspace(xmin,xmax,Nx), np.linspace(ymin,ymax,Ny))

# Keep a copy of the initial fluid element grid, in order to
# correctly visualize the Lyapunov field:

xy_ref, yx_ref = np.copy(xy), np.copy(yx)

# To begin with, we're interested in finding the FTLE, so finding 
# an estimate at each time instance seems like a reasonable approach.
# Seeing as we're using an equidistant grid, there's no need to keep
# hold of the coordinates of the fluid elements. By the definition 
# of the FTLE, we need the initial separation between neighboring
# fluid elements, and their separation at time instant t, to find
# the FTLE at time instant t.

dx = (xmax - xmin)/(Nx - 1)
dy = (ymax - ymin)/(Ny - 1)

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

left_offset = np.ones(xy.shape)
right_offset = np.ones(xy.shape)

top_offset = np.ones(xy.shape)
bottom_offset = np.ones(xy.shape)

# We also preallocate a container for the FTLE estimate:
lyap = np.zeros(xy.shape)

# Now, we step forward in time. At each time instant, we must find the
# largest FTLE estimate, based on trajectory divergence between each
# fluid element and its initial nearest neighbors.

# As a first approximation, we take a set number of snapshots of the
# Lyapunov field:
n_snaps = 10

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
             xy,           # (MxN) meshgrid of x-coordinates for the
                           # fluid elements at the current time level
             yx,           # -------''------- y-coordinates ----''----
                           # ----------------''----------------
             h,            # Timestep
             deriv,        # Function handle for the derivatives,
                           # in our case, the velocity field
             integrator,   # Function handle for the numerical
                           # integrator to use, e.g., 'euler' or 'rk4'
             lyap,         # (MxN) container array for the Lyapunov
                           # exponent (overwritten)
             dx,           # Initial offset between nearest neighbor
                           # fluid elements, in the x-direction
             dy,           # -------------------''-------------------
                           # ---------''----------  y-direction
             left_offset,  # (MxN) container array for trajectory 
                           # offsets for each fluid element's initial
                           # nearest neighbor to the left
             right_offset, # ------------------''--------------------
                           # ------------------''--------------------
                           # ----------''----------- right
             top_offset,   # ------------------''--------------------
                           # ------------------''--------------------
                           # nearest neighbor above
             bottom_offset # ------------------''--------------------
                           # ------------------''--------------------
                           # ---.---''------- beneath
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
   t,(xy, yx), h = integrator(t,
                              np.array([xy, yx]),
                              h,
                              deriv
                             )
   left_offset[1:-2,1:-2] = np.sqrt((xy[0:-3,1:-2]-xy[1:-2,1:-2])**2
                                    +(yx[0:-3,1:-2]-yx[1:-2,1:-2])**2)
   right_offset[1:-2,1:-2] = np.sqrt((xy[2:-1,1:-2]-xy[1:-2,1:-2])**2
                                     +(yx[2:-1,1:-2]-yx[1:-2,1:-2])**2)

   top_offset[1:-2,1:-2] = np.sqrt((xy[1:-2,0:-3]-xy[1:-2,1:-2])**2
                                   +(yx[1:-2,0:-3]-yx[1:-2,1:-2])**2)
   bottom_offset[1:-2,1:-2] = np.sqrt((xy[1:-2,2:-1]-xy[1:-2,1:-2])**2
                                      +(yx[1:-2,2:-1]-yx[1:-2,1:-2])**2)

   lyap = np.fmax(np.log(np.fmax(left_offset,right_offset))
                      /np.log(dx)/t,
                  np.log(np.fmax(top_offset,bottom_offset))
                      /np.log(dy)/t
                  )
   # We return the new time level, the updated coordinates,
   # the timestep and the calculated lyapunov field.
   # The time level and timestep can be useful if adaptive timestep
   # integrators are used.
   return t, xy, yx, h, lyap


# We need a container for the current simulation time:
t = t_min

# We need to choose a numerical integrator:
integrator = rk3

# Lastly, we need a canvas:
plt.figure()

# Now, we're ready to step forwards in time:

# First, we loop over the number of snapshots we want to generate:
for i in range(n_snaps):
    # We step forwards in time from one snapshot to the next:
    while t < (t_min +(i+1)*t_incr):
        t, xy, yx, h, lyap = timestep(t,
                                      xy,
                                      yx,
                                      h,
                                      vel,
                                      integrator,
                                      lyap,
                                      dx,
                                      dy,
                                      left_offset,
                                      right_offset,
                                      top_offset,
                                      bottom_offset
                                    )
        h = np.minimum(h, t_min + (i+1)*t_incr - t)
    # Because we force the snapshots to be generated at fixed time
    # levels, we must perform an explicit check as to whether each
    # timestep will result in overshootiing.
    # Seeing as the timestep will be modified, catering to our
    # predetermined snapshot times, we have to reinitialize the
    # timestep after every snapshot. Hence:
    h = h_ref

    # We plot the calculated FTLE field:
    plt.pcolormesh(xy_ref,yx_ref,lyap,cmap='RdBu_r')
    plt.colorbar()
    plt.title(r'$t=$ {}, {}, dt = {}, dx = {}'.format(t,
                                                      integrator.__name__,
                                                      h,
                                                      dx
                                                      )
             )
    plt.savefig('figure_debug/' +
                'lyapunov_{}_t={}_dt={}_dx={}'.format(integrator.__name__,
                                                      t,
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
       
                      
