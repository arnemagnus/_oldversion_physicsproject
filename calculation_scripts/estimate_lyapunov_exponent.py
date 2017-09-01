# Changelog:
#     2017-08-30: File created. Calculation not yet fully implemented.
#     2017-08-31: Calculation yields what appear to be reasonable
#                 results, although as of this moment I have yet
#                 to find a way of verifying the numerical solutions.
#                 Attempts to speed up the calculations by use of 
#                 numba have, as of yet, not resulted in significant
#                 speedup, if at all.
#     2017-09-01: Changed function calls to the numerical integrator,
#                 in accordance with the recent change in function call
#                 signature [f(x,t) --> f(t,x)]

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
# the velocity field until t = tmax (TBD)

# We consider the 2-norm of the distance between each fluid element and 
# its initial nearest neighbors. Per definition, the neighbors whose
# trajectories diverge the quickest have the largest Lyapunov exponent.
# For our purposes, the largest local Lyapunov exponent is of interest.
# Thus, for each internal fluid element, we assign the largest
# Lyapunov exponent. 

# First, let's define the time interval of interest:

tmin, tmax = 0, 5

# We must define the timestep to be used in the transport calculation:

h = 0.01

# Let's define the domain of the velocity field:

xmin, xmax = 0, 2
ymin, ymax = 0, 1

# Seeing as we desire a quadratic grid, the number of grid points
# in either directions will be codependent. I choose the number of
# grid points in the x-direction to be the dependent variable. Hence:

Ny = 101
Nx = 1 + int(np.floor(Ny-1)*(xmax-xmin)/(ymax-ymin))

# With the above definition, we will get a quadratic grid.
# Regardless of the integration scheme, we will need to keep hold of 
# the x- and y-coordinates of each fluid element at each timestep.
# For this purpose, I choose to use meshgrids:

xy, yx = np.meshgrid(np.linspace(xmin,xmax,Nx), np.linspace(ymin,ymax,Ny))

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

# We preallocate containers for the offset trajectories starting off
# as nearest neighbors. We initialize with ones(:) rather than zeros(:)
# to avoid headaches when we take the logarithm later.

left_offset = np.ones(xy.shape)
right_offset = np.ones(xy.shape)

top_offset = np.ones(xy.shape)
bottom_offset = np.ones(xy.shape)

# We also preallocate a container for the FTLE estimate:

lyap = np.zeros(xy.shape)

# Now, we step forward in time. At each time instant, we must find the
# largest FTLE estimate, based on trajectory divergence between each
# fluid element and its initial nearest neighbors.

# As a first approximation, we take snapshots at integer multiples
# of tenths of the total amount of simulation steps. 
# The easy way to do this, is by storing the number of steps we
# want to take:

n_steps = int(np.ceil((tmax-tmin)/h))
n_tenth = int(n_steps/10)

# Defining a function to move the calculation process from inside
# the timeloop seems to be a good idea, as it enables just-in-time
# compilation, among other things.

plt.figure()

# To avoid an if statement inside the upcoming timestep loop, we 
# make an interated loop instead. That is, the outer loop sets the
# limits for the inner loop, as follows:
@jit
def timestep(xy,yx,lyap,t,h,deriv,integrator):
   global left_offset, right_offset, top_offset, bottom_offset
   xy, yx = integrator(t,
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
                  /np.log(dx)/((n_tenth*i+j+1)*h),
                      np.log(np.fmax(top_offset,bottom_offset))
                      /np.log(dy)/((n_tenth*i+j+1)*h)
                  )

   return xy, yx, lyap

integrator = euler

for i in range(int(n_steps/n_tenth)):
   for j in range(n_tenth):
      xy, yx, lyap = timestep(xy, yx, lyap, (i*n_tenth+j)*h,h,vel,integrator=integrator)
   plt.pcolormesh(xy,yx,lyap,cmap='RdBu_r')
   plt.colorbar()
   plt.title(r'$t=$ {}'.format(tmin+(i*n_tenth+j)*h))
   plt.savefig('figure_debug/lyapunov_{}_t={}.png'.format(integrator,tmin+(i*n_tenth+j)*h))
   plt.clf()
