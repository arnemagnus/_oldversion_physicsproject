# Changelog:
#     2017-08-30: File created. Calculation not yet fully implemented.

# Written by Arne M. T. Løken as part of a specialization project in 
# physics at NTNU, fall 2017.

#---------------------------------------------------------------------#

# First of all, I import the function definition for the velocity
# field, and the numerical integrator
from velocity_field import vel
from numerical_integrators impoort rk4

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
# For this purpose, I chose to use meshgrids:

xy, yx = np.meshgrid(np.linspace(xmin,xmax,Nx), np.linspace(ymin,ymax,Ny))

