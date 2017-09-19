# Changelog:
#     2017-09-19: Added working implementation of the Heun-Euler 2(1)
#                 scheme
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.

def rkhe21(t, x, h, f, atol = None, rtol = None):
   """This function attempts a single time step forwards, using the
   Heun-Euler 2(1) adaptive timestep integrator scheme. If the
   new step is not accepted, the time level and the coordinates are
   not updated, while the time increment is refined.

   The Heun-Euler 2(1) method calculates two independent
   approximations to a step forwards in time for an ODE system, of
   first and second order, respectively. The scheme is tuned such that
   the error of the second order solution is minimal.

   The first order solution (interpolant) is used in order to find a
   criterion for rejecting / accepting the trial step:
       - If the difference between the two solutions is larger than 
         some threshold, the solution is rejected, and the time
         increment refined
       - If the difference between the solutions is smaller than or
         equal to some threshold, the second order solution is 
         accepted, and the solver attempts to increase the time
         increment

   Input:
      t:    Current time level
      x:    Current coordinates, array-like
      h:    Current time increment
      f:    Function handle for the derivatives (the RHS of the ODE
               system), function signature: f = f(t, x)
      atol: Absolute tolerance level (OPTIONAL)
      rtol: Relative toleranve level (OPTIONAL)

   Output:
      _t:   New time level (if the trial step is accepted)
            Current time level (unaltered, if the trial step is
               rejected)
      _x:   Heun-Euler 2(1) approximation of the coordinates at
               the new time level (if the trial step is accepted)
            Current coordinates (unaltered, if the trial step is
               rejected)
      _h:   Updated time increment. Generally increased or decreased,
               depending on whether the trial step is accepted or
               rejected
   """
   # numpy contains very useful representations of abs(:) and
   # max(:) functions, among other things:
   import numpy as np

   # We import the predefined default tolerance levels:
   from _adaptive_timestep_params import atol_default, rtol_default

   # We import the predefined safety factors for timestep correction:
   from _adaptive_timestep_params import fac, maxfac

   # We explicitly handle the optional arguments:
   if not atol:
       atol = atol_default
   if not rtol:
       rtol = rtol_default

   # Nodes
   c2 = 1.

   # Matrix elements
   a11 = 1.

   # First-order weights:
   b11 = 1.
   b12 = 0.

   # Second-order weights:
   b21 = 1./2.
   b22 = 1./2.

   # Find "slopes"
   k1 = f(t       , x                      )
   k2 = f(t + c2*h, x + h*a21*k1           )

   # Find first and second order prediction of new point
   x_1 = x + h*(b11*k1 + b12*k2)
   x_2 = x + h*(b21*k1 + b22*k2)

      # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 2nd order, with 1st order interpolation, hence:
   q = 1.
   
   sc = atol + np.maximum(np.abs(x_1), np.abs(x_2)) * rtol
   err = np.amax(np.sqrt((x_1-x_2)**2)/sc)

   if err <= 1.:
       # Step is accepted, use first order result as next position
       _x = x_2
       _t = t + h
       # Refining h:
       # Should err happen to be 0, the optimal h is infinity.
       # We set an upper limit to get sensible behaviour:
       if err == 0.:
           h_opt = 10
       else:
           h_opt = h * (1./err) ** (1./(q + 1.))
       _h = max(maxfac * h, fac * h_opt)
   else:
       # Step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h

   

