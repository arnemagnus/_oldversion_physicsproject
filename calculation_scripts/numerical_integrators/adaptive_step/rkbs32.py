# Changelog:
#     2017-09-07: Bogacki-Shampine 3(2) method successfully
#                 implemented for the first time, albeit with a
#                 radically different overhanging file structure.
#
#     2017-09-19: Radically altered the structure of the numerical
#                 integrator package. From here on out, each 
#                 integrator is contained within its own file, 
#                 facilitating finding any given integrator in the
#                 event that changes must be made.
#
#                 In addition, the integrators now follow a more 
#                 logical hierarchial system, with single-step
#                 integrators clearly differentiated from their
#                 multi-step brethren, for instance.
#
#                 This change was partially made with multi-step
#                 methods in mind, where a single-step method
#                 must be used at the first step, but also as a means
#                 to provide more robust program code which should
#                 be easier to maintain than was the case for my
#                 original structure.
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.


def rkbs32(t, x, h, f, atol = None, rtol = None):
   """This function attempts a single time step forwards, using the
   Bogacki-Shampine 3(2) adaptive timestep integrator scheme. If
   the new step is not accepted, the time level and the coordinates 
   are not updated, while the time increment is refined.

   The Bogacki-Shampine 3(2) method calculates two independent
   approximations to a step forwards in time for an ODE system, of
   second and third order, respectively. The scheme is tuned such 
   that the error of the third order solution is minimal.

   The second order solution (interpolant) is used in order to find 
   a criterion for rejecting / accepting the trial step:
       - If the difference between the two solutions is larger than 
         some threshold, the solution is rejected, and the time
         increment refined
       - If the difference between the solutions is smaller than or
         equal to some threshold, the third order solution is 
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
      _x:   Bogacki-Shampine 3(2) approximation of the coordinates
               at the new time level (if the trial step is accepted)
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
   c2 = 1./2.
   c3 = 3./4.
   c4 = 1.

   # Matrix elements
   a21 = 1./2.
   a31 = 0.
   a32 = 3./4.
   a41 = 2./9.
   a42 = 1./3.
   a43 = 4./9.

   # Second order weights
   b21 = 7./24.
   b22 = 1./4.
   b23 = 1./3.
   b24 = 1./8.

   # Third order weights
   b31 = 2./9.
   b32 = 1./3.
   b33 = 4./9.
   b34 = 0.

   # Find "slopes"
   k1 = f(t       , x                                 )
   k2 = f(t + c2*h, x + a21*h*k1                      )
   k3 = f(t + c2*h, x + a31*h*k1 + a32*h*k2           )
   k4 = f(t + c3*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3)

   # Find second and third order prediction of new point
   x_2 = x + h*(b21*k1 + b22*k2 + b23*k3 + b24*k4)
   x_3 = x + h*(b31*k1 + b32*k2 + b33*k3 + b34*k4)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 3rd order, with 2nd order interpolation, hence:
   q = 2.
   
   sc = atol + np.maximum(np.abs(x_2), np.abs(x_3)) * rtol
   err = np.amax(np.sqrt((x_2-x_3)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
   if err <= 1.:
       # Step is accepted, use third order result as next position
       _x = x_3
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

