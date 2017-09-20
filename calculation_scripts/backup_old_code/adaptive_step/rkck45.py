# -*- coding: utf-8 -*-

# Changelog:
#     2017-09-07: Cash-Karp 4(5) method successfully implemented
#                 for the first time, albeit with a radically
#                 different overhanging file structure.
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


def rkck45(t, x, h, f, atol = None, rtol = None):
   """This function attempts a single time step forwards, using the
   Cash-Karp 4(5) adaptive timestep integrator scheme. If the
   new step is not accepted, the time level and the coordinates are
   not updated, while the time increment is refined.

   The Cash-Karp 4(5) method calculates two independent
   approximations to a step forwards in time for an ODE system, of
   fourth and fifth order, respectively. The scheme is tuned such 
   that the error of the fourth order solution is minimal.

   The fourth order solution (interpolant) is used in order to find 
   a criterion for rejecting / accepting the trial step:
       - If the difference between the two solutions is larger than 
         some threshold, the solution is rejected, and the time
         increment refined
       - If the difference between the solutions is smaller than or
         equal to some threshold, the fourth order solution is 
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
      _x:   Cash-Karp 4(5) approximation of the coordinates at
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
   c2 = 1./5.
   c3 = 3./10.
   c4 = 3./5.
   c5 = 1.
   c6 = 7./8.

   # Matrix elements
   a21 = 1./5.
   a31 = 3./40.
   a32 = 9./40.
   a41 = 3./10.
   a42 = -9./10.
   a43 = 6./5.
   a51 = -11./54.
   a52 = 5./2.
   a53 = -70./27.
   a54 = 35./27.
   a61 = 1631./55296.
   a62 = 175./512.
   a63 = 575./13824.
   a64 = 44275./110592.
   a65 = 253./4096.

   # Fourth-order weights
   b41 = 2825./27648.
   b42 = 0.
   b43 = 18575./48384.
   b44 = 13525./55296.
   b45 = 277./14336.
   b46 = 1./4.

   # Fifth-order weights
   b51 = 37./378.
   b52 = 0.
   b53 = 250./621.
   b54 = 125./594.
   b55 = 0.
   b56 = 512./1771.

   # Find "slopes"
   k1 = f(t       , x                                                )
   k2 = f(t + c2*h, x + a21*h*k1                                     )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                          )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3               )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4    )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
                                                           + a65*h*k5)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 4th order, with 5th order interpolation, hence:
   q = 5.

   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   if err <= 1.:
      # Step is accepted, use fourth order result as next position
      _x = x_4
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

