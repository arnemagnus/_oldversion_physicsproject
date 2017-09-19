# Changelog:
#     2017-09-01: Dormand-Prince 5(4) method successfully implemented
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


def rkdp54(t, x, h, f, atol = None, rtol = None):
   """This function attempts a single time step forwards, using the
   Dormand-Prince 5(4) adaptive timestep integrator scheme. If the
   new step is not accepted, the time level and the coordinates are
   not updated, while the time increment is refined.

   The Dormand-Prince 5(4) method calculates two independent
   approximations to a step forwards in time for an ODE system, of
   fifth and fourth order, respectively. The scheme is tuned such that
   the error of the fifth order solution is minimal.

   The fourth order solution (interpolant) is used in order to find a
   criterion for rejecting / accepting the trial step:
       - If the difference between the two solutions is larger than 
         some threshold, the solution is rejected, and the time
         increment refined
       - If the difference between the solutions is smaller than or
         equal to some threshold, the fifth order solution is 
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
      _x:   Dormand-Prince 5(4) approximation of the coordinates at
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
   c4 = 4./5.
   c5 = 8./9.
   c6 = 1.
   c7 = 1.

   # Matrix elements
   a21 = 1./5.
   a31 = 3./40.
   a32 = 9./40.
   a41 = 44./45.
   a42 = -56./15.
   a43 = 32./9.
   a51 = 19372./6561.
   a52 = -25350./2187.
   a53 = 64448./6561.
   a54 = -212./729.
   a61 = 9017./3168.
   a62 = -335./33.
   a63 = 46732./5247.
   a64 = 49./176.
   a65 = -5103./18656.
   a71 = 35./384.
   a72 = 0.
   a73 = 500./1113.
   a74 = 125./192.
   a75 = -2187./6784.
   a76 = 11./84.

   # Fourth-order weights
   b41 = 5179./57600.
   b42 = 0.
   b43 = 7571./16695.
   b44 = 393./640.
   b45 = -92097./339200.
   b46 = 187./2100.
   b47 = 1./40.

   # Fifth-order weights
   b51 = 35./384.
   b52 = 0.
   b53 = 500./1113.
   b54 = 125./192.
   b55 = -2187./6784.
   b56 = 11./84.
   b57 = 0.

   # Find "slopes"
   k1 = f(t       , x                                                )
   k2 = f(t + c2*h, x + a21*h*k1                                     )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                          )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3               )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4    )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
                                                         + a65*h*k5)
   k7 = f(t + c7*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4
                                                         + a75*h*k5
                                                           + a76*h*k6)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6
                                                             + b47*k7)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6
                                                             + b57*k7)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.
   
   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   if err <= 1.:
       # Step is accepted, use fifth order result as next position
       _x = x_5
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
