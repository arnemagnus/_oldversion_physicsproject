# -*- coding: utf-8 -*-

# Changelog:
#     2017-09-07: Bogacki-Shampine 5(4) method successfully 
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


def rkbs54(t, x, h, f, atol = None, rtol = None):
   """This function attempts a single time step forwards, using the
   Bogacki-Shampine 5(4) adaptive timestep integrator scheme. If the
   new step is not accepted, the time level and the coordinates are
   not updated, while the time increment is refined.

   The Bogacki-Shampine 5(4) method calculates three independent
   approximations to a step forwards in time for an ODE system, one
   fifth and two of fourth order, respectively. The scheme is tuned 
   tuned such that the error of the fifth order solution is minimal.

   The fourth order solutions (interpolant) are used in order to find 
   a criterion for rejecting / accepting the trial step:
       - If the difference between the fifth order solution and either
         of the fourth order solutions is larger than some threshold
         the solution is rejected, and the time increment refined
       - If the difference between fifth order solution and both the
         fourth order solutions is smaller than or equal to some 
         some threshold, the fifth order solution is accepted, and the
         solver attempts to increase the time increment

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
      _x:   Bogacki-Shampine 5(4) approximation of the coordinates at
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
   c2 = 1./6.
   c3 = 2./9.
   c4 = 3./7.
   c5 = 2./3.
   c6 = 3./4.
   c7 = 1.
   c8 = 1.

   # Matrix elements
   a21 = 1./6.
   a31 = 2./27.
   a32 = 4./27.
   a41 = 183./1372.
   a42 = -162./343.
   a43 = 1053./1372.
   a51 = 68./297.
   a52 = -4./11.
   a53 = 42./143.
   a54 = 1960./3861.
   a61 = 597./22528.
   a62 = 81./352.
   a63 = 63099./585728.
   a64 = 58653./366080.
   a65 = 4617./20480.
   a71 = 174197./959244.
   a72 = -30942./79937.
   a73 = 8152137./19744439.
   a74 = 666106./1039181.
   a75 = -29421./29068.
   a76 = 482048./414219.
   a81 = 587./8064.
   a82 = 0.
   a83 = 4440339./15491840.
   a84 = 24353./124800.
   a85 = 387./44800.
   a86 = 2152./5985.
   a87 = 7267./94080.

   # First of the fourth-order weights
   b41 = 6059./80640.
   b42 = 0.
   b43 = 8559189./30983680.
   b44 = 26411./124800.
   b45 = -927./89600.
   b46 = 443./1197.
   b47 = 7267./94080.
   b48 = 0.
   
   # Second of the fourth-order weights
   _b41 = 2479./34992.
   _b42 = 0.
   _b43 = 123./416.
   _b44 = 612941./3411720.
   _b45 = 43./1440.
   _b46 = 2272./6561.
   _b47 = 79937./1113912.
   _b48 = 3293./556956.

   # Fifth-order weights
   b51 = 587./8064.
   b52 = 0.
   b53 = 4440339./15491840.
   b54 = 24353./124800.
   b55 = 387./44800.
   b56 = 2152./5985.
   b57 = 7267./94080.
   b58 = 0.

   # Find "slopes"
   k1 = f(t       , x                                                )
   k2 = f(t + c2*h, x + a21*h*k1                                     )
   k3 = f(t + c3*h, x + a31*h*k1 + a32*h*k2                          )
   k4 = f(t + c4*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3               )
   k5 = f(t + c5*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4    )
   k6 = f(t + c6*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4
                                     + a65*h*k5                      )
   k7 = f(t + c7*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4
                                     + a75*h*k5 + a76*h*k6           )
   k8 = f(t + c8*h, x + a81*h*k1 + a82*h*k2 + a83*h*k3 + a84*h*k4
                                     + a85*h*k5 + a86*h*k6 + a87*h*k7)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*( b41*k1 +  b42*k2 +  b43*k3 +  b44*k4 +  b45*k5
                                        +  b46*k6 +  b47*k7 +  b48*k8)
   _x_4 = x + h*(_b41*k1 + _b42*k2 + _b43*k3 + _b44*k4 + _b45*k5
                                        + _b46*k6 + _b47*k7 + _b48*k8)
   x_5 = x + h*( b51*k1 +  b52*k2 +  b53*k3 +  b54*k4+  b55*k5
                                        +  b56*k6 +  b57*k7 +  b58*k8)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.
   
   sc = atol + np.maximum(np.abs(x_4), np.abs(_x_4)) * rtol
   err = np.amax(np.sqrt((x_4-_x_4)**2)/sc)


   if err <= 1.:
       # First trial step accepted.
       sc = atol + np.maximum(np.abs(_x_4), np.abs(x_5)) * rtol
       err = np.amax(np.sqrt((_x_4-x_5)**2)/sc)
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
            # Step is rejected, position and time not updated.
            _x = x
            _t = t
            # Refining h:
            h_opt = h * (1./err) ** (1./(q + 1.))
            _h = fac * h_opt
   else:
       # First trial step is rejected, position and time not updated
       _x = x
       _t = t
       # Refining h, based on first trial step:
       h_opt = h * (1./err) ** (1./(q + 1.))
       _h = fac * h_opt
   return _t, _x, _h
    
