# This file contains a set of numerical integration schemes for
# general-purpose use. To this end, the integrators are implemented
# with a consistent call signature. In addition, they can handle 
# any number of coordinates, as long as a function returning the
# derivative (RHS) is provided for each of them.

# Changelog: 
#     2017-08-25: File created. 
#                 Explicit Euler method, Heun's method, 
#                 Kutta's method and the classical RK4 method added.
#                 Added spaces (whitespace) everywhere slopes are 
#                 estimated, in order to make the differences between
#                 the various methods more apparent.
#
#     2017-08-30: Clarified the function signature for the derivative
#                 functions.
#
#     2017-09-01: Changed function signature of the integrators as
#                 well as the derivative functions, from
#                 f(x,t) --> f(t,x) in accordance with the literature.
#
#                 Added implementation of the Cash-Karp, Fehlberg and
#                 Dormand-Prince automatic step size integrators.
#                 Changed return variables of the fixed-stepsize 
#                 integrators, so that they are consistent with
#                 the return variables from their automatic stepsize
#                 siblings.
#
#                 Added a not fully functioning implementation
#                 of the Bogacki-Shampine 4(5) scheme.
# 
#     2017-09-07: Corrected several errors in terms of the matrix
#                 elements of the Cash-Karp 4(5) scheme. The scheme
#                 works as intended, now.
#
#                 Performed several similar corrections to the
#                 Bogacki-Shampine scheme, which now works as intended.
#
# Written by Arne M. T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.                

#--------------------------------------------------------------------#

import numpy as np

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Methods with fixed stepsize   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Explicit Euler method (also known as the simplest, one-stage
#                        Runge-Kutta method, 1st order accurate)
def euler(t, # Current time
          x, # Coordinates, as an array
          h, # Time step
          f  # Function handle for the derivatives (RHS),
             # function signature: f = f(t, x)
         ):
   # This function performs a single time step forwards, using the 
   # explicit Euler scheme, and returns the new coordinates at the
   # new time level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + k1*h
   
   return _t, _x, h



# Heun's method (also known as a two-stage Runge-Kutta method, 
#                2nd order accurate)
def rk2(t, # Current time
        x, # Coordinates, as an array
        h, # Time step
        f  # Function handle for the derivatives (RHS),
           # function signature: f = f(t, x)
        ):
   # This function performs a single time step forwards, using the
   # Heun scheme, and returns the new coordinates at the new time 
   # level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   k2 = f(t + h  , x + k1 * h       )
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + (k1 + k2)*h/2

   return _t, _x, h



# Kutta's method (also known as a three-stage Runge-Kutta method, 
#                 3rd order accurate)
def rk3(t, # Current time
        x, # Coordinates, as an array
        h, # Time step
        f  # Function handle for the derivatives (RHS),
           # function signature: f = f(t, x)
        ):
   # This function performs a single time step forwards, using the
   # Kutta scheme, and returns the new coordinates at the new time
   # level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   k2 = f(t + h/2, x + k1*h/2       )
   k3 = f(t + h  , x - k1*h + 2*k2*h)
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + (k1 + 4*k2 + k3)*h/6

   return _t, _x, h



# "The" Runge-Kutta method (also known as a four-step Runge-Kutta
#                           method, 4th order accurate)
def rk4(t, # Current time
        x, # Coordinates, as an array
        h, # Time step
        f  # Function handle for the derivatives (RHS),
           # function signature: f = f(t, x)
        ):
   # This function performs a single time step forwards, using the
   # RK4 scheme, and returns the new coordinates at the new time
   # level, t' = t + h

   # Find "slopes"
   k1 = f(t      , x                )
   k2 = f(t + h/2, x + k1*h/2       )
   k3 = f(t + h/2, x + k2*h/2       )
   k4 = f(t + h  , x + k3*h         )
   # Find new time level
   _t = t + h
   # Find estimate for coordinates at new time level
   _x = x + (k1 + 2*k2 + 2*k3 + k4)*h/6

   return _t, _x, h
   
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#   Methods with adaptive stepsize   #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# The Cash-Karp method (5th order method with 4th order
#                       interpolation, adaptive timestep)
def cash_karp(t,           # Current time
              x,           # Coordinates, as an array
              h,           # Time step
              f,           # Function handle for the derivatives (RHS),
                           # function signature: f = f(t, x)
              atol = 1e-6, # Absolute tolerance level (optional)
              rtol = 1e-9  # Relative tolerance level (optional)
             ):
   # This function attempts a single time step forwards, using the 
   # Cash-Karp adaptive timestep integrator scheme. If the new step is
   # not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./5.
   c3 = 3./10.
   c4 = 3./5.
   c5 = 1.
   c6 = 7./8.

   # Matrix elements
   a11 = 1./5.
   a21 = 3./40.
   a22 = 9./40.
   a31 = 3./10.
   a32 = -9./10.
   a33 = 6./5.
   a41 = -11./54.
   a42 = 5./2.
   a43 = -70./27.
   a44 = 35./27.
   a51 = 1631./55296.
   a52 = 175./512.
   a53 = 575./13824.
   a54 = 44275./110592.
   a55 = 253./4096.

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
   k1 = f(t       , x                                                       )
   k2 = f(t + c2*h, x + a11*h*k1                                            )
   k3 = f(t + c3*h, x + a21*h*k1 + a22*h*k2                                 )
   k4 = f(t + c4*h, x + a31*h*k1 + a32*h*k2 + a33*h*k3                      )
   k5 = f(t + c5*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3 + a44*h*k4           )
   k6 = f(t + c6*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4 + a55*h*k5)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.

   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
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

# The Runge-Kutta-Fehlberg method (also known as the Fehlberg method,
#                                  5th order method with 4th order
#                                  interpolation, adaptive timestep)
def rkf(t,           # Current time
        x,           # Coordinates, as an array
        h,           # Time step
        f,           # Function handle for the derivatives (RHS),
                     # function signature: f = f(t, x)
        atol = 1e-6, # Absolute tolerance level (optional)
        rtol = 1e-9  # Relative tolerance level (optional)
        ):
   # This function attempts a single time step forwards, using the 
   # Fehlberg adaptive timestep integrator scheme. If the new step is
   # not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./4.
   c3 = 3./8.
   c4 = 12./13.
   c5 = 1.
   c6 = 1./2.

   # Matrix elements
   a11 = 1./4.
   a21 = 3./32.
   a22 = 9./32.
   a31 = 1932./2197.
   a32 = -7200./2197.
   a33 = 7296./2197.
   a41 = 439./216.
   a42 = -8.
   a43 = 3680./513
   a44 = -845./4104.
   a51 = -8./27.
   a52 = 2.
   a53 = -3544./2565.
   a54 = 1859./4104.
   a55 = -11./40.

   # Fourth-order weights
   b41 = 25./216.
   b42 = 0.
   b43 = 1408./2565.
   b44 = 2197./4104.
   b45 = -1./5.
   b46 = 0.

   # Fifth-order weights
   b51 = 16./135.
   b52 = 0.
   b53 = 6656./12825.
   b54 = 28561./56430.
   b55 = -9./50.
   b56 = 2./55.

   # Find "slopes"
   k1 = f(t       , x                                                       )
   k2 = f(t + c2*h, x + a11*h*k1                                            )
   k3 = f(t + c3*h, x + a21*h*k1 + a22*h*k2                                 )
   k4 = f(t + c4*h, x + a31*h*k1 + a32*h*k2 + a33*h*k3                      )
   k5 = f(t + c5*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3 + a44*h*k4           )
   k6 = f(t + c6*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4 + a55*h*k5)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.

   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
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


# The Dormand-Prince method (5th order method with 4th order
#                            interpolation, adaptive timestep)
def dopri(t,           # Current time
          x,           # Coordinates, as an array
          h,           # Time step
          f,           # Function handle for the derivatives (RHS),
                       # function signature: f = f(t, x)
          atol = 1e-6, # Absolute tolerance level (optional)
          rtol = 1e-9  # Relative tolerance level (optional)
         ):
   # This function attempts a single time step forwards, using the 
   # Dormand-Prince adaptive timestep integrator scheme. If the new
   # step is not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./5.
   c3 = 3./10.
   c4 = 4./5.
   c5 = 8./9.
   c6 = 1
   c7 = 1.

   # Matrix elements
   a11 = 1./5.
   a21 = 3./40.
   a22 = 9./40.
   a31 = 44./45.
   a32 = -56./15.
   a33 = 32./9.
   a41 = 19372./6561.
   a42 = -25350./2187.
   a43 = 64448./6561.
   a44 = -212./729.
   a51 = 9017./3168.
   a52 = -335./33.
   a53 = 46732./5247.
   a54 = 49./176.
   a55 = -5103./18656.
   a61 = 35./384.
   a62 = 0.
   a63 = 500./1113.
   a64 = 125./192.
   a65 = -2187./6784.
   a66 = 11./84.

   # Fourth-order weights
   b41 = 35./384.
   b42 = 0.
   b43 = 500./1113.
   b44 = 125./192.
   b45 = -2187./6784.
   b46 = 11./84.
   b47 = 0.

   # Fifth-order weights
   b51 = 5179./57600.
   b52 = 0.
   b53 = 7571./16695.
   b54 = 393./640.
   b55 = -92097./339200.
   b56 = 187./2100.
   b57 = 1./40.

   # Find "slopes"
   k1 = f(t       , x                                                                  )
   k2 = f(t + c2*h, x + a11*h*k1                                                       )
   k3 = f(t + c3*h, x + a21*h*k1 + a22*h*k2                                            )
   k4 = f(t + c4*h, x + a31*h*k1 + a32*h*k2 + a33*h*k3                                 )
   k5 = f(t + c5*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3 + a44*h*k4                      )
   k6 = f(t + c6*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4 + a55*h*k5           )
   k7 = f(t + c7*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5 + a66*h*k6)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6 + b47*k7)
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6 + b57*k7)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.
   
   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
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



# The (fifth order) Bogacki-Shampine method (5th order method with 4th
#                                            order interpolation,
#                                            interpolation, adaptive
#                                            timestep)
# For reference, see:
#  Bogacki, P., Shampine, L.F. (1996):
#                  "An efficient Runge-Kutta (4,5) pair",
#                  Computers Math. Applic., Vol. 32, no. 6, pp. 15-28

def bs5(t,           # Current time
        x,           # Coordinates, as an array
        h,           # Time step
        f,           # Function handle for the derivatives (RHS),
                     # function signature: f = f(t, x)
        atol = 1e-6, # Absolute tolerance level (optional)
        rtol = 1e-9  # Relative tolerance level (optional)
       ):
   # This function attempts a single time step forwards, using the 
   # Bogacki-Shampine adaptive timestep integrator scheme. If the new
   # step is not accepted, the time and coordinates are not updated.

   # Nodes
   c2 = 1./6.
   c3 = 2./9.
   c4 = 3./7.
   c5 = 2./3.
   c6 = 3./4.
   c7 = 1.
   c8 = 1.

   # Matrix elements
   a11 = 1./6.
   a21 = 2./27.
   a22 = 4./27.
   a31 = 183./1372.
   a32 = -162./343.
   a33 = 1053./1372.
   a41 = 68./297.
   a42 = -4./11.
   a43 = 42./143.
   a44 = 1960./3861.
   a51 = 597./22528.
   a52 = 81./352.
   a53 = 63099./585728.
   a54 = 58653./366080.
   a55 = 4617./20480.
   a61 = 174197./959244.
   a62 = -30942./79937.
   a63 = 8152137./19744439.
   a64 = 666106./1039181.
   a65 = -29421./29068.
   a66 = 482048./414219.
   a71 = 587./8064.
   a72 = 0.
   a73 = 4440339./15491840.
   a74 = 24353./124800.
   a75 = 387./44800.
   a76 = 2152./5985.
   a77 = 7267./94080.

   # First of the fourth-order weights
   b41 = 587./8064.
   b42 = 0.
   b43 = 4440339./15491840.
   b44 = 24353./124800.
   b45 = 387./44800.
   b46 = 2152./5985.
   b47 = 7267./94080.

   # Second of the fourth-order weights
   _b41 = 6059./80640.
   _b42 = 0.
   _b43 = 8559189./30983680.
   _b44 = 26411./124800.
   _b45 = -927./89600.
   _b46 = 443./1197.
   _b47 = 7267./94080.

   # Fifth-order weights
   b51 = 2479./34992.
   b52 = 0.
   b53 = 123./416.
   b54 = 612941./3411720.
   b55 = 43./1440.
   b56 = 2272./6561.
   b57 = 79937./1113912.
   b58 = 3293./556956.

   # Find "slopes"
   k1 = f(t       , x                                                                             )
   k2 = f(t + c2*h, x + a11*h*k1                                                                  )
   k3 = f(t + c3*h, x + a21*h*k1 + a22*h*k2                                                       )
   k4 = f(t + c4*h, x + a31*h*k1 + a32*h*k2 + a33*h*k3                                            )
   k5 = f(t + c5*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3 + a44*h*k4                                 )
   k6 = f(t + c6*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4 + a55*h*k5                      )
   k7 = f(t + c7*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5 + a66*h*k6           )
   k8 = f(t + c8*h, x + a71*h*k1 + a72*h*k2 + a73*h*k3 + a74*h*k4 + a75*h*k5 + a76*h*k6 + a77*h*k7)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b41*k1 + b42*k2 + b43*k3 + b44*k4 + b45*k5 + b46*k6 + b47*k7        )
   _x_4 = x + h*(_b41*k1 + _b42*k2 + _b43*k3 + _b44*k4 + _b45*k5 + _b46*k6 + _b47*k7 )
   x_5 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6 + b57*k7 + b58*k8)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 5th order, with 4th order interpolation, hence:
   q = 4.
   
   sc = atol + np.maximum(np.abs(x_4), np.abs(_x_4)) * rtol
   err = np.amax(np.sqrt((x_4-_x_4)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
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
