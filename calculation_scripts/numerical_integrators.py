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
#     2017-08-30: Clarified the function signature for the derivative
#                 functions.
#     2017-09-01: Changed function signature of the integrators as
#                 well as the derivative functions, from
#                 f(x,t) --> f(t,x) in accordance with the literature.
#                 Added implementation of the Fehlberg and
#                 Dormand-Prince automatic step size integrators.
#                 Changed return variables of the fixed-stepsize 
#                 integrators, so that they are consistent with
#                 the return variables from their automatic stepsize
#                 siblings.

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

# The Runge-Kutta-Fehlberg method (also known as the Fehlberg method,
#                                  4th order method with 5th order
#                                  correction, adaptive timestep)
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

   # The method is 4th order, with 5th order correction, hence:
   q = 4.

   sc = atol + np.maximum(np.abs(x_4), np.abs(x_5)) * rtol
   err = np.amax(np.sqrt((x_4-x_5)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
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


# The Dormand-Prince method (4th order method with 5th (6th) order
#                            correction, adaptive timestep)
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

   # Fifth-order weights
   b51 = 35./384.
   b52 = 0.
   b53 = 500./1113.
   b54 = 125./192.
   b55 = -2187./6784.
   b56 = 11./84.
   b57 = 0.

   # Sixth-order weights
   b61 = 5179./57600.
   b62 = 0.
   b63 = 7571./16695.
   b64 = 393./640.
   b65 = -92097./339200.
   b66 = 187./2100.
   b67 = 1./40.

   # Find "slopes"
   k1 = f(t       , x                                                                  )
   k2 = f(t + c2*h, x + a11*h*k1                                                       )
   k3 = f(t + c3*h, x + a21*h*k1 + a22*h*k2                                            )
   k4 = f(t + c4*h, x + a31*h*k1 + a32*h*k2 + a33*h*k3                                 )
   k5 = f(t + c5*h, x + a41*h*k1 + a42*h*k2 + a43*h*k3 + a44*h*k4                      )
   k6 = f(t + c6*h, x + a51*h*k1 + a52*h*k2 + a53*h*k3 + a54*h*k4 + a55*h*k5           )
   k7 = f(t + c7*h, x + a61*h*k1 + a62*h*k2 + a63*h*k3 + a64*h*k4 + a65*h*k5 + a66*h*k6)

   # Find fourth and fifth order prediction of new point
   x_4 = x + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4 + b55*k5 + b56*k6 + b57*k7)
   x_5 = x + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5 + b66*k6 + b67*k7)

   # Implementing error check and variable stepsize roughly as in
   # Hairer, Nørsett and Wanner: "Solving ordinary differential
   #                              equations I -- Nonstiff problems",
   #                              pages 167 and 168 in the 2008 ed.

   # The method is 4th order, with 5th (6th) order correction, hence:
   q = 4.
   
   sc = atol + np.maximum(np.abs(x_5), np.abs(x_6)) * rtol
   err = np.amax(np.sqrt((x_5-x_6)**2)/sc)

   # Safety factor for timestep correction
   fac = 0.8
   maxfac = 2
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



