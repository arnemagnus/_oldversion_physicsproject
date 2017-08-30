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

# Written by Arne M. T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.                

#--------------------------------------------------------------------#

# Explicit Euler method (also known as the simplest, one-stage
#                        Runge-Kutta method, 1st order accurate)
def euler(x, # Coordinates, as an array
          t, # Current time
          h, # Time step
          f  # Function handle for the derivatives (RHS),
             # function signature: f = f(x, t)
         ):
   # This function performs a single time step forwards, using the 
   # explicit Euler scheme, and returns the new coordinates at the
   # new time level, t' = t + h

   # Find "slopes"
   k1 = f(x                , t       )
   # Find estimate for coordinates at new time level
   _x = x + k1*h
   
   return _x



# Heun's method (also known as a two-stage Runge-Kutta method, 
#                2nd order accurate)
def rk2(x, # Coordinates, as an array
        t, # Current time
        h, # Time step,
        f  # Function handle for the derivatives (RHS)
           # function signature: f = f(x, t)
       ):
   # This function performs a single time step forwards, using the
   # Heun scheme, and returns the new coordinates at the new time 
   # level, t' = t + h

   # Find "slopes"
   k1 = f(x                , t       )
   k2 = f(x + k1*h         , t + h   )

   # Find estimate for coordinates at new time level
   _x = x + (k1 + k2)*h/2

   return _x



# Kutta's method (also known as a three-stage Runge-Kutta method, 
#                 3rd order accurate)
def rk3(x, # Coordinates, as an array
        t, # Current time
        h, # Time step,
        f  # Function handle for the derivatives (RHS)
           # function signature: f = f(x, t)
       ):
   # This function performs a single time step forwards, using the
   # Kutta scheme, and returns the new coordinates at the new time
   # level, t' = t + h

   # Find "slopes"
   k1 = f(x                , t      )
   k2 = f(x + k1*h/2       , t + h/2)
   k3 = f(x - k1*h + 2*k2*h, t + h  )

   # Find estimate for coordinates at new time level
   _x = x + (k1 + 4*k2 + k3)*h/6

   return _x



# "The" Runge-Kutta method (also known as a four-step Runge-Kutta
#                           method, 4th order accurate)
def rk4(x, # Coordinates, as an array
        t, # Current time
        h, # Time step,
        f  # Function handle for the derivatives (RHS)
           # function signature: f = f(x, t)        
       ):
   # This function performs a single time step forwards, using the
   # RK4 scheme, and returns the new coordinates at the new time
   # level, t' = t + h

   # Find "slopes"
   k1 = f(x, t)
   k2 = f(x + k1*h/2       , t + h/2)
   k3 = f(x + k2*h/2       , t + h/2)
   k4 = f(x + k3*h         , t + h  )

   # Find estimate for coordinates at new time level
   _x = x + (k1 + 2*k2 + 2*k3 + k4)*h/6

   return _x 
   
