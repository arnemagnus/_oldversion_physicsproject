# Changelog:
#     2017-08-25: Explicit Euler scheme successfully implemented
#                 for the first time, albeit with a radically
#                 different overhanging file structure.
#
#     2017-08-30: Function signature for the derivative function
#                 standardized clarified (via code comments)
#
#     2017-09-01: Changed function signature of the integrator as well
#                 as the derivative function, from f(x,t) --> f(t,x),
#                 in accordance with convention from literature.
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


def euler(t, x, h, f):
   """This function performs a single time step forwards, using the
   explicit Euler scheme, finding an approximation of the coordinates
   at the new time level.

   The explicit Euler scheme is the simplest Runge-Kutta scheme,
   and is first-order accurate.

   Input:
      t: Current time level
      x: Current coordinates, array-like
      h: Time increment (fixed)
      f: Function handle for the derivatives (the RHS of the ODE
             system), function signature: f = f(t, x)

   Output:
      _t: New time level
      _x: Explicit Euler approximation of the coordinates at the
              new time level
      _h: Time increment (unaltered, yet returned, in order for
              the return variable signatures of the numerical
              integrators to remain consistent across single-,
              multi- and adaptive step methods
   """
   # Find "slope"
   k1 = f(t, x)
   # Find new time level
   _t = t + h
   # Find estimate of coordinates at new time level
   _x = x + k1*h   
   
   return _t, _x, h
