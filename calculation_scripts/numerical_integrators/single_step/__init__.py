# Changelog:
#     2017-09-19: File created.
#
#                 The following schemes hace been implemented  
#                 previously, and collected from my previous numerical
#                 integrator file system:
#                    - Euler scheme       (1st order)
#                    - Heun sceheme       (2nd order, RK2)
#                    - Kutta scheme       (3rd order, RK3)
#                    - Runge-Kutta scheme (4th order, RK4)
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.

"""This module contains a selection of fixed increment single-step
integrators intended for general-purpose use. All integrators have
the same function signature, as in, they take the same input
parameters and return the same output variables in the same order,
with the difference being the underlying integration scheme.

The module contains the following fixed increment single-step
integrators:
   euler: Euler scheme (1st order)
   rk2:   Heun scheme (2nd order)
   rk3:   Kutta scheme (3rd order)
   rk4:   Runge-Kutta scheme (4th order)


All functions have the same structure:

def scheme(t, x, h, f):
   [...]
   return _t, _x, _h

where t:    Current time level
      x:    Current coordinates, array-like
      h:    Time increment (fixed)
      f:    Function handle for the derivatives (the RHS of the ODE
               system), function signature: f = f(t, x)

      _t:   New time level 
      _x:   Approximation of the coordinates at the new time level
      _h:   Time increment (unaltered, yet returned, in order for
              the return variable signatures of the numerical
              integrators to remain consistent across single-,
              multi- and adaptive step methods

"""
import numerical_integrators.single_step.euler
import numerical_integrators.single_step.rk2
import numerical_integrators.single_step.rk3
import numerical_integrators.single_step.rk4
