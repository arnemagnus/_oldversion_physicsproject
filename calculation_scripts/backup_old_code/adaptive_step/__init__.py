# -*- coding: utf-8 -*-

# Changelog:
#     2017-09-19: File created.
#
#                 The following schemes hace been implemented  
#                 previously, and collected from my previous numerical
#                 integrator file system:
#                    - Bogacki-Shampine 3(2) and 4(5) schemes
#                    - Cash-Karp 4(5) scheme
#                    - Dormand-Prince 5(4) and 8(7) schemes
#                    - Runge-Kutta-Fehlberg 4(5) scheme
#
#                 whereas the following schemes were implemented
#                 from scratch (i.e., from inspecting Butcher
#                 Tableaus):
#                    - Fehlberg 1(2) scheme
#                    - Heun-Euler 2(1) scheme       
#
# Written by Arne Magnus T. Løken as part of a specialization
# project in physics at NTNU, fall 2017.

"""This module contains a selection of adaptive timestep integrators
intended for general-purpose use. All integrators have the same
function signature, as in, they take the same input parameters and
return the same output variables in the same order, with the
difference being the underlying integration scheme.

The module contains the following adaptive step size integrators:
   rkbs32: Bogacki-Shampine 3(2)
   rkbs54: Bogacki-Shampine 5(4)
   rkck45: Cash-Karp 4(5)
   rkdp54: Dormand-Prince 5(4)
   rkdp87: Dormand-Prince 8(7)
   rkf12:  Runge-Kutta-Fehlberg 1(2)
   rkf45:  Runge-Kutta-Fehlberg 4(5)
   rkhe21: Heun-Euler 2(1)

where the digit outside of the parenthesis indicates the method order,
and the digit within the parenthesis indicates the order of the
interpolant solution (used in adjusting the time step).

All functions have the same structure:

def scheme(t, x, h, f, atol, rtol):
   [...]
   return _t, _x, _h

where t:    Current time level
      x:    Current coordinates, array-like
      h:    Current time increment
      f:    Function handle for the derivatives (the RHS of the ODE
               system), function signature: f = f(t, x)
      atol: Absolute tolerance level (OPTIONAL)
      rtol: Relative tolerance level (OPTIONAL)

      _t:   New time level (if trial step is accepted)
            Current time level (unaltered, if the trial step is
               rejected)
      _x:   Approximation of the coordinates at the new time level
               (if trial step is accepted)
            Current coordinates (unaltered, if the trial step is
               rejected)
      _h:   Updated time increment. Generally increased or decreased,
               depending on whether the trial step is accepted or not
"""
import numerical_integrators.adaptive_step.rkbs32
import numerical_integrators.adaptive_step.rkbs54
import numerical_integrators.adaptive_step.rkck45
import numerical_integrators.adaptive_step.rkdp54
import numerical_integrators.adaptive_step.rkdp87
import numerical_integrators.adaptive_step.rkf12
import numerical_integrators.adaptive_step.rkf45
import numerical_integrators.adaptive_step.rkhe21


