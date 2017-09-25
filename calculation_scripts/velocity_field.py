# -*- coding=utf-8 -*-

# This file contains all the necessary function definitions for
# the implementation of the velocity field. My intention is for
# this file to be imported into other plot-generating and
# trajectory-calculating files for further analysis.

# Changelog:
#     2017-08-23: File created, vx and vy defined as separate
#                 functions
#     2017-08-25: vx and vy replaced by a single function, returning
#                 the velocity components as a two-component numpy
#                 array. This is in order to make use of my general-
#                 purpose numerical integration schemes, rather than
#                 explicitly coding the numerical integrators every
#                 time they are used.
#     2017-09-01: Changed the function call signature, from f(x,t) to
#                 f(t,x), in accordance with the literature.

# Written by Arne M. T. Løken as part of a specialization project in
# physics at NTNU, fall 2017

#--------------------------------------------------------------------#

from __future__ import division

# Seeing as the analytical velocity field in question involves
# the sine and cosine functions, as well as pi, we require a
# representation of those functions here. Numpy does the job:
import numpy as np

# Because the velocity field is given, the implementation
# is in fact rather straight-forward. Seeing as the "subfunction"
# f and its spatial derivative are not needed outside this file
# specifically, I preappend their function names with an underscore.

# The following four function definitions are consituent parts of the
# velocity field, but not particularly interesting on their own:

def _a(t,eps,w):
    return eps*np.sin(w*t)

def _b(t,eps,w):
    return 1 - 2*eps*np.sin(w*t)

def _f(t,x,eps,w):
    return _a(t,eps,w) * x**2 + _b(t,eps,w) * x

def _dfdx(t,x,eps,w):
    return 2 * _a(t,eps,w) * x + _b(t,eps,w)

# The velocity field makes use of the above function definitions.
# I choose to define the velocity fields with optional parameters
# for A, eps and w, because all the problems in the exam text
# use the same parameter values. The velocity field is implemented
# as follows:

def vel(t,x,A=None,eps=None,w=None):
   if A == None:
      A = 0.1
   if eps == None:
      eps = 0.25
   if w == None:
      w = 1

   x, y = x[0], x[1]

   vx = -np.pi*A * np.sin(np.pi*_f(t,x,eps,w)) * np.cos(np.pi*y)
   vy = (np.pi*A * np.cos(np.pi*_f(t,x,eps,w)) * np.sin(np.pi*y)
                               *_dfdx(t,x,eps,w)
        )

   return np.array([vx, vy])


