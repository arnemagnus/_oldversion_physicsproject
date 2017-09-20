# -*- coding=utf-8 -*-

# This file contains default values for absolute and relative
# tolerance levels, as well as safety factors for updating the
# time increment, in adaptive timestep integrators.
# 
# Changelog:
#     2017-09-19: File created, as part of a radical restructuring of
#                 the numerical integrator package. From here on out, 
#                 the integrators are sorted in separate files based, 
#                 on their type, i.e., whether they are single-,
#                 multi- or adaptive step methods, facilitating
#                 finding any given integrator in the event that
#                 changes must be made.
#
#                 Thus, the integrators now follow a more 
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

# Default value for absolute tolerance:
atol_default = 1e-6

# Default value for relative tolerance:
rtol_default = 1e-9

# Safety factors for timestep correction:
fac = 0.8
maxfac = 2.
