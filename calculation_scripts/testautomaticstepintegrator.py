from __future__ import division

from numerical_integrators.adaptive_step import *

import numpy as np

import matplotlib.pyplot as plt

def get_trajectory(f, integrator, t_max, h0, x0, t0, **kwargs):
    Xs = [x0]
    Ts = [t0]
    t = t0
    x = x0
    h = h0
    while t < t_max:
        # Ensure we don't overstep:
        h = np.min([h, t_max - t])
        t_, x_, h = integrator(t, x, h, f, **kwargs)
        # Check if step was accepted
        if t_ > t:
            t = t_
            x = x_
            Xs.append(x)
            Ts.append(t)
    return Ts ,Xs

def get_endpoint(f, integrator, t_max, h0, x0, t0, **kwargs):
    t = t0
    x = x0
    h = h0
    while t < t_max:
        # Ensure we don't overstep:
        h = np.min(h, t_max - t)
        t_, x_, h = integrator(t, x, h, f, **kwargs)
        # Check if step was accepted
        if t_ > t:
            t = t_
            x = x_
    return t, x


# Test problem:
def f(t, x):
    # Derivative of arctan function
    return 1/(1 + t*t)

# Get trajectory, using Bogacki-Shampine methods
Ts, Xs = get_trajectory(f,rkbs54,t_max=30,h0=0.1,x0=0,t0=0)
print('Bogacki-Shampine 5(4) steps: {}'.format(len(Ts)))
plt.plot(Ts,Xs,'g-o', label = 'Bogacki-Shampine 5(4) solution')
Ts, Xs = get_trajectory(f,rkbs32,t_max=30,h0=0.1,x0=0,t0=0)
print('Bogacki-Shampine 3(2) steps: {}'.format(len(Ts)))
plt.plot(Ts,Xs,'m-x', label = 'Bogacki-Shampine 3(2) solution')
# Get trajectory, using Fehlberg method
Ts, Xs = get_trajectory(f,rkf45,t_max=30,h0=1e-2,x0=0,t0=0)
print('Fehlberg 4(5) steps: {}'.format(len(Ts)))
plt.plot(Ts,Xs, 'r.', label='Fehlberg 4(5) solution')
# Get trajectory, using Dormand-Prince methods
Ts, Xs = get_trajectory(f,rkdp54,t_max=30,h0=1e-2,x0=0,t0=0)
print('Dormand-Prince 5(4) steps: {}'.format(len(Ts)))
plt.plot(Ts, Xs, 'bo', label='Dormand-Prince 5(4) solution')
Ts, Xs = get_trajectory(f,rkdp87,t_max=30,h0=1e-2,x0=0,t0=0)
print('Dormand-Prince 8(7) steps: {}'.format(len(Ts)))
plt.plot(Ts, Xs, 'cx', label='Dormand-Prince 8(7) solution')
# Get trajectory, using Cash-Karp method
Ts, Xs = get_trajectory(f, rkck45,t_max=30,h0=1e-2,x0=0,t0=0)
print('Cash-Karp 5(4) steps: {}'.format(len(Ts)))
plt.plot(Ts, Xs, 'yx', label='Cash-Karp 5(4) solution')
plt.plot(Ts, np.arctan(Ts), 'g--', label = 'Analytical solution')
plt.legend()
# Plot horizontal asymptote
plt.plot(Ts, np.ones(len(Ts))*np.pi/2)
plt.show()
