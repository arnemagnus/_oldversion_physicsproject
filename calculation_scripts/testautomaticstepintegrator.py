from numerical_integrators import *

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

# Get trajectory, using Bogacki-Shampine method
Ts, Xs = get_trajectory(f,bs5,t_max=30,h0=0.1,x0=0,t0=0)
print('BS5 steps: {}'.format(len(Ts)))
plt.plot(Ts,Xs,'g-o', label = 'Bogacki-Shampine solution')
# Get trajectory, using Fehlberg method
Ts, Xs = get_trajectory(f,rkf,t_max=30,h0=1e-2,x0=0,t0=0)
print('Fehlberg steps: {}'.format(len(Ts)))
plt.plot(Ts,Xs, 'r.', label='Fehlberg solution')
# Get trajectory, using Dormand-Prince method
Ts, Xs = get_trajectory(f,dopri,t_max=30,h0=1e-2,x0=0,t0=0)
print('Dormand-Prince steps: {}'.format(len(Ts)))
plt.plot(Ts, Xs, 'bo', label='Dormand-Prince solution')
plt.plot(Ts, np.arctan(Ts), 'g--', label = 'Analytical solution')
plt.legend()
# Plot horizontal asymptote
plt.plot(Ts, np.ones(len(Ts))*np.pi/2)
plt.show()
