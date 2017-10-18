from numba import jit
import numpy as np

@jit#(nopython=True)
def velocity_field(t, x, A, e, w):
    a = e * np.sin(w*t)
    b = 1 - 2*e*np.sin(w*t)
    f = a*x[0,:]**2 + b*x[0,:]
    v = np.empty(np.shape(x))
    v[0,:] = -np.pi*A*np.sin(np.pi*f)*np.cos(np.pi*x[1,:]) # x-component
    v[1,:] = np.pi*A*np.cos(np.pi*f)*np.sin(np.pi*x[1,:])*(2*a*x[0,:] + b) # y-component
    return v
