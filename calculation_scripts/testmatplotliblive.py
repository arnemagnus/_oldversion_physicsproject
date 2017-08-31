import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

x = np.linspace(0,2,201)
y = np.linspace(0,1,101)

xy, yx = np.meshgrid(x,y)

ims = []

tmin, tmax = 0, 0.02

h = 1e-3

fig = plt.figure()

for i in range(int(np.ceil((tmax-tmin)/h))):
    plt.pcolormesh(xy,yx,np.sin(xy+np.pi*i*h)+np.cos(yx+np.pi*i*h),
                  cmap='RdBu_r')
    plt.show()
    plt.pause(0.01)
    plt.clf()
