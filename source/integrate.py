import numpy as np
from numpy import trapz

def trap(f,delta):
    sum1 = f[0] + f[-1]
    sum2 = 2 * np.sum(f[1:-1])
    s = sum1 + sum2
    return np.sqrt(delta*s/2)

def trap2d(f,x,y):
    return np.trapz(np.trapz(f, y, axis=0), x, axis=0)