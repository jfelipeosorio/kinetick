import numpy as np

def trap(f,delta):
    sum1 = f[0] + f[-1]
    sum2 = 2 * np.sum(f[1:-1])
    s = sum1 + sum2
    return delta*s/2 