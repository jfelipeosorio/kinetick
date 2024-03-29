import numpy as np
import jax.numpy as jnp
from jax import grad, jacfwd, vmap, jit
import scipy.integrate as integrate

# heat 1D

def heat_u0(x):
    return -4*jnp.square(x - 0.5) + 1.

def bn(k):        
    def myfun(y,k):
        return 2*(-4*(y-0.5)**2+1)*np.sin(k*np.pi*y)
    return integrate.quad(myfun,0,1,args=(k,))
      
def heat_sol(x,t,m):
    '''
    t: time coordinate.
    x: space coordinate.
    n: level of truncation in analytic solution.
    '''
    sum = 0.
    for i in range(1,m):
        sum += bn(i)[0] * np.exp(-i**2*np.pi**2*t)*np.sin(i*np.pi*x)

    return sum

# advection-diffusion 1D    

def ad_u0(x):
    return -4*jnp.square(x - 0.5) + 1.

def ad_bn(k,beta,c):        
    def myfun(y,k,beta,c):
        return 2*np.exp(-y*beta/(2*(c**2)))*(-4*jnp.square(y - 0.5) + 1.)*np.sin(k*np.pi*y)
    return integrate.quad(myfun,0,1,args=(k,beta,c,))  

def ad_sol(x,t,beta,c,m):
    '''
    x: space coordinate.
    t: time coordinate.
    beta: advection coefficient
    c: diffusion coefficient
    m: level of truncation in analytic solution.
    '''
    sum = 0.
    for i in range(1,m):
        sum += ad_bn(i,beta,c)[0] * np.exp(-(c**2)*(i**2)*(np.pi**2)*t)*np.sin(i*np.pi*x)

    return np.exp(-(beta**2)*t/(4*(c**2)))*np.exp(beta*x/(2*c**2))*sum

# Transport 1d

def transport_u0(x):
    return jnp.exp(-x**2)

def transport_sol(x,t,c):
    return transport_u0(x-c*t)

# Laplace 2d

def u_lap(x):
    return np.exp(-4*((x[:,0]-0.5)**2+(x[:,1]-0.5)**2))

def f_lap(x):
    A = -16*np.exp(-4*((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2))
    B = 64*np.exp(-4*((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2))*((x[:,0]-0.5)**2)
    C = 64*np.exp(-4*((x[:,0]-0.5)**2 + (x[:,1]-0.5)**2))*((x[:,1]-0.5)**2)
    return A+B+C

# Fokker-Planck 1d

def M(x):
    return np.exp(-np.abs(x)**2/2)

def fp_1d_sol(t,x):
    return 1 / ((np.sqrt(2*np.pi*(1-np.exp(-2*t))))) * np.exp( (-np.abs(x)**2) / (2*(1-np.exp(-2*t))) )

def fp_1d_u0(x):
    return fp_1d_sol(0.1,x)

# Fokker-Planck 2d

def M2d(x,y):
    return np.exp(-(x**2 + y**2)/2)

def fp_2d_sol(t,x,y):
    K = 2*(1-np.exp(-2*t))
    return 1/(K*np.pi)*np.exp(-(np.power(x,2) + np.power(y,2))/K)
    
def fp_2d_u0(x,y):
    return fp_2d_sol(0.1,x,y)

# 1D Heat Equation with variable coeff

def a(x):
	return jnp.square(x)

def heat_var_u0(x):
    return jnp.sine(jnp.pi*x)

def heat_var_sol(t,x):
    return 0
    

# ode
def ODE_solutions(X, k, d, c, m = 3):
	
    N = len(X)

    u    = np.zeros((N,m))
    u[:,0] = (np.sin(k*np.pi*X))
    u[:,1] = (X**d + 1.0)
    u[:,2] = (c*np.exp(X))
    
    u_dot  = np.zeros((N,m))
    u_dot[:,0] = (k*np.pi*np.cos(k*np.pi*X))
    u_dot[:,1] = (d*X**(d-1))
    u_dot[:,2] = (c*np.exp(X))
    
    u_ddot = np.zeros((N,m))
    u_ddot[:,0] = (-(k**2)*(np.pi**2)*np.sin(k*np.pi*X))
    u_ddot[:,1] = (d*(d-1)*X**(d-2))
    u_ddot[:,2] = (c*np.exp(X))
    
    return u, u_dot, u_ddot


# Darcy's flow
def u1(X0,X1,k):
    return jnp.sin(k*jnp.pi*X0 + k*jnp.pi*X1)

def u2(X0,X1,d):
    return jnp.array((X0-0.5)**d + (X1-0.5)**d)

def u3(X0,X1,c):
    return c*jnp.exp(X0+ X1)

def a(X0,X1,k=1):
    t1 = jnp.exp(jnp.sin(jnp.pi*X0)+jnp.sin(jnp.pi*X1))
    t2 = jnp.exp(-jnp.sin(jnp.pi*X0)-jnp.sin(jnp.pi*X1))
    return t1 + t2

def u_dot(u, T, params, arg):
	u_Dot = jit(grad(u,arg))
	return vmap(lambda t : u_Dot(t[0], t[1], params))(T)

def u_ddot(u, T, params, arg1, arg2):
	u_2Dot = jit(grad(grad(u,arg1),arg2))
	return vmap(lambda t : u_2Dot(t[0], t[1], params))(T)

def u_dddot(u, T, params, arg1, arg2, arg3):
	u_3Dot = jit(grad(grad(grad(u,arg1),arg2),arg3))
	return vmap(lambda t : u_3Dot(t[0], t[1], params))(T)

def u_ddddot(u, T, params, arg1, arg2, arg3, arg4):
	u_4Dot = jit(grad(grad(grad(grad(u,arg1),arg2),arg3),arg4))
	return vmap(lambda t : u_4Dot(t[0], t[1], params))(T)


def darcy_solutions(X, k, d, c, m=3):
    
    N = len(X)
    
    # u
    u = np.zeros((N,m))
    u[:,0] = u1(X[:,0], X[:,1], k)
    u[:,1] = u2(X[:,0], X[:,1], d)
    u[:,2] = u3(X[:,0], X[:,1], c)
    
    # u_x
    u_x = np.zeros((N,m))
    u_x[:,0] = u_dot(u1, X, k, 0)
    u_x[:,1] = u_dot(u2, X, d, 0)
    u_x[:,2] = u_dot(u3, X, c, 0)
    
    # u_y
    u_y = np.zeros((N,m))
    u_y[:,0] = u_dot(u1, X, k, 1)
    u_y[:,1] = u_dot(u2, X, d, 1)
    u_y[:,2] = u_dot(u3, X, c, 1)

    # u_xx
    u_xx = np.zeros((N,m))
    u_xx[:,0] = u_ddot(u1, X, k, 0, 0)
    u_xx[:,1] = u_ddot(u2, X, d, 0, 0)
    u_xx[:,2] = u_ddot(u3, X, c, 0, 0)

    # u_yy
    u_yy = np.zeros((N,m))
    u_yy[:,0] = u_ddot(u1, X, k, 1, 1)
    u_yy[:,1] = u_ddot(u2, X, d, 1, 1)
    u_yy[:,2] = u_ddot(u3, X, c, 1, 1)

    # a
    a_vals = a(X[:,0], X[:,1])

    # a_x
    a_x = u_dot(a, X, 2, 0)
    
    # a_y
    a_y = u_dot(a, X, 2, 1)

    return u, u_x, u_y, u_xx, u_yy, a_vals, a_x, a_y


def rel_mse(true, pred, root = True):

    '''
    true: Array of ground truth. 
    pred: Array of predictions. 
    root: If True, it computes the relative root mse.  
    '''
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(true))
    error = num/den
    if root:
        error = np.sqrt(error)
    return error