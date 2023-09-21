import jax.numpy as jnp
from jax import grad, jacfwd, vmap, jit
from functools import partial

#	1D 

# Gaussian Kernel

def Gaussian(t,t_,params):
	sigma = params
	r2 = jnp.dot(t-t_,t-t_)
	return jnp.exp(-r2/(2*sigma**2))

# Polynomial Kernel

def Polynomial(t,t_,params):
	c0,d = params
	return (jnp.dot(t,t_)+c0)**d

# Matern Kernel

def Matern_Kernel_0(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2
	return coef * jnp.exp(-d/rho)

def Matern_Kernel_1(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2 * (1 + (jnp.sqrt(3)*d/rho))
	return coef * jnp.exp(-jnp.sqrt(3)*d/rho)

def Matern_Kernel_2(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2 * (1 + (jnp.sqrt(5)*d/rho) + (5*d**2/3*rho**2))
	return coef * jnp.exp(-jnp.sqrt(5)*d/rho)

# Kernel Matrices

def K(kernel, T, T_, params):
	return vmap(lambda t: vmap(lambda t_: kernel(t,t_, params))(T_))(T)

def K_dot(kernel, T ,T_, params, arg):
	K_Dot = jit(grad(kernel,arg))
	return vmap(lambda t: vmap(lambda t_: K_Dot(t, t_, params))(T_))(T)

def K_ddot(kernel, T ,T_, params, arg1, arg2):
	K_2Dot = jit(grad(grad(kernel,arg1),arg2))
	return vmap(lambda t: vmap(lambda t_: K_2Dot(t ,t_, params))(T_))(T)

def K_dddot(kernel, T ,T_, params, arg1, arg2, arg3):
	K_3Dot = jit(grad(grad(grad(kernel,arg1),arg2),arg3))
	return vmap(lambda t: vmap(lambda t_: K_3Dot(t ,t_, params))(T_))(T)
 
def K_ddddot(kernel, T ,T_, params, arg1, arg2, arg3, arg4):
	K_4Dot = jit(grad(grad(grad(grad(kernel,arg1),arg2),arg3),arg4))
	return vmap(lambda t: vmap(lambda t_: K_4Dot(t ,t_, params))(T_))(T)

# Kernel Matrices - Fokker-Planck 1d

def M(x):
	return jnp.exp(-x**2/2)

def KoverMx(t,t_,kernel,params):
	return kernel(t,t_,params) / M(t)

def partial_KoverMx(t,t_,kernel,params):
	KoverMx_Dot = grad(KoverMx,0)
	return KoverMx_Dot(t,t_,kernel,params)

def MtimesPartialx(t,t_,kernel,params):
	return M(t)*partial_KoverMx(t,t_,kernel,params)

def partial_MtimesPartialx(T,T_,kernel,params):
	Dot_KoverMx_Dot = grad(MtimesPartialx,0)
	return vmap(lambda t: vmap(lambda t_: Dot_KoverMx_Dot(t,t_,kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_MtimesPartialx_(t,t_,kernel,params):
	Dot_KoverMx_Dot = grad(MtimesPartialx,0)
	return Dot_KoverMx_Dot(t,t_,kernel,params)



def KoverMy(t,t_,kernel,params):
	return kernel(t,t_,params) / M(t_)

def partial_KoverMy(t,t_,kernel,params):
	KoverMx_Dot = grad(KoverMy,1)
	return KoverMx_Dot(t,t_,kernel,params)

def MtimesPartialy(t,t_,kernel,params):
	return M(t_)*partial_KoverMy(t,t_,kernel,params)

def partial_MtimesPartialy(T,T_,kernel,params):
	Dot_KoverMy_Dot = grad(MtimesPartialy,1)
	return vmap(lambda t: vmap(lambda t_: Dot_KoverMy_Dot(t,t_,kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_MtimesPartialy_(t,t_,kernel,params):
	Dot_KoverMy_Dot = grad(MtimesPartialy,1)
	return Dot_KoverMy_Dot(t,t_,kernel,params)


# Build the biggest term

def partialx_MtimesPartialx_overMy(t,t_,kernel,params):
	Dot_KoverMx_Dot = grad(MtimesPartialx,0)
	return Dot_KoverMx_Dot(t,t_,kernel,params)/M(t_)

def partialy_partialx_MtimesPartialx_overMy_timesM(t,t_,kernel,params):
	Dot_y = grad(partialx_MtimesPartialx_overMy,1)
	return Dot_y(t,t_,kernel,params) * M(t_)


def big_term(T,T_,kernel,params):
	final = grad(partialy_partialx_MtimesPartialx_overMy_timesM, 1)
	return vmap(lambda t: vmap(lambda t_: final(t,t_,kernel,params))(T_))(T)
 



#	2D

# Gaussian Kernel

def Gaussian2D(x1,x2,y1,y2,params):
  sigma = params
  r2 = ((x1-y1)**2 + (x2-y2)**2)
  return jnp.exp(-r2/(2*sigma**2))

# Kernel matrices
def K_2D(kernel, T,T_, params):
  return vmap(lambda t: vmap(lambda t_: kernel(t[0],t[1], t_[0],t_[1], params))(T_))(T)

def K_dot2D(kernel, T ,T_, params, arg):
  K_Dot = jit(grad(kernel,arg))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)

def K_ddot2D(kernel, T ,T_, params, arg1, arg2):
  K_Dot = jit(grad(grad(kernel,arg1),arg2))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)
 
def K_dddot2D(kernel, T ,T_, params, arg1, arg2, arg3):
  K_Dot = jit(grad(grad(grad(kernel,arg1),arg2),arg3))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)
 
def K_ddddot2D(kernel, T ,T_, params, arg1, arg2, arg3, arg4):
  K_Dot = jit(grad(grad(grad(grad(kernel,arg1),arg2),arg3),arg4))
  return vmap(lambda t: vmap(lambda t_: K_Dot(t[0],t[1], t_[0],t_[1], params))(T_))(T)
