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

def Matern_Kernel_32_1D(t,t_,params):
	rho, sigma = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_))
	coef = sigma**2 * (1 + (jnp.sqrt(3)*d/rho))
	return coef * jnp.exp(-jnp.sqrt(3)*d/rho)

def Matern_Kernel_52_1D(t,t_,params):
	rho = params
	d = jnp.sqrt(jnp.dot(t-t_,t-t_) + 1e-8)
	coef = 1 + (jnp.sqrt(5)*d/rho) + (5*d**2/(3*rho**2))
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

#	2D

# Gaussian Kernel

def Gaussian2D(x1,x2,y1,y2,params):
  sigma = params
  r2 = ((x1-y1)**2 + (x2-y2)**2)
  return jnp.exp(-r2/(2*sigma**2))

@jit
def Matern_Kernel_52_2D(x1,x2,y1,y2,params):
	rho = params
	d = jnp.sqrt(((x1-y1)**2 + 10*(x2-y2)**2) + 1e-8)
	coef = (1 + (jnp.sqrt(5)*d/rho) + (5*d**2/(3*rho**2)))
	return coef * jnp.exp(-jnp.sqrt(5)*d/rho)

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


# Kernel Matrices - Fokker-Planck 1d

def M(x):
	return jnp.exp(-x**2/2)

def KoverMx(t,t_,kernel,params):
	return kernel(t,t_,params) / M(t)

def partial_KoverMx_(T,T_,kernel,params):
	KoverMx_Dot = grad(KoverMx,0)
	return vmap(lambda t: vmap(lambda t_: KoverMx_Dot(t,t_,kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_KoverMx(t,t_,kernel,params):
	KoverMx_Dot = grad(KoverMx,0)
	return KoverMx_Dot(t,t_,kernel,params)

def MtimesPartialx_(T, T_, kernel, params):
	return vmap(lambda t: vmap(lambda t_: M(t)*partial_KoverMx(t,t_,kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
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

def MtimesPartialy_(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: M(t_)*partial_KoverMy(t,t_,kernel,params))(T_))(T)

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
 
# boundary-boundary

def partialx_overMy(t, t_, kernel, params):
	return partial_KoverMx(t,t_,kernel,params) / M(t_)

def partial_partial(T, T_, kernel, params):
	pp = grad(partialx_overMy,1)
	return vmap(lambda t: vmap(lambda t_: pp(t,t_,kernel,params))(T_))(T)

# Kernel Matrices - Fokker-Planck 2d

def M_2D(x1,x2):
	return jnp.exp(-(x1**2/2 + x2**2/2))

def KoverMx2D(t1,t2,t1_,t_2,kernel,params):
	return kernel(t1,t2,t1_,t_2,params) / M_2D(t1,t2)

def partial_KoverMx_2D(T,T_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D,0),1)
	return vmap(lambda t: vmap(lambda t_: KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D, 0), 1)
	return KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)

def MtimesPartialx_2D(T, T_, kernel, params): 
	return vmap(lambda t: vmap(lambda t_: M_2D(t[0],t[1])*partial_KoverMx2D(t[0],t[1],t_[0],t_[1],kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def MtimesPartialx2D(t1,t2,t1_,t_2,kernel,params):
	return M_2D(t1,t2)*partial_KoverMx2D(t1,t2,t1_,t_2,kernel,params)

def partial_MtimesPartialx2D(T,T_,kernel,params):
	Dot_KoverMx_Dot_0 = grad(MtimesPartialx2D, 0)
	Dot_KoverMx_Dot_1 = grad(MtimesPartialx2D, 1)
	return vmap(lambda t: vmap(lambda t_: Dot_KoverMx_Dot_0(t[0],t[1], t_[0],t_[1],kernel,params) + Dot_KoverMx_Dot_1(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T) # partial outside is a divergence

# This version of the function only takes scalars as inputs, not arrays
def partial_MtimesPartialx_2D(t1,t2,t1_,t2_,kernel,params):
	# Dot_KoverMx_Dot = grad(grad(MtimesPartialx2D,0),1)
	Dot_KoverMx_Dot_0 = grad(MtimesPartialx2D,0)
	Dot_KoverMx_Dot_1 = grad(MtimesPartialx2D,1)
	return Dot_KoverMx_Dot_0(t1,t2,t1_,t2_,kernel,params) + Dot_KoverMx_Dot_1(t1,t2,t1_,t2_,kernel,params)



def KoverMy2D(t1,t2,t1_,t2_,kernel,params):
	return kernel(t1,t2,t1_,t2_,params) / M_2D(t1_,t2_)

def partial_KoverMy2D(t1,t2,t1_,t2_,kernel,params):
	KoverMy_Dot = grad(grad(KoverMy2D,1),2)
	return KoverMy_Dot(t1,t2,t1_,t2_,kernel,params)

def MtimesPartialy2D(t1,t2,t1_,t2_,kernel,params):
	return M_2D(t1_,t2_)*partial_KoverMy2D(t1,t2,t1_,t2_,kernel,params)

def MtimesPartialy_2D(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: M_2D(t_[0],t_[1])*partial_KoverMy2D(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T)

def partial_MtimesPartialy2D(T,T_,kernel,params):
	Dot_KoverMy_Dot_1 = grad(MtimesPartialy2D,1)
	Dot_KoverMy_Dot_2 = grad(MtimesPartialy2D,2)
	return vmap(lambda t: vmap(lambda t_: Dot_KoverMy_Dot_1(t[0],t[1], t_[0],t_[1],kernel,params) + Dot_KoverMy_Dot_2(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_MtimesPartialy_2D(t1,t2,t1_,t2_,kernel,params):
	Dot_KoverMy_Dot_1 = grad(MtimesPartialy2D,1)
	Dot_KoverMy_Dot_2 = grad(MtimesPartialy2D,2)
	return Dot_KoverMy_Dot_1(t1,t2,t1_,t2_,kernel,params) + Dot_KoverMy_Dot_2(t1,t2,t1_,t2_,kernel,params)


# Build the biggest term

def partialx_MtimesPartialx_overMy2D(t1,t2,t1_,t2_,kernel,params):
	Dot_KoverMx_Dot = grad(grad(MtimesPartialx2D,0),1)
	return Dot_KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)/M_2D(t1_,t2_)

def partialy_partialx_MtimesPartialx_overMy_timesM2D(t1,t2,t1_,t2_,kernel,params):
	Dot_y = grad(grad(partialx_MtimesPartialx_overMy2D,1),2)
	return Dot_y(t1,t2,t1_,t2_,kernel,params) / M_2D(t1_,t2_)

def big_term2D(T,T_,kernel,params):
	final = grad(grad(partialy_partialx_MtimesPartialx_overMy_timesM2D, 1),2)
	return vmap(lambda t: vmap(lambda t_: final(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T)
 
# boundary-boundary

def partialx_overMy2D(t1,t2,t1_,t2_, kernel, params):
	return partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params) / M_2D(t1_,t2_)

def partial_partial2D(T, T_, kernel, params):
	pp = grad(grad(partialx_overMy2D,1),2)
	return vmap(lambda t: vmap(lambda t_: pp(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T)


# Boundary phis

def bottom_partial_KoverMx_2D(T,T_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D,0),1)
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,-1.]), KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def bottom_partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D, 0), 1)
	return jnp.sum(jnp.dot(jnp.array([0.,-1.]),KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)))

def right_partial_KoverMx_2D(T,T_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D,0),1)
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([1.,0.]), KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def right_partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D, 0), 1)
	return jnp.sum(jnp.dot(jnp.array([1.,0.]),KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)))

def top_partial_KoverMx_2D(T,T_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D,0),1)
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def top_partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D, 0), 1)
	return jnp.sum(jnp.dot(jnp.array([0.,1.]),KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)))

def left_partial_KoverMx_2D(T,T_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D,0),1)
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def left_partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params):
	KoverMx_Dot = grad(grad(KoverMx2D, 0), 1)
	return jnp.sum(jnp.dot(jnp.array([-1.,0.]),KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)))