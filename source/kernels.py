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
	KoverMx_Dot = jacfwd(KoverMx2D,argnums=[0,1])
	return vmap(lambda t: vmap(lambda t_: KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params):
	KoverMx_Dot = jacfwd(KoverMx2D,argnums=[0,1])
	return KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)

def MtimesPartialx_2D(T, T_, kernel, params): 
	return vmap(lambda t: vmap(lambda t_: M_2D(t[0],t[1])*jnp.array(partial_KoverMx2D(t[0],t[1],t_[0],t_[1],kernel,params)))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def MtimesPartialx2D(t1,t2,t1_,t_2,kernel,params):
	return M_2D(t1,t2)*jnp.array(partial_KoverMx2D(t1,t2,t1_,t_2,kernel,params))

def partial_MtimesPartialx_2D(T,T_,kernel,params):
	Dot_KoverMx_Dot = jacfwd(MtimesPartialx2D,argnums=[0,1])
	return vmap(lambda t: vmap(lambda t_: jnp.trace(jnp.array(Dot_KoverMx_Dot(t[0],t[1], t_[0],t_[1],kernel,params))))(T_))(T) 

# This version of the function only takes scalars as inputs, not arrays
def partial_MtimesPartialx2D(t1,t2,t1_,t2_,kernel,params):
	Dot_KoverMx_Dot = jacfwd(MtimesPartialx2D,argnums=[0,1])
	return jnp.trace(jnp.array(Dot_KoverMx_Dot(t1,t2,t1_,t2_,kernel,params)))



def KoverMy2D(t1,t2,t1_,t2_,kernel,params):
	return kernel(t1,t2,t1_,t2_,params) / M_2D(t1_,t2_)

def partial_KoverMy2D(t1,t2,t1_,t2_,kernel,params):
	KoverMy_Dot = jacfwd(KoverMy2D,argnums=[2,3])
	return KoverMy_Dot(t1,t2,t1_,t2_,kernel,params)

def MtimesPartialy_2D(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: M_2D(t_[0],t_[1])*jnp.array(partial_KoverMy2D(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def MtimesPartialy2D(t1,t2,t1_,t2_,kernel,params):
	return M_2D(t1_,t2_)*jnp.array(partial_KoverMy2D(t1,t2,t1_,t2_,kernel,params))

def partial_MtimesPartialy_2D(T,T_,kernel,params):
	Dot_KoverMy_Dot = jacfwd(MtimesPartialy2D,argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.trace(jnp.array(Dot_KoverMy_Dot(t[0],t[1], t_[0],t_[1],kernel,params))))(T_))(T)

# This version of the function only takes scalars as inputs, not arrays
def partial_MtimesPartialy2D(t1,t2,t1_,t2_,kernel,params):
	Dot_KoverMy_Dot = jacfwd(MtimesPartialy2D,argnums=[2,3])
	return Dot_KoverMy_Dot(t1,t2,t1_,t2_,kernel,params)


# Build the biggest term

def partialx_MtimesPartialx_overMy2D(t1,t2,t1_,t2_,kernel,params):
	return jnp.array(partial_MtimesPartialx2D(t1,t2,t1_,t2_,kernel,params))/M_2D(t1_,t2_)

def partialy_partialx_MtimesPartialx_overMy_timesMy2D(t1,t2,t1_,t2_,kernel,params):
	Dot_y = jacfwd(partialx_MtimesPartialx_overMy2D, argnums=[2,3])
	return jnp.array(Dot_y(t1,t2,t1_,t2_,kernel,params)) / M_2D(t1_,t2_)

def big_term2D(T,T_,kernel,params):
	final = jacfwd(partialy_partialx_MtimesPartialx_overMy_timesMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.trace(jnp.array(final(t[0],t[1], t_[0],t_[1],kernel,params))))(T_))(T)



#  block 1,2 : bottom : first_term12 - h*second_term12
def first_term_12(T,T_,kernel,params):
	first_term = jacfwd(KoverMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,-1.]), first_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def second_term12(T,T_,kernel,params):
	second_term = jacfwd(partialx_MtimesPartialx_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,-1.]), second_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)


#  block 1,3 : right : first_term13 - h*second_term13
def first_term_13(T,T_,kernel,params):
	first_term = jacfwd(KoverMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([1.,0.]), first_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def second_term13(T,T_,kernel,params):
	second_term = jacfwd(partialx_MtimesPartialx_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([1.,0.]), second_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

#  block 1,4 : top : first_term14 - h*second_term14
def first_term_14(T,T_,kernel,params):
	first_term = jacfwd(KoverMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), first_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def second_term14(T,T_,kernel,params):
	second_term = jacfwd(partialx_MtimesPartialx_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), second_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

#  block 1,5 : top : first_term15 - h*second_term15
def first_term_15(T,T_,kernel,params):
	first_term = jacfwd(KoverMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), first_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def second_term15(T,T_,kernel,params):
	second_term = jacfwd(partialx_MtimesPartialx_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), second_term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 2,2 : bottom-bottom

def partialx_nbottom_overMy2D(t1,t2,t1_,t2_,kernel,params):
	return jnp.dot(jnp.array([0.,-1.]),jnp.array(partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params)))/M_2D(t1_,t2_)

def partialy_nbottom_partialx_nbottom_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nbottom_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,-1.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 2,3

def partialy_nright_partialx_nbottom_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nbottom_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([1.,0.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 2,4

def partialy_ntop_partialx_nbottom_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nbottom_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 2,5

def partialy_nleft_partialx_nbottom_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nbottom_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 3,3 : right-right

def partialx_nright_overMy2D(t1,t2,t1_,t2_,kernel,params):
	return jnp.dot(jnp.array([1.,0.]),jnp.array(partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params)))/M_2D(t1_,t2_)

def partialy_nright_partialx_nright_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nright_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([1.,0.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 3,4

def partialy_ntop_partialx_nright_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nright_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 3,5

def partialy_nleft_partialx_nright_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nright_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 4,4 : top-top

def partialx_ntop_overMy2D(t1,t2,t1_,t2_,kernel,params):
	return jnp.dot(jnp.array([0.,1.]),jnp.array(partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params)))/M_2D(t1_,t2_)

def partialy_ntop_partialx_ntop_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_ntop_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 4,5

def partialy_nleft_partialx_ntop_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_ntop_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

# block 5,5 : left-left

def partialx_nleft_overMy2D(t1,t2,t1_,t2_,kernel,params):
	return jnp.dot(jnp.array([-1.,0.]),jnp.array(partial_KoverMx2D(t1,t2,t1_,t2_,kernel,params)))/M_2D(t1_,t2_)

def partialy_nleft_partialx_nleft_overMy2D(T, T_,kernel,params):
	term = jacfwd(partialx_nleft_overMy2D, argnums=[2,3])
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), term(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)


# Boundary terms

# top term can be built as it is

def partialx_nbottom_overMx_2D(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,-1.]), partial_KoverMx2D(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def partialx_nright_overMx_2D(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([1.,0.]), partial_KoverMx2D(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def partialx_ntop_overMx_2D(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([0.,1.]), partial_KoverMx2D(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)

def partialx_nleft_overMx_2D(T,T_,kernel,params):
	return vmap(lambda t: vmap(lambda t_: jnp.dot(jnp.array([-1.,0.]), partial_KoverMx2D(t[0],t[1], t_[0],t_[1],kernel,params)))(T_))(T)