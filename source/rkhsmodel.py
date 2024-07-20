import jax.numpy as jnp
from jax import jit,jacrev
import jax
from functools import partial
from tools import get_kernel_block_ops,eval_k,diagpart
from jax.scipy.linalg import block_diag,cholesky,solve_triangular
from typing import Optional

class InducedRKHS():
    """
    Still have to go back and allow for multiple operator sets
        For example, points on boundary only need evaluation, not the rest of the operators if we know boundary conditions
    This only does 1 dimensional output for now. 
    """
    def __init__(
            self,
            basis_points,
            operators,
            kernel_function,
            ) -> None:
        # collocation points
        self.basis_points = basis_points
        # phi
        self.operators = operators
        # kernel
        self.k = kernel_function
        # k(phi,phi) function
        self.get_all_op_kernel_matrix = jit(get_kernel_block_ops(self.k,self.operators,self.operators))
        # k(delta,phi) function
        self.get_eval_op_kernel_matrix = jit(get_kernel_block_ops(self.k,[eval_k],self.operators))
        # k(phi,phi)[basis_pts, basis_pts]
        self.kmat = self.get_all_op_kernel_matrix(self.basis_points,self.basis_points)
        # dimension alpha
        self.num_params = len(basis_points) * len(operators)
    
    @partial(jit, static_argnames=['self'])
    def evaluate_all_ops(self,eval_points,params): # k(phi,phi)[eval_pts, basis_pts] @ alpha ---> why ?
        return self.get_all_op_kernel_matrix(eval_points,self.basis_points)@params
    
    @partial(jit, static_argnames=['self'])
    def point_evaluate(self,eval_points,params): # k(delta,phi)[eval_pts, basis_pts] @ alpha ---> ok
        return self.get_eval_op_kernel_matrix(eval_points,self.basis_points)@params
    
    @partial(jit, static_argnames=['self','operators'])
    def evaluate_operators(self,operators,eval_points,params): # k(phi*,phi)[eval_pts, basis_pts] @ alpha ---> why ?
        return get_kernel_block_ops(self.k,operators,self.operators)(eval_points,self.basis_points)@params
    
    def get_fitted_params(self,X,y,lam = 1e-6,eps = 1e-4): # K(phi,phi)[X, basis_pts]^{-1} @ y --- why X and not basis_pts ?
        K = self.get_eval_op_kernel_matrix(X,self.basis_points)
        coeffs = jax.scipy.linalg.solve(K.T@K + lam * (self.kmat+eps * diagpart(self.kmat)),K.T@y,assume_a = 'pos')
        return coeffs
    
    def get_eval_function(self,params): 
        def u(x): 
            return (self.get_eval_op_kernel_matrix(x.reshape(1,-1),self.basis_points)@params)[0]
        return u
    
    def get_damping(self):
        return self.kmat