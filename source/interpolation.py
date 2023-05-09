import numpy as np
from kernels import *
from parameter_learning import *

def kernel_parameters(X_train,U_train):
    '''
    X_train: N x d array with collocation points.
    U_train: N x m array with values of u at X_train.
    '''
    m = U_train.shape[1] # Number of functions
    N = len(X_train)
    optim_sgm = np.zeros(m)
    optim_lmbd = np.zeros(m)
    alphas = np.zeros((N,m))
    for i in range(m):
        optim_sgm[i],optim_lmbd[i] = grid_search_RBF(X_train,U_train[:,i].reshape(-1,1))
        G = K(Gaussian,X_train,X_train,optim_sgm[i]) 
        M = (G + optim_lmbd[i]*jnp.eye(N))
        alphas[:,i] = jnp.linalg.solve(M,U_train[:,i])
    return optim_sgm, alphas, optim_lmbd

def predictions(X, X_train, kernel, optim_sgm, alphas):
    m = len(optim_sgm)
    N = len(X)
    u_pred      = np.zeros((N,m))
    u_dot_pred  = np.zeros((N,m))
    u_ddot_pred = np.zeros((N,m))
    for i in range(m):
        u_pred[:,i]      = np.dot(K(kernel, X, X_train, optim_sgm[i]), alphas[:,i])
        u_dot_pred[:,i]  = np.dot(K_dot(kernel, X, X_train, optim_sgm[i], 0), alphas[:,i])
        u_ddot_pred[:,i] = np.dot(K_ddot(kernel, X, X_train, optim_sgm[i], 0, 0), alphas[:,i])
    
    return u_pred, u_dot_pred, u_ddot_pred

def predictions_darcy(X, X_train, kernel, optim_sgm, alphas):
    m = len(optim_sgm)
    N = len(X)
    u_pred   = np.zeros((N,m))
    u_x_pred = np.zeros((N,m))
    u_y_pred = np.zeros((N,m))
    u_xx_pred = np.zeros((N,m))
    u_yy_pred = np.zeros((N,m))
    for i in range(m):
        u_pred[:,i]      = np.dot(K_2D(kernel, X, X_train, optim_sgm[i]), alphas[:,i])
        u_x_pred[:,i]  = np.dot(K_dot2D(kernel, X, X_train, optim_sgm[i], 0), alphas[:,i])
        u_y_pred[:,i] = np.dot(K_dot2D(kernel, X, X_train, optim_sgm[i], 1), alphas[:,i])
        u_xx_pred[:,i] = np.dot(K_ddot2D(kernel, X, X_train, optim_sgm[i], 0, 0), alphas[:,i])
        u_yy_pred[:,i] = np.dot(K_ddot2D(kernel, X, X_train, optim_sgm[i], 1, 1), alphas[:,i])
    return u_pred, u_x_pred, u_y_pred, u_xx_pred, u_yy_pred

