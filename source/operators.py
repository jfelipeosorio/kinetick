from jax import grad, jacfwd, hessian, jit
from jax.numpy import trace



def laplacian_k(k,index):
    def lapk(*args):
        return trace(hessian(k,index)(*args))
    return lapk

# What is this doing vs. get_grad
def get_selected_grad(k,index,selected_index):
    gradf = grad(k,index)
    def selgrad(*args):
        return gradf(*args)[selected_index]
    return selgrad

def dx_k(k,index):
    return get_selected_grad(k,index,0)

# Another divergence option ?
def get_selected_divergence(k,index,selected_index):
    divergencef = jit(lambda x: trace(jacfwd(k,index)(x)))
    def seldiv(*args):
        return divergencef(*args)[selected_index]
    return seldiv
def divergence(k,index):
    return get_selected_divergence(k,index,0)