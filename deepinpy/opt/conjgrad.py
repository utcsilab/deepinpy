#!/usr/bin/env python

import torch

from deepinpy.utils import utils
from deepinpy.opt import dot_batch, dot_single_batch, zdot_batch, zdot_single_batch

class ConjGrad(torch.nn.Module):
    """A class which implements conjugate gradient descent as a torch module.

    This implementation of conjugate gradient descent works as a standard torch module, with the functions forward
    and get_metadata overridden. It is used as an optimization block within a Recon object.

    Args:
        rhs (Tensor): The residual vector b in some conjugate gradient descent algorithms.
        Aop_fun (func): A function performing the A matrix operation.
        max_iter (int): Maximum number of times to run conjugate gradient descent.
        l2lam (float): The L2 lambda, or regularization parameter (must be positive).
        eps (float): Determines how small the residuals must be before termination.
        verbose (bool): If true, prints extra information to the console.

    Attributes:
        rhs (Tensor): The residual vector, b in some conjugate gradient descent algorithms.
        Aop_fun (func): A function performing the A matrix operation.
        max_iter (int): The maximum number of times to run conjugate gradient descent.
        l2lam (float): The L2 lambda regularization parameter.
        eps (float): Minimum residuals for termination.
        verbose (bool): Whether or not to print extra info to the console.
    """

    def __init__(self, rhs, Aop_fun, max_iter=20, l2lam=0., eps=1e-6, verbose=True):
        super(ConjGrad, self).__init__()

        self.rhs = rhs
        self.Aop_fun = Aop_fun
        self.max_iter = max_iter
        self.l2lam = l2lam
        self.eps = eps
        self.verbose = verbose

        self.num_cg = None

    def forward(self, x):
        """Performs one forward pass through the conjugate gradient descent algorithm.

        Args:
            x (Tensor): The input to the gradient algorithm.

        Returns:
            The forward pass on x.

        """
        x, num_cg = zconjgrad(x, self.rhs, self.Aop_fun, max_iter=self.max_iter, l2lam=self.l2lam, eps=self.eps, verbose=self.verbose)
        self.num_cg = num_cg
        return x

    def get_metadata(self):
        """Accesses metadata for the algorithm.

        Returns:
            A dict containing metadata.
        """

        return {
                'num_cg': self.num_cg,
                }

def conjgrad(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-4, verbose=True):
    """Conjugate Gradient Algorithm applied to batches; assumes the first index is batch size.

    Args:
    x (Tensor): The initial input to the algorithm.
    b (Tensor): The residual vector
    Aop_fun (func): A function performing the normal equations, A.H * A
    max_iter (int): Maximum number of times to run conjugate gradient descent.
    l2lam (float): The L2 lambda, or regularization parameter (must be positive).
    eps (float): Determines how small the residuals must be before termination…
    verbose (bool): If true, prints extra information to the console.

    Returns:
    	A tuple containing the output vector x and the number of iterations performed.
    """

    return conjgrad_priv(x, b, Aop_fun, max_iter=max_iter, l2lam=l2lam, eps=eps, verbose=verbose, complex=False)





def zconjgrad(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-4, verbose=True):
    """Conjugate Gradient Algorithm for a complex vector space applied to batches; assumes the first index is batch size.

    Args:
    x (complex-valued Tensor): The initial input to the algorithm.
    b (complex-valued Tensor): The residual vector
    Aop_fun (func): A function performing the normal equations, A.H * A
    max_iter (int): Maximum number of times to run conjugate gradient descent.
    l2lam (float): The L2 lambda, or regularization parameter (must be positive).
    eps (float): Determines how small the residuals must be before termination…
    verbose (bool): If true, prints extra information to the console.

    Returns:
    	A tuple containing the output vector x and the number of iterations performed.
    """

    return conjgrad_priv(x, b, Aop_fun, max_iter=max_iter, l2lam=l2lam, eps=eps, verbose=verbose, complex=True)

def conjgrad_priv(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-4, verbose=True, complex=True):
    """Conjugate Gradient Algorithm applied to batches; assumes the first index is batch size.

    Args:
    x (Tensor): The initial input to the algorithm.
    b (Tensor): The residual vector
    Aop_fun (func): A function performing the normal equations, A.adjoint * A
    max_iter (int): Maximum number of times to run conjugate gradient descent.
    l2lam (float): The L2 lambda, or regularization parameter (must be positive).
    eps (float): Determines how small the residuals must be before termination…
    verbose (bool): If true, prints extra information to the console.
    complex (bool): If true, uses complex vector space

    Returns:
    	A tuple containing the output Tensor x and the number of iterations performed.
    """

    if complex:
        _dot_single_batch = lambda r: zdot_single_batch(r).real
        _dot_batch = lambda r, p: zdot_batch(r, p).real
    else:
        _dot_single_batch = dot_single_batch
        _dot_batch = dot_batch

    # explicitly remove r from the computational graph
    #r = b.new_zeros(b.shape, requires_grad=False, dtype=torch.cfloat)

    # the first calc of the residual may not be necessary in some cases...
    # note that l2lam can be less than zero when training due to finite # of CG iterations
    r = b - (Aop_fun(x) + l2lam * x)
    p = r

    rsnot = _dot_single_batch(r)
    rsold = rsnot
    rsnew = rsnot

    eps_squared = eps ** 2

    reshape = (-1,) + (1,) * (len(x.shape) - 1)

    num_iter = 0

    for i in range(max_iter):

        if verbose:
            print('{i}: {rsnew}'.format(i=i, rsnew=utils.itemize(torch.sqrt(rsnew))))

        if rsnew.max() < eps_squared:
            break

        Ap = Aop_fun(p) + l2lam * p
        pAp = _dot_batch(p, Ap)

        #print(utils.itemize(pAp))

        alpha = (rsold / pAp).reshape(reshape)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = _dot_single_batch(r)

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r
        num_iter += 1


    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x, num_iter
