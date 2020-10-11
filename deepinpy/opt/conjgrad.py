#!/usr/bin/env python

import torch

from deepinpy.utils import utils
from deepinpy.opt import ip_batch, dot_batch

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
        x, num_cg = conjgrad(x, self.rhs, self.Aop_fun, max_iter=self.max_iter, l2lam=self.l2lam, eps=self.eps, verbose=self.verbose)
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
    """A function that implements batched conjugate gradient descent; assumes the first index is batch size.

    Args:
    	x (Tensor): The initial input to the algorithm.
    b (Tensor): The residual vector
    Aop_fun (func): A function performing the A matrix operation.
    max_iter (int): Maximum number of times to run conjugate gradient descent.
    l2lam (float): The L2 lambda, or regularization parameter (must be positive).
    eps (float): Determines how small the residuals must be before termination…
    verbose (bool): If true, prints extra information to the console.

    Returns:
    	A tuple containing the updated vector x and the number of iterations performed.
    """

    # explicitly remove r from the computational graph
    r = b.new_zeros(b.shape, requires_grad=False)

    # the first calc of the residual may not be necessary in some cases...
    if l2lam > 0:
        r = b - (Aop_fun(x) + l2lam * x)
    else:
        r = b - Aop_fun(x)
    p = r

    rsnot = ip_batch(r)
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

        if l2lam > 0:
            Ap = Aop_fun(p) + l2lam * p
        else:
            Ap = Aop_fun(p)
        pAp = dot_batch(p, Ap)

        #print(utils.itemize(pAp))

        alpha = (rsold / pAp).reshape(reshape)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = ip_batch(r)

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r
        num_iter += 1


    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x, num_iter
