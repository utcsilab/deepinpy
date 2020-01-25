#!/usr/bin/env python

import torch

from deepinpy.utils import utils
from deepinpy.opt import ip_batch, dot_batch

class ConjGrad(torch.nn.Module):
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
        x, num_cg = conjgrad(x, self.rhs, self.Aop_fun, max_iter=self.max_iter, l2lam=self.l2lam, eps=self.eps, verbose=self.verbose)
        self.num_cg = num_cg
        return x

    def get_metadata(self):
        return {
                'num_cg': self.num_cg,
                }

def conjgrad(x, b, Aop_fun, max_iter=10, l2lam=0., eps=1e-4, verbose=True):
    ''' batched conjugate gradient descent. assumes the first index is batch size '''

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

        if rsnew.max() < eps:
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
