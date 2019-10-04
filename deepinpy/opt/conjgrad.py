#!/usr/bin/env python

import torch
import numpy as np
import sys

import deepinpy.utils.utils
import deepinpy.utils.complex as cp
from deepinpy.opt.opt import ip_batch, dot_batch

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

    for i in range(max_iter):

        if verbose:
            print('{i}: {rsnew}'.format(i=i, rsnew=deepinpy.utils.utils.itemize(torch.sqrt(rsnew))))

        if rsnew.max() < eps:
            break

        if l2lam > 0:
            Ap = Aop_fun(p) + l2lam * p
        else:
            Ap = Aop_fun(p)
        pAp = dot_batch(p, Ap)

        #print(deepinpy.utils.itemize(pAp))

        alpha = (rsold / pAp).reshape(reshape)

        x = x + alpha * p
        r = r - alpha * Ap

        rsnew = ip_batch(r)

        beta = (rsnew / rsold).reshape(reshape)

        rsold = rsnew

        p = beta * p + r


    if verbose:
        print('FINAL: {rsnew}'.format(rsnew=torch.sqrt(rsnew)))

    return x
