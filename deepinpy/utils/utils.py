#!/usr/bin/env python

import torch
import numpy as np

import deepinpy.utils.complex as cp

'''
Utility functions
'''

def topk(inp, k, dim):
    _topk, _idx = torch.topk(abs(inp), k, dim=dim)
    _topk = torch.gather(inp, dim, _idx).sign() * _topk
    out = 0*inp
    return out.scatter_(dim, _idx, _topk)

def t2n(x):
    return cp.r2c(t2n2(x))

def t2n2(x):
    return np.array(x.detach().cpu())

def itemize(x):
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()
