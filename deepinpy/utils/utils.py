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

def fftmod(out):
    out2 = out.copy()
    out2[...,::2,:] *= -1
    out2[...,:,::2] *= -1
    out2 *= -1
    return out2

def fftshift(x):
    axes = (-2, -1)
    return scipy.fftpack.fftshift(x, axes=axes)

def ifftshift(x):
    axes = (-2, -1)
    return scipy.fftpack.ifftshift(x, axes=axes)

def fft2c(x):
    return fftshift(fft2(ifftshift(x)))

def ifft2c(x):
    return ifftshift(ifft2(fftshift(x)))

def fft2uc(x):
    return fft2c(x) / np.sqrt(np.prod(x.shape[-2:]))

def ifft2uc(x):
    return ifft2c(x) * np.sqrt(np.prod(x.shape[-2:]))

def fft2(x):
    axes = (-2, -1)
    return scipy.fftpack.fft2(x, axes=axes)

def ifft2(x):
    axes = (-2, -1)
    return scipy.fftpack.ifft2(x, axes=axes)

