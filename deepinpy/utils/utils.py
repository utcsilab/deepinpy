#!/usr/bin/env python

import torch
import numpy as np

import deepinpy.utils.complex as cp

'''
Utility functions """
'''

# TODO: Unused, potentially depreciated
# not used, returns the top k values of a vector along dimension dim (maybe name hard threshold?)
def topk(inp, k, dim):
    _topk, _idx = torch.topk(abs(inp), k, dim=dim)
    _topk = torch.gather(inp, dim, _idx).sign() * _topk
    out = 0*inp
    return out.scatter_(dim, _idx, _topk)

# torch to numpy with complex values
def t2n(x):
    return cp.r2c(t2n2(x))

# only goes from torch to numpy, helper for t2n
def t2n2(x):
    return np.array(x.detach().cpu())

def itemize(x):
    if x.shape[0] > 1:
        return [xx.item() for xx in x]
    else:
        return x.item()

# Different fftship, but multiplies every other line by -1 which is the same as e^j*pi so same as n/2 shift, modulation basically (modulating the output by +/- pi)
def fftmod(out):
    out2 = out.copy()
    out2[...,::2,:] *= -1
    out2[...,:,::2] *= -1
    out2 *= -1
    return out2

# FIXME: Take axes as input instead of hardcoding
# axes definition with axes, apply shift along last two dimensions of vector, but should be able to take axes as an input
def fftshift(x):
    axes = (-2, -1)
    return scipy.fftpack.fftshift(x, axes=axes)

# FIXME: Take axes as input instead of hardcoding
def ifftshift(x):
    axes = (-2, -1)
    return scipy.fftpack.ifftshift(x, axes=axes)

# c = centered
def fft2c(x):
    return fftshift(fft2(ifftshift(x)))

def ifft2c(x):
    return ifftshift(ifft2(fftshift(x)))

# uc = unitary centered
def fft2uc(x):
    return fft2c(x) / np.sqrt(np.prod(x.shape[-2:]))

def ifft2uc(x):
    return ifft2c(x) * np.sqrt(np.prod(x.shape[-2:]))

# FIXME: Take axes as input instead of hardcoding
def fft2(x):
    axes = (-2, -1)
    return scipy.fftpack.fft2(x, axes=axes)

# FIXME: Take axes as input instead of hardcoding
def ifft2(x):
    axes = (-2, -1)
    return scipy.fftpack.ifft2(x, axes=axes)
