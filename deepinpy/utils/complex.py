#!/usr/bin/env python

import numpy as np
import torch

'''
Defines complex-valued arithmetic for ndarrays, where the real and imaginary
channels are stored in the last dimension
'''

def c2r(z):
    ''' Convert from complex to 2-channel real '''
    assert type(z) is np.ndarray, 'Must be numpy.ndarray'
    return np.stack((z.real, z.imag), axis=-1)

def r2c(x):
    ''' Convert from 2-channel real to complex '''
    assert type(x) is np.ndarray, 'Must be numpy.ndarray'
    return x[...,0] + 1j *  x[...,1]

def zmul(x1, x2):
    ''' complex-valued multiplication '''
    xr = x1[...,0] * x2[...,0] -  x1[...,1] * x2[...,1]
    xi = x1[...,0] * x2[...,1] +  x1[...,1] * x2[...,0]
    if type(x1) is np.ndarray:
        return np.stack((xr, xi), axis=-1)
    elif type(x1) is torch.Tensor:
        return torch.stack((xr, xi), dim=-1)
    else:   
        return xr, xi

def zconj(x):
    ''' complex-valued conjugate '''
    if type(x) is np.ndarray:
        return np.stack((x[...,0], -x[...,1]), axis=-1)
    elif type(x) is torch.Tensor:
        return torch.stack((x[...,0], -x[...,1]), dim=-1)
    else:   
        return x[...,0], -x[...,1]

def zabs(x):
    ''' complex-valued magnitude '''
    if type(x) is np.ndarray:
        return np.sqrt(zmul(x, zconj(x)))[...,0]
    elif type(x) is torch.Tensor:
        return torch.sqrt(zmul(x, zconj(x)))
    else:   
        return -1.
