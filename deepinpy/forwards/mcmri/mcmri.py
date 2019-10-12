#!/usr/bin/env python

import numpy as np
import torch

import deepinpy.utils.complex as cp

class MultiChannelMRI(torch.nn.Module):
    def __init__(self, maps, mask, l2lam=False):
        super(MultiChannelMRI, self).__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam

        #if normal is None:
            #self.normal_fun = self._normal
        #else:
            #self.normal_fun = normal

    def forward(self, x):
        return sense_forw(x, self.maps, self.mask)

    def adjoint(self, y):
        return sense_adj(y, self.maps, self.mask)

    def normal(self, x):
        out = self.adjoint(self.forward(x))
        if self.l2lam:
            out = out + self.l2lam * x
        return out

    #def normal(self, x):
        #return self.normal_fun(x)

def maps_forw(img, maps):
    return cp.zmul(img[:,None,:,:,:], maps)

def maps_adj(cimg, maps):
    return torch.sum(cp.zmul(cp.zconj(maps), cimg), 1, keepdim=False)

def fft_forw(x, ndim=2):
    return torch.fft(x, signal_ndim=ndim, normalized=True)

def fft_adj(x, ndim=2):
    return torch.ifft(x, signal_ndim=ndim, normalized=True)

def mask_forw(y, mask):
    return y * mask[:,None,:,:,None]

def sense_forw(img, maps, mask):
    return mask_forw(fft_forw(maps_forw(img, maps)), mask)

def sense_adj(ksp, maps, mask):
    return maps_adj(fft_adj(mask_forw(ksp, mask)), maps)

def sense_normal(img, maps, mask):
    return maps_adj(fft_adj(mask_forw(fft_forw(maps_forw(img, maps)), mask)), maps)
