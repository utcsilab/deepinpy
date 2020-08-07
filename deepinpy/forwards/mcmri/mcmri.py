#!/usr/bin/env python
"""Data transformation and processing class with functions."""

import numpy as np
import torch

import deepinpy.utils.complex as cp

class MultiChannelMRI(torch.nn.Module):
    """Class which implements a forward operator (matrix A) for data processing and transformation.

    This class performs signal processing functions and various transformations such as masking to data passed to it,
    which prepares it for model processing. Because it extends a nn.Module, it can be wired into the Recon as part of
    the System block.

    Args:
        maps (...): The sensitivity profiles for each MRI coil.
        mask (...): The mask to incoherently subsample k-space.
        img_shape (tuple): Dimensions of image.
        use_sigpy (boolean): Whether or not to use the sigpy package for processing.
        noncart (boolean): ...

    Attributes:
        maps (...): The sensitivity profiles for each MRI coil.
        mask (...): The mask to incoherently subsample k-space.
        img_shape (tuple): Dimensions of image.
        use_sigpy (boolean): Whether or not to use the sigpy package for processing.
        noncart (boolean): ...
    """

    def __init__(self, maps, mask, l2lam=False, img_shape=None, use_sigpy=False, noncart=False, num_spatial_dims=2):
        super(MultiChannelMRI, self).__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam
        self.img_shape = img_shape
        self.noncart = noncart
        self._normal = None
        self.num_spatial_dims = num_spatial_dims

        if self.noncart:
            assert use_sigpy, 'Must use SigPy for NUFFT!'

        if use_sigpy: # FIXME: Not yet Implemented for 3D
            from sigpy import from_pytorch, to_device, Device
            sp_device = Device(self.maps.device.index)
            self.maps = to_device(from_pytorch(self.maps, iscomplex=True), device=sp_device)
            self.mask = to_device(from_pytorch(self.mask, iscomplex=False), device=sp_device)
            self.img_shape = self.img_shape[:-1] # convert R^2N to C^N
            self._build_model_sigpy()

        #if normal is None:
            #self.normal_fun = self._normal
        #else:
            #self.normal_fun = normal

    def _build_model_sigpy(self):
        from sigpy.linop import Multiply
        if self.noncart:
            from sigpy.linop import NUFFT, NUFFTAdjoint
        else:
            from sigpy.linop import FFT
        from sigpy import to_pytorch_function

        if self.noncart:
            Aop_list = []
            Aop_adjoint_list = []
            Aop_normal_list = []
            _img_shape = self.img_shape[1:]
            for i in range(self.img_shape[0]):
                _maps = self.maps[i, ...]
                _mask = self.mask[i, ...]
                Sop = Multiply(_img_shape, _maps)
                Fop = NUFFT(_maps.shape, _mask)
                Aop = Fop * Sop
                Fop_H = NUFFTAdjoint(_maps.shape, _mask)

                Aop_H = Sop.H * Fop_H
                Aop_list.append(to_pytorch_function(Aop, input_iscomplex=True, output_iscomplex=True).apply)
                Aop_adjoint_list.append(to_pytorch_function(Aop_H, input_iscomplex=True, output_iscomplex=True).apply)
                Aop_normal_list.append(to_pytorch_function(Aop_H * Aop, input_iscomplex=True, output_iscomplex=True).apply)

            self.Aop_list = Aop_list
            self.Aop_adjoint_list = Aop_adjoint_list
            self.Aop_normal_list = Aop_normal_list
            self._forward = self._nufft_batch_forward
            self._adjoint = self._nufft_batch_adjoint
            self._normal = self._nufft_batch_normal

        else:
            Sop = Multiply(self.img_shape, self.maps)
            Fop = FFT(self.maps.shape, axes=(-2, -1), center=False)
            Pop = Multiply(self.maps.shape, self.mask)
            Aop = Pop * Fop * Sop

            self._forward = to_pytorch_function(Aop, input_iscomplex=True, output_iscomplex=True).apply
            self._adjoint = to_pytorch_function(Aop.H, input_iscomplex=True, output_iscomplex=True).apply

    def _nufft_batch_forward(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_list[i](x[i])), axis=0)
            return out

    def _nufft_batch_adjoint(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_adjoint_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_adjoint_list[i](x[i])), axis=0)
            return out

    def _nufft_batch_normal(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_normal_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_normal_list[i](x[i])), axis=0)
            return out

    def _forward(self, x):
        return sense_forw(x, self.maps, self.mask, ndim=self.num_spatial_dims) 

    def _adjoint(self, y):
        return sense_adj(y, self.maps, self.mask, ndim=self.num_spatial_dims)

    def forward(self, x):
        return self._forward(x)

    def adjoint(self, y):
        return self._adjoint(y)

    def normal(self, x):
        if self._normal:
            out = self._normal(x)
        else:
            out = self.adjoint(self.forward(x)) # A transpose Ax
        if self.l2lam:
            out = out + self.l2lam * x
        return out

    #def normal(self, x):
        #return self.normal_fun(x)

def maps_forw(img, maps):
    return cp.zmul(img[:,None,...], maps)

def maps_adj(cimg, maps):
    return torch.sum(cp.zmul(cp.zconj(maps), cimg), 1, keepdim=False)

def fft_forw(x, ndim=2):
    return torch.fft(x, signal_ndim=ndim, normalized=True)

def fft_adj(x, ndim=2):
    return torch.ifft(x, signal_ndim=ndim, normalized=True)

def mask_forw(y, mask):
    return y * mask[:,None,...,None]
    
def sense_forw(img, maps, mask, ndim=2): 
    return mask_forw(fft_forw(maps_forw(img, maps), ndim), mask)

def sense_adj(ksp, maps, mask, ndim=2): # A transpose * y
    return maps_adj(fft_adj(mask_forw(ksp, mask), ndim), maps)

def sense_normal(img, maps, mask, ndim=2):
    return maps_adj(fft_adj(mask_forw(fft_forw(maps_forw(img, maps), ndim), mask), ndim), maps) # A transpose Ax
