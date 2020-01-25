#!/usr/bin/env python

import numpy as np
import torch

import deepinpy.utils.complex as cp

class MultiChannelMRI(torch.nn.Module):
    def __init__(self, maps, mask, l2lam=False, img_shape=None, use_sigpy=False, use_kbnufft=False, noncart=False):
        super(MultiChannelMRI, self).__init__()
        self.maps = maps
        self.mask = mask
        self.l2lam = l2lam
        self.img_shape = img_shape
        self.noncart = noncart
        self._normal = None


        if use_sigpy:
            from sigpy import from_pytorch, to_device, Device
            sp_device = Device(self.maps.device.index)
            self.maps = to_device(from_pytorch(self.maps, iscomplex=True), device=sp_device)
            self.mask = to_device(from_pytorch(self.mask, iscomplex=False), device=sp_device)
            self.img_shape = self.img_shape[:-1] # convert R^2N to C^N
            self._build_model_sigpy()
        elif use_kbnufft:
            self._build_model_torchkbnufft()

        #if normal is None:
            #self.normal_fun = self._normal
        #else:
            #self.normal_fun = normal

    def _build_model_torchkbnufft(self):
        from torchkbnufft import MriSenseNufft, AdjMriSenseNufft, ToepSenseNufft
        from torchkbnufft.nufft.toep_functions import calc_toep_kernel
        assert self.noncart, 'only for NUFFT'
        Aop_list = []
        Aop_adjoint_list = []
        Aop_normal_list = []
        Aop_kern_list = []
        Aop_traj_list = []
        self._img_shape = self.img_shape[1:-1]
        self._grid_shape = [2*a for a in self._img_shape]
        self._mask_shape = self.mask[0,...].shape
        for i in range(self.img_shape[0]):
            _maps = self.maps[i, ...].unsqueeze(0).permute((0, 1, 4, 2, 3))
            _mask = self.mask[i, ...].unsqueeze(0).permute((0, 3, 1, 2)).reshape((1, 2, -1))
            Aop_traj_list.append(_mask)
            sensenufft_ob = MriSenseNufft(smap=_maps, im_size=self._img_shape, grid_size=self._grid_shape).to(_maps.dtype).to(_maps.device)
            print(_maps.dtype, _maps.device)
            adjsensenufft_ob = AdjMriSenseNufft(smap=_maps, im_size=self._img_shape, grid_size=self._grid_shape).to(_maps.dtype).to(_maps.device)
            toep_ob = ToepSenseNufft(smap=_maps)
            normal_kern = calc_toep_kernel(adjsensenufft_ob, _mask)

            Aop_list.append(sensenufft_ob)
            Aop_adjoint_list.append(adjsensenufft_ob)
            Aop_normal_list.append(toep_ob)
            Aop_kern_list.append(normal_kern)

        self.Aop_list = Aop_list
        self.Aop_adjoint_list = Aop_adjoint_list
        self.Aop_normal_list = Aop_normal_list
        self.Aop_kern_list = Aop_kern_list
        self.Aop_traj_list = Aop_traj_list

        self._forward = self._kbnufft_batch_forward
        self._adjoint = self._kbnufft_batch_adjoint
        self._normal = self._kbnufft_batch_normal


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
            self._normal = to_pytorch_function(Aop.H * Aop, input_iscomplex=True, output_iscomplex=True).apply

    def _kbnufft_batch_forward(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_list[i](x[i])), axis=0)
            return out

    def _kbnufft_batch_adjoint(self, x):
        x = x.permute((0, 1, 4, 2, 3))
        batch_size = x.shape[0]
        print(x.shape, self.Aop_traj_list[0].shape)
        out0 = self.Aop_adjoint_list[0](x[0].unsqueeze(0), self.Aop_traj_list[0])
        if batch_size == 1:
            return out0.permute((0, 2, 3, 4, 1))
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_adjoint_list[i](x[i].unsqueeze(0), self.Aop_traj_list[i]))[0], axis=0)
            return out.permute((0, 2, 3, 4, 1))

    def _kbnufft_batch_normal(self, x):
        batch_size = x.shape[0]
        out0 = self.Aop_normal_list[0](x[0])
        if batch_size == 1:
            return out0[None,...]
        else:
            out = out0
            for i in range(1, batch_size):
                out = torch.stack((out, self.Aop_normal_list[i](x[i])), axis=0)
            return out

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
        return sense_forw(x, self.maps, self.mask)

    def _adjoint(self, y):
        return sense_adj(y, self.maps, self.mask)

    def forward(self, x):
        return self._forward(x)

    def adjoint(self, y):
        return self._adjoint(y)

    def normal(self, x):
        if self._normal:
            out = self._normal(x)
        else:
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
