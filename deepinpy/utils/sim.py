#!/usr/bin/env python
import time

#import sigpy.mri.samp
import torch.utils.data
import numpy as np
import scipy.fftpack
import h5py
import pathlib

import cfl

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from bart import bart

import deepinpy.utils.complex as cp

'''
Simulates data and creates Dataset objects for pytorch
'''

def load_data_legacy(idx, data_file, gen_masks=False):
    with h5py.File(data_file, 'r') as F:
        masks = np.array(F['trnMask'][idx,...], dtype=np.float)
        imgs = np.array(F['trnOrg'][idx,...], dtype=np.complex)
        maps = np.array(F['trnCsm'][idx,...], dtype=np.complex)
        masks = fftshift(masks)

    if len(masks.shape) == 2:
        imgs, maps, masks = imgs[None,...], maps[None,...], masks[None,...]
    return imgs, maps, masks

def load_data(idx, data_file, gen_masks=False):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
        maps = np.array(F['maps'][idx,...], dtype=np.complex)
        if not gen_masks:
            try:
                masks = np.array(F['masks'][idx,...], dtype=np.float)
            except:
                masks = poisson_mask(imgs.shape, seed=idx)
    if gen_masks:
        masks = poisson_mask(imgs.shape, seed=idx)

    if len(masks.shape) == 2:
        imgs, maps, masks = imgs[None,...], maps[None,...], masks[None,...]
    return imgs, maps, masks

def load_data_ksp(idx, data_file, gen_masks=False):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'][idx,...], dtype=np.complex)
        maps = np.array(F['maps'][idx,...], dtype=np.complex)
        ksp = np.array(F['ksp'][idx,...], dtype=np.complex)
        if not gen_masks:
            try:
                masks = np.array(F['masks'][idx,...], dtype=np.float)
            except:
                masks = poisson_mask(imgs.shape, seed=idx)
    if gen_masks:
        masks = poisson_mask(imgs.shape, seed=idx)

    if len(masks.shape) == 2:
        imgs, maps, masks, ksp = imgs[None,...], maps[None,...], masks[None,...], ksp[None,...]
    return imgs, maps, masks, ksp

def poisson_mask(masks_shape, R=6, r=10, seed=None):
    if seed is None:
        seed = int(np.random.rand()*1000)
    masks = np.zeros(masks_shape)
    XX, YY = np.meshgrid(np.linspace(-1, 1, masks_shape[-1]), np.linspace(-1, 1, masks_shape[-2]))
    ZZ = ((XX**2 + YY**2) > 1).astype(np.float)
    if len(masks_shape) > 2:
        for i in range(masks_shape[0]):
            #masks[i,...] = cfl.readcfl('poisson_masks/poisson_sigpy_R12_C2/mask_brain_{:03d}'.format(seed[i] % 320)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/poisson_sigpy_R6_C10/mask_{:03d}'.format(seed[i] % 500)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/poisson_sigpy_R6_C10/mask_brain_{:03d}'.format(seed[i] % 320)).real.squeeze()
            #masks[i,...] = cfl.readcfl('masks/mask_{:03d}'.format(seed[i])).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/unif/mask_knees_full_{:03d}'.format(seed[i] % 500)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_brain_full_{:03d}'.format(seed[i] % 400 )).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/unif/mask_brain_{:03d}'.format(seed[i])).real.squeeze()
            #masks[i,...] = sigpy.mri.samp.poisson(masks_shape[1:], accel=R, calib=[r, r], dtype=np.float32, crop_corner=True, seed=seed+i)

            masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_knees_full_{:04d}'.format(seed[i] % 5000)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_knees_full_2.5x2.5_{:04d}'.format(seed[i] % 5000)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_knees_full_2.5x2.5_V1_{:04d}'.format(seed[i] % 5000)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_knees_full_2x2_V_3_{:04d}'.format(seed[i] % 5000)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_knees_full_1.6x1.6_{:04d}'.format(seed[i] % 5000)).real.squeeze()
            #masks[i,...] = cfl.readcfl('poisson_masks/vd/mask_knees_full_1.6x1.6_{:04d}'.format(seed[i] % 5000)).real.squeeze()
            masks[i,...] = np.logical_or(masks[i,...], ZZ)
    else:
        #masks = cfl.readcfl('poisson_masks/poisson_sigpy_R12_C2/mask_brain_{:03d}'.format(seed % 320)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/poisson_sigpy_R6_C10/mask_{:03d}'.format(seed % 500)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/poisson_sigpy_R6_C10/mask_brain_{:03d}'.format(seed % 320)).real.squeeze()
        #masks = cfl.readcfl('masks/mask_{:03d}'.format(seed)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/unif/mask_brain_{:03d}'.format(seed)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/unif/mask_knees_{:03d}'.format(seed % 500)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/vd/mask_brain_full_{:03d}'.format(seed % 400)).real.squeeze()
        #masks = sigpy.mri.samp.poisson(masks_shape, accel=R, calib=[r, r], dtype=np.float32, crop_corner=True, seed=seed)

        masks = cfl.readcfl('poisson_masks/vd/mask_knees_full_{:04d}'.format(seed % 5000)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/vd/mask_knees_full_2.5x2.5_{:04d}'.format(seed % 5000)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/vd/mask_knees_full_2.5x2.5_V1_{:04d}'.format(seed % 5000)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/vd/mask_knees_full_2x2_V_3_{:04d}'.format(seed % 5000)).real.squeeze()
        #masks = cfl.readcfl('poisson_masks/vd/mask_knees_full_1.6x1.6_{:04d}'.format(seed % 5000)).real.squeeze()
        masks = np.logical_or(masks,  ZZ)

    return masks

def load_data_cached(data_file):
    with h5py.File(data_file, 'r') as F:
        imgs = np.array(F['imgs'], dtype=np.complex)
        maps = np.array(F['maps'], dtype=np.complex)
        masks = np.array(F['masks'], dtype=np.float)
        out = np.array(F['out'], dtype=np.complex)
    return imgs, maps, masks, out

def save_data_cached(data_file, imgs, maps, masks, out):
    with h5py.File(data_file, 'w') as F:
        F.create_dataset('imgs', data=imgs)
        F.create_dataset('maps', data=maps)
        F.create_dataset('masks', data=masks)
        F.create_dataset('out', data=out)



class Dataset(torch.utils.data.Dataset):
    ''' MRI data set '''

    def __init__(self, data_file, stdev=.01, num_data_sets=None, adjoint=True, preload=False, id=None, clear_cache=False, cache_data=False, gen_masks=False, sure=False, scale_data=False, fully_sampled=False, data_idx=None, inverse_crime=True):
        self.data_file = data_file
        self.stdev = stdev
        self.adjoint = adjoint
        self.preload = preload
        self.id = id
        self.cache_data = cache_data
        self.gen_masks = gen_masks
        self.max_data_sets = self._len()
        self.sure = sure
        self.fully_sampled = fully_sampled
        self.data_idx = data_idx
        self.scale_data = scale_data
        self.inverse_crime = inverse_crime

        if self.data_idx is not None:
            self.num_data_sets = 1
        else:
            if num_data_sets is None:
                self.num_data_sets = self.max_data_sets
            else:
                self.num_data_sets = int(np.minimum(num_data_sets, self.max_data_sets))

        if self.cache_data and clear_cache:
            print('clearing cache')
            for p in pathlib.Path('.').glob('cache/{}_*_{}'.format(self.id, self.data_file)):
                pathlib.Path.unlink(p)


    def _len_legacy(self):
        with h5py.File(self.data_file, 'r') as F:
            max_data_sets = F['trnMask'].shape[0]
        return max_data_sets

    def _len(self):
        with h5py.File(self.data_file, 'r') as F:
            max_data_sets = F['imgs'].shape[0]
        return max_data_sets

    def __len__(self):
        return self.num_data_sets

    def __getitem__(self, idx):
        if self.data_idx is not None:
            idx = self.data_idx

        if self.cache_data and self.id:
            data_file = 'cache/{}_{}_{}'.format(self.id, idx, self.data_file)
            try:
                imgs, maps, masks, out = load_data_cached(data_file)
            except:
                imgs, maps, masks, out = self._load_data(idx)
                save_data_cached(data_file, imgs, maps, masks, out)
        else:
                imgs, maps, masks, out = self._load_data(idx)

        data = {
                'imgs': cp.c2r(imgs.squeeze()).astype(np.float32),
                'maps': cp.c2r(maps.squeeze()).astype(np.float32),
                'masks': masks.squeeze().astype(np.float32),
                'out': cp.c2r(out.squeeze()).astype(np.float32)
                }

        return idx, data

    def _load_data(self, idx):
        if self.inverse_crime:
            #imgs, maps, masks = load_data_legacy(idx, self.data_file, self.gen_masks)
            imgs, maps, masks = load_data(idx, self.data_file, self.gen_masks)
        else:
            imgs, maps, masks, ksp = load_data_ksp(idx, self.data_file, self.gen_masks)
        if self.scale_data:
            ## FIXME: batch mode
            assert not self.scale_data, 'SEE FIXME'
            sc = np.percentile(abs(imgs), 99, axis=(-1, -2))
            imgs = imgs / sc
            ksp = ksp / sc

        if self.sure or self.fully_sampled:
            masks = np.ones(masks.shape)

        if self.inverse_crime:
            out = self._sim_data(imgs, maps, masks)
        else:
            out = self._sim_data(imgs, maps, masks, ksp)

        maps = fftmod(maps) # FIXME: slow
        return imgs, maps, masks, out

    def _sim_data(self, imgs, maps, masks, ksp=None):

        if self.sure:
            noise = np.random.randn(*imgs.shape) + 1j * np.random.randn(*imgs.shape)
            out = imgs + 1 / np.sqrt(2) * self.stdev * noise

        else:
            # N, nc, nx, ny
            noise = np.random.randn(*maps.shape) + 1j * np.random.randn(*maps.shape)
            #out = masks[:,None,:,:] * (bart(1, 'fft -u 12', imgs[:,None,:,:] * maps) + 1 / np.sqrt(2) * self.stdev * noise)
            if self.inverse_crime and ksp is None:
                out = masks[:,None,:,:] * (fft2uc(imgs[:,None,:,:] * maps) + 1 / np.sqrt(2) * self.stdev * noise)
            else:
                out = masks[:,None,:,:] * (ksp + 1 / np.sqrt(2) * self.stdev * noise)

            if self.adjoint:
                #out = np.sum(np.conj(maps) * bart(1, 'fft -iu 12', out), axis=1).squeeze()
                out = np.sum(np.conj(maps) * ifft2uc(out), axis=1).squeeze()
            else:
                out = fftmod(out)
        return out


def fftmod(out):
    out2 = out.copy()
    out2[...,::2,:] *= -1
    out2[...,:,::2] *= -1
    out2 *= -1
    return out2
    #return bart(1, 'fftmod 12', out)

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

