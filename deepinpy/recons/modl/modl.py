#!/usr/bin/env python

import numpy as np
import torch

from deepinpy.utils import utils
from deepinpy.opt import ConjGrad
from deepinpy.models import ResNet5Block, ResNet, UnrollNet
from deepinpy.forwards import MultiChannelMRI
from deepinpy.recons import Recon

class MoDLRecon(Recon):

    def __init__(self, hparams):
        super(MoDLRecon, self).__init__(hparams)
        self.l2lam = torch.nn.Parameter(torch.tensor(hparams.l2lam_init))

        copy_shape = np.array(self.D.shape)
        if hparams.num_spatial_dimensions == 2:
            num_channels = 2*np.prod(copy_shape[1:-2])
        elif hparams.num_spatial_dimensions == 3:
            num_channels = 2*np.prod(copy_shape[1:-3])
        else:
            raise ValueError('only 2D or 3D number of spatial dimensions are supported!')
        self.in_channels = num_channels

        if hparams.network == 'ResNet5Block':
            self.denoiser = ResNet5Block(num_filters_start=self.in_channels, num_filters_end=self.in_channels, num_filters=hparams.latent_channels, filter_size=7, batch_norm=hparams.batch_norm)
        elif hparams.network == 'ResNet':
            self.denoiser = ResNet(in_channels=self.in_channels, latent_channels=hparams.latent_channels, num_blocks=hparams.num_blocks, kernel_size=7, batch_norm=hparams.batch_norm)

        modl_recon_one_unroll = MoDLReconOneUnroll(denoiser=self.denoiser, l2lam=self.l2lam, hparams=hparams)
        self.unroll_model = UnrollNet(module_list=[modl_recon_one_unroll], data_list=[None],  num_unrolls=self.hparams.num_unrolls)

    def batch(self, data):
        self.unroll_model.batch(data)
        self.x_adj = self.unroll_model.module_list[0].x_adj
        self.A = self.unroll_model.module_list[0].A

    def forward(self, y):
        return self.unroll_model(self.A.adjoint(y))

    def get_metadata(self):
        return {
                'num_cg':  np.array([m['num_cg'] for m in self.unroll_model.get_metadata()]),
                }

class MoDLReconOneUnroll(torch.nn.Module):

    def __init__(self, denoiser, l2lam, hparams):
        super(MoDLReconOneUnroll, self).__init__()
        self.l2lam = l2lam
        self.num_cg = None
        self.x_adj = None
        self.hparams = hparams
        self.denoiser = denoiser

    def batch(self, data):

        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        self.A = MultiChannelMRI(maps, masks, l2lam=0., img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)
        self.x_adj = self.A.adjoint(inp)

    def forward(self, x):

        assert self.x_adj is not None, "x_adj not computed!"
        r = self.denoiser(x)

        cg_op = ConjGrad(self.x_adj + self.l2lam * r, self.A.normal, l2lam=self.l2lam, max_iter=self.hparams.cg_max_iter, eps=self.hparams.cg_eps, verbose=False)
        x = cg_op.forward(x)
        self.num_cg = cg_op.num_cg

        return x

    def get_metadata(self):
        return {
                'num_cg': self.num_cg,
                }
