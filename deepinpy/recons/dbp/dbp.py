#!/usr/bin/env python

import numpy as np
import torch
import tqdm

from deepinpy.utils import utils
from  deepinpy.opt import ConjGrad
from deepinpy import opt
from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import ResNet5Block, ResNet
from deepinpy.recons import Recon

class DeepBasisPursuitRecon(Recon):

    def __init__(self, hparams):
        super(DeepBasisPursuitRecon, self).__init__(hparams)
        self.l2lam = torch.nn.Parameter(torch.tensor(hparams.l2lam_init))
        self.num_admm = hparams.num_admm

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

        self.debug_level = 0

        self.mean_residual_norm = 0

    def batch(self, data):

        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        with torch.no_grad():
            self.A = MultiChannelMRI(maps, masks, l2lam=0., img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)
            self.x_adj = self.A.adjoint(inp)

            self.eps = (self.A.maps.shape[1] * torch.sum(self.A.mask.reshape((self.A.mask.shape[0], -1)), dim=1)).sqrt() * self.hparams.stdev

    def forward(self, y):
        with torch.no_grad():
            self.mean_eps = torch.mean(self.eps)
            #print('epsilon is {}'.format(self.eps))
        x = self.A.adjoint(y)
        z = self.A(x)
        z_old = z
        u = z.new_zeros(z.shape)

        x.requires_grad = False
        z.requires_grad = False
        z_old.requires_grad = False
        u.requires_grad = False

        self.num_cg = np.zeros((self.hparams.num_unrolls,self.hparams.num_admm,))

        for i in range(self.hparams.num_unrolls):
            r = self.denoiser(x)

            for j in range(self.hparams.num_admm):

                rhs = self.l2lam * self.A.adjoint(z - u) + r
                fun = lambda xx: self.l2lam * self.A.normal(xx) + xx
                cg_op = ConjGrad(rhs, fun, max_iter=self.hparams.cg_max_iter, eps=self.hparams.cg_eps, verbose=False)
                x = cg_op.forward(x)
                n_cg = cg_op.num_cg
                self.num_cg[i, j] = n_cg

                Ax_plus_u = self.A(x) + u
                z_old = z
                z = y + opt.l2ball_proj_batch(Ax_plus_u - y, self.eps)
                u = Ax_plus_u - z

                # check ADMM convergence
                with torch.no_grad():
                    Ax = self.A(x)
                    tmp = Ax - z
                    tmp = tmp.contiguous()
                    r_norm = torch.real(opt.zdot_single_batch(tmp)).sqrt()

                    tmp = self.l2lam * self.A.adjoint(z - z_old)
                    tmp = tmp.contiguous()
                    s_norm = torch.real(opt.zdot_single_batch(tmp)).sqrt()

                    if (r_norm + s_norm).max() < 1E-2:
                        if self.debug_level > 0:
                            tqdm.tqdm.write('stopping early, a={}'.format(a))
                        break
                    tmp = y - Ax
                    self.mean_residual_norm = torch.mean(torch.sqrt(torch.real(opt.zdot_single_batch(tmp))))
        return x

    def get_metadata(self):
        return {
                'num_cg': self.num_cg.ravel(),
                'mean_residual_norm': self.mean_residual_norm,
                'mean_eps': self.mean_eps,
                }
