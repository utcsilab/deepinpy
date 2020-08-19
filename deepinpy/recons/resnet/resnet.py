#!/usr/bin/env python

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import ResNet5Block, ResNet
from deepinpy.recons import Recon
import numpy as np

class ResNetRecon(Recon):

    def __init__(self, hparams):
        super(ResNetRecon, self).__init__(hparams)

        copy_shape = np.array(self.D.shape)
        if hparams.num_spatial_dimensions == 2:
            num_channels = 2*np.prod(copy_shape[1:-2])
        elif hparams.num_spatial_dimensions == 3:
            num_channels = 2*np.prod(copy_shape[1:-3])
        else:
            raise ValueError('only 2D or 3D number of spatial dimensions are supported!')
        self.in_channels = num_channels
        
        if self.hparams.network == 'ResNet5Block': # FIX ALSO
            self.network = ResNet5Block(num_filters_start=self.in_channels, num_filters_end=self.in_channels, num_filters=self.hparams.latent_channels, filter_size=7, batch_norm=self.hparams.batch_norm)
        elif self.hparams.network == 'ResNet':
            self.network = ResNet(in_channels=self.in_channels, latent_channels=self.hparams.latent_channels, num_blocks=self.hparams.num_blocks, kernel_size=7, batch_norm=self.hparams.batch_norm)

    def batch(self, data):

        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)
        self.x_adj = self.A.adjoint(inp)

    def forward(self, y):
        return self.network(self.x_adj)

    def get_metadata(self):
        return {}
