#!/usr/bin/env python

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import ResNet5Block, ResNet
from deepinpy.recons import Recon

class ResNetRecon(Recon):

    def __init__(self, args):
        super(ResNetRecon, self).__init__(args)

        if args.network == 'ResNet5Block':
            self.denoiser = ResNet5Block(num_filters=args.latent_channels, filter_size=7, batch_norm=args.batch_norm)
        elif args.network == 'ResNet':
            self.denoiser = ResNet(latent_channels=args.latent_channels, num_blocks=args.num_blocks, kernel_size=7, batch_norm=args.batch_norm)

    def forward(self, y, A):
        x_adj = A.adjoint(y)
        return self.denoiser(x_adj), 0
