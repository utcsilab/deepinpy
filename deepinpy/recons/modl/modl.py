#!/usr/bin/env python

import numpy as np
import torch

from deepinpy.utils import utils
from deepinpy.opt import conjgrad
from deepinpy.models import ResNet5Block, ResNet
from deepinpy.recons import Recon

class MoDLRecon(Recon):

    def __init__(self, args):
        super(MoDLRecon, self).__init__(args)
        self.l2lam = torch.nn.Parameter(torch.tensor(args.l2lam_init))

        if args.network == 'ResNet5Block':
            self.denoiser = ResNet5Block(num_filters=args.latent_channels, filter_size=7, batch_norm=args.batch_norm)
        elif args.network == 'ResNet':
            self.denoiser = ResNet(latent_channels=args.latent_channels, num_blocks=args.num_blocks, kernel_size=7, batch_norm=args.batch_norm)

    def forward(self, y, A):
        num_cg = np.zeros((self.num_unrolls,))
        x_adj = A.adjoint(y)
        x = x_adj
        for i in range(self.num_unrolls):
            r = self.denoiser(x)
            x, n_cg = conjgrad(r, x_adj + self.l2lam * r, A.normal, verbose=False, eps=1e-5, max_iter=self.cg_max_iter, l2lam=self.l2lam)
            num_cg[i] = n_cg
        return x, num_cg
