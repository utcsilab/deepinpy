#!/usr/bin/env python

import numpy as np
import torch

from deepinpy.utils import utils
from deepinpy.opt import ConjGrad
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
        self.num_cg = np.zeros((self.num_unrolls,))
        x_adj = A.adjoint(y)
        x = x_adj
        for i in range(self.num_unrolls):
            r = self.denoiser(x)
            cg_op = ConjGrad(x_adj + self.l2lam * r, A.normal, l2lam=self.l2lam, max_iter=self.cg_max_iter, eps=self.eps, verbose=False)
            x = cg_op.forward(x)
            n_cg = cg_op.num_cg
            self.num_cg[i] = n_cg
        return x

    def get_metadata(self):
        return {
                'num_cg': self.num_cg,
                }
