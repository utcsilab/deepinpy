#!/usr/bin/env python

import numpy as np
import torch
#import torch.nn.functional
import cfl
import sys

from deepinpy.utils import utils
import deepinpy.utils.complex as cp
import deepinpy.opt.conjgrad
from deepinpy.opt import opt
from deepinpy.utils import sim
from deepinpy.models.mcmri.mcmri import MultiChannelMRI
from deepinpy.models.resnet.resnet import ResNet5Block, ResNet
from deepinpy.recons.recon import Recon

import torchvision.utils

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
            x, n_cg = deepinpy.opt.conjgrad.conjgrad(r, x_adj + self.l2lam * r, A.normal, verbose=False, eps=1e-5, max_iter=self.cg_max_iter, l2lam=self.l2lam)
            num_cg[i] = n_cg
        return x, num_cg
