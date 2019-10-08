#!/usr/bin/env python

import numpy as np
import torch
#import torch.nn.functional
import cfl
import sys

from deepinpy.utils import utils
import deepinpy.utils.complex as cp
import deepinpy.opt.conjgrad
from deepinpy.utils import sim
from deepinpy.opt import opt
from deepinpy.models.mcmri.mcmri import MultiChannelMRI
from deepinpy.recons.recon import Recon

import torchvision.utils

class CGSenseRecon(Recon):

    def __init__(self, args):
        super(CGSenseRecon, self).__init__(args)
        self.l2lam = torch.nn.Parameter(torch.tensor(args.l2lam_init))

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, self.l2lam)

    def forward(self, y, A):
        x_adj = A.adjoint(y)
        return deepinpy.opt.conjgrad.conjgrad(x_adj, x_adj, A.normal, l2lam=0., verbose=False, eps=self.eps, max_iter=self.cg_max_iter)

