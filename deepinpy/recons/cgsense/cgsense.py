#!/usr/bin/env python

import torch

from deepinpy.utils import utils
from deepinpy.opt import conjgrad
from deepinpy.forwards import MultiChannelMRI
from deepinpy.recons import Recon

class CGSenseRecon(Recon):

    def __init__(self, args):
        super(CGSenseRecon, self).__init__(args)
        self.l2lam = torch.nn.Parameter(torch.tensor(args.l2lam_init))

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, self.l2lam)

    def forward(self, y, A):
        x_adj = A.adjoint(y)
        return conjgrad(x_adj, x_adj, A.normal, l2lam=0., verbose=False, eps=self.eps, max_iter=self.cg_max_iter)

