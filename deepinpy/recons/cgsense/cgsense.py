#!/usr/bin/env python

import torch

from deepinpy.utils import utils
from deepinpy.opt import ConjGrad
from deepinpy.forwards import MultiChannelMRI
from deepinpy.recons import Recon

class CGSenseRecon(Recon):

    def __init__(self, args):
        super(CGSenseRecon, self).__init__(args)
        self.l2lam = torch.nn.Parameter(torch.tensor(args.l2lam_init))

    def forward(self, y, A):
        x_adj = A.adjoint(y)
        cg_op = ConjGrad(x_adj, A.normal, l2lam=self.l2lam, max_iter=self.cg_max_iter, eps=self.eps, verbose=False)
        x_out = cg_op.forward(x_adj)
        self.num_cg = cg_op.num_cg
        return x_out

    def get_metadata(self):
        return {
                'num_cg': self.num_cg,
                }
