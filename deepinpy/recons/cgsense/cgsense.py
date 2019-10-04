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
from deepinpy.models.mcmri.mcmri import MultiChannelMRI

import pytorch_lightning as pl

class CGSense(pl.LightningModule):

    def __init__(self, l2lam, step=.0005):
        super(CGSense, self).__init__()
        self.l2lam = torch.nn.Parameter(torch.tensor(l2lam))
        self._build_data()
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self.step = step

    def _build_data(self):
        self.D = sim.Dataset(data_file="/home/jtamir/projects/deepinpy_git/data/dataset_train.h5", stdev=0.001, num_data_sets=100, adjoint=False, id=0, clear_cache=False, cache_data=False, gen_masks=False, sure=False, scale_data=False, fully_sampled=False, data_idx=None, inverse_crime=False)

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, self.l2lam)

    def forward(self, x_adj, A):
        return deepinpy.opt.conjgrad.conjgrad(x_adj, x_adj, A.normal, l2lam=0., verbose=False, eps=1e-5, max_iter=100)

    def training_step(self, batch, batch_nb):
        idx, data = batch
        imgs = data['imgs']
        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        #self._build_MCMRI(maps, masks)
        #print(idx, data)

        A = self._build_MCMRI(maps, masks)

        x_adj = A.adjoint(inp)
        x_hat = self.forward(x_adj, A)
        if batch_nb == 0:
            cfl.writecfl('x_hat', utils.t2n(x_hat))
            cfl.writecfl('x_gt', utils.t2n(imgs))
            cfl.writecfl('masks', utils.t2n2(masks))
            cfl.writecfl('maps', utils.t2n(maps))
            cfl.writecfl('ksp', utils.t2n(inp))
        return {
                'loss': self.loss_fun(x_hat, imgs),
                'progress': {'lambda': self.l2lam.clone().detach().requires_grad_(False)},
                }

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.step)]

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.D, batch_size=3, shuffle=True, num_workers=16, drop_last=True)
