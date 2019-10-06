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

class MoDL(pl.LightningModule):

    def __init__(self, l2lam, step=.0005, num_unrolls=4):
        super(MoDL, self).__init__()
        self.l2lam = torch.nn.Parameter(torch.tensor(l2lam))
        self._build_data()
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self.step = step
        self.num_unrolls = num_unrolls

        self.denoiser = ResNet5Block()

    def _build_data(self):
        self.D = sim.Dataset(data_file="/home/jtamir/projects/deepinpy_git/data/dataset_train.h5", stdev=0.001, num_data_sets=100, adjoint=False, id=0, clear_cache=False, cache_data=False, gen_masks=False, sure=False, scale_data=False, fully_sampled=False, data_idx=None, inverse_crime=False)

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, 0.)

    def forward(self, x_adj, A):

        x = x_adj
        for i in range(self.num_unrolls):
            r = self.denoiser(x)
            x, n_cg = deepinpy.opt.conjgrad.conjgrad(r, x_adj + self.l2lam * r, A.normal, verbose=False, eps=1e-5, max_iter=self.max_cg, l2lam=self.l2lam)
        return x

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


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)


class ResNet5Block(torch.nn.Module):
    def __init__(self, num_filters=32, filter_size=3, T=4, num_filters_start=2, num_filters_end=2, batch_norm=False):
        super(ResNet5Block, self).__init__()
        num_filters_start = num_filters_end = 2
        if batch_norm:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters_end,filter_size)
            )
        else:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters_end,filter_size)
            )
        self.T = T
        
    def forward(self,x,device='cpu'):
        return x + self.step(x, device=device)
    
    def step(self, x, device='cpu'):
        # reshape (batch,x,y,channel=2) -> (batch,channe=2,x,y)
        x = x.permute(0, 3, 1, 2)
        y = self.model(x)
        return y.permute(0, 2, 3, 1)


