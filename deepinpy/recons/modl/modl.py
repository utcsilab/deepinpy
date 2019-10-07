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

import torchvision.utils

import pytorch_lightning as pl

class MoDLRecon(pl.LightningModule):

    def __init__(self, args):
        super(MoDLRecon, self).__init__()
        self.l2lam = torch.nn.Parameter(torch.tensor(args.l2lam_init))
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self.step = args.step
        self.num_unrolls = args.num_unrolls
        self.solver = args.solver
        self.stdev = args.stdev
        self.num_data_sets = args.num_data_sets
        self.fully_sampled = args.fully_sampled
        self.cg_max_iter = args.cg_max_iter
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle

        self._build_data()

        if args.network == 'ResNet5Block':
            self.denoiser = ResNet5Block(num_filters=args.latent_channels, filter_size=7, batch_norm=args.batch_norm)
        elif args.network == 'ResNet':
            self.denoiser = ResNet(latent_channels=args.latent_channels, num_blocks=args.num_blocks, kernel_size=7, batch_norm=args.batch_norm)

    def _build_data(self):
        self.D = sim.Dataset(data_file="/home/jtamir/projects/deepinpy_git/data/dataset_train.h5", stdev=self.stdev, num_data_sets=self.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, gen_masks=False, sure=False, scale_data=False, fully_sampled=self.fully_sampled, data_idx=None, inverse_crime=False)

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, 0.)

    def forward(self, x_adj, A):

        num_cg = np.zeros((self.num_unrolls,))
        x = x_adj
        for i in range(self.num_unrolls):
            r = self.denoiser(x)
            x, n_cg = deepinpy.opt.conjgrad.conjgrad(r, x_adj + self.l2lam * r, A.normal, verbose=False, eps=1e-5, max_iter=self.cg_max_iter, l2lam=self.l2lam)
            num_cg[i] = n_cg
        return x, num_cg

    def training_step(self, batch, batch_nb):
        idx, data = batch
        idx = utils.itemize(idx)
        imgs = data['imgs']
        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        A = self._build_MCMRI(maps, masks)

        x_adj = A.adjoint(inp)
        x_hat, num_cg = self.forward(x_adj, A)
        if 0 in idx:
            _idx = idx.index(0)
            cfl.writecfl('x_hat', utils.t2n(x_hat[_idx,...]))
            cfl.writecfl('x_gt', utils.t2n(imgs[_idx,...]))
            cfl.writecfl('masks', utils.t2n2(masks[_idx,...]))
            cfl.writecfl('maps', utils.t2n(maps[_idx,...]))
            cfl.writecfl('ksp', utils.t2n(inp[_idx,...]))
            myim = cp.zabs(torch.cat((x_hat[_idx,...], imgs[_idx,...]), dim=1))[None,None,...,0]
            grid = torchvision.utils.make_grid(myim, scale_each=True, normalize=True, nrow=1)
            self.logger.experiment.add_image('Train prediction', grid, 0)


        loss = self.loss_fun(x_hat, imgs)

        _loss = loss.clone().detach().requires_grad_(False)
        _lambda = self.l2lam.clone().detach().requires_grad_(False)
        _epoch = self.current_epoch
        _nrmse = (opt.ip_batch(x_hat - imgs) / opt.ip_batch(imgs)).sqrt().mean().detach().requires_grad_(False)
        _num_cg = np.max(num_cg)

        if self.logger:
            self.logger.log_metrics({
                'lambda': _lambda,
                'loss': _loss,
                'epoch': _epoch,
                'nrmse': _nrmse, 
                'max_num_cg': _num_cg,
                })
        return {
                'loss': loss
                }

    def configure_optimizers(self):
        if 'adam' in self.solver:
            return [torch.optim.Adam(self.parameters(), lr=self.step)]
        elif 'sgd' in self.solver:
            return [torch.optim.SGD(self.parameters(), lr=self.step)]

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.D, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
