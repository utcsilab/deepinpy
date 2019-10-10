
#!/usr/bin/env python

import numpy as np
import torch
import cfl
import sys

import pytorch_lightning as pl

from deepinpy.utils import utils
import deepinpy.utils.complex as cp
import deepinpy.opt.conjgrad
from deepinpy.utils import sim
from deepinpy.opt import opt
from deepinpy.models.mcmri.mcmri import MultiChannelMRI


import torchvision.utils

class Recon(pl.LightningModule):

    def __init__(self, args):
        super(Recon, self).__init__()

        self._init_args(args)
        self._build_data()
        self.loss_fun = torch.nn.MSELoss(reduction='sum')

    def _init_args(self, args):
        self.step = args.step
        self.stdev = args.stdev
        self.num_data_sets = args.num_data_sets
        self.num_unrolls = args.num_unrolls
        self.fully_sampled = args.fully_sampled
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.shuffle = args.shuffle
        self.cg_max_iter = args.cg_max_iter
        self.eps = args.cg_eps
        self.solver=args.solver
        self.data_file = args.data_file
        self.inverse_crime = args.inverse_crime

    def _build_data(self):
        self.D = sim.Dataset(data_file=self.data_file, stdev=self.stdev, num_data_sets=self.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, gen_masks=False, sure=False, scale_data=False, fully_sampled=self.fully_sampled, data_idx=None, inverse_crime=self.inverse_crime)

    def _build_MCMRI(self, maps, masks):
        return MultiChannelMRI(maps, masks, 0.)

    def training_step(self, batch, batch_nb):
        idx, data = batch
        idx = utils.itemize(idx)
        imgs = data['imgs']
        maps = data['maps']
        masks = data['masks']
        inp = data['out']

        A = self._build_MCMRI(maps, masks)

        x_adj = A.adjoint(inp)
        x_hat, num_cg = self.forward(inp, A)
        _b = inp.shape[0]
        if _b == 1 and idx == 0:
                _idx = 0
        elif _b > 1 and 0 in idx:
            _idx = idx.index(0)
        else:
            _idx = None
        if _idx is not None:
            cfl.writecfl('x_hat', utils.t2n(x_hat[_idx,...]))
            cfl.writecfl('x_gt', utils.t2n(imgs[_idx,...]))
            cfl.writecfl('masks', utils.t2n2(masks[_idx,...]))
            cfl.writecfl('maps', utils.t2n(maps[_idx,...]))
            cfl.writecfl('ksp', utils.t2n(inp[_idx,...]))
            myim = cp.zabs(torch.cat((x_adj[_idx,...], x_hat[_idx,...], imgs[_idx,...]), dim=1))[None,None,...,0]
            grid = torchvision.utils.make_grid(myim, scale_each=True, normalize=True, nrow=1)
            self.logger.experiment.add_image('Train prediction', grid, 0)

        loss = self.loss_fun(x_hat, imgs)

        _loss = loss.clone().detach().requires_grad_(False)
        try:
            _lambda = self.l2lam.clone().detach().requires_grad_(False)
        except:
            _lambda = 0
        _epoch = self.current_epoch
        _nrmse = (opt.ip_batch(x_hat - imgs) / opt.ip_batch(imgs)).sqrt().mean().detach().requires_grad_(False)
        _num_cg = np.max(num_cg)

        log_dict = {
                'lambda': _lambda,
                'loss': _loss,
                'epoch': self.current_epoch,
                'nrmse': _nrmse, 
                'max_num_cg': _num_cg,
                }
        return {
                'loss': loss,
                'log': log_dict,
                'progress_bar': log_dict,
                }


    def configure_optimizers(self):
        if 'adam' in self.solver:
            return [torch.optim.Adam(self.parameters(), lr=self.step)]
        elif 'sgd' in self.solver:
            return [torch.optim.SGD(self.parameters(), lr=self.step)]

    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.D, batch_size=self.batch_size, shuffle=self.shuffle, num_workers=self.num_workers, drop_last=True)
