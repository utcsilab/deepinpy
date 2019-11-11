
#!/usr/bin/env python

import numpy as np
import torch
import cfl
import sys

import pytorch_lightning as pl

from deepinpy.utils import utils
from deepinpy import opt
import deepinpy.utils.complex as cp

from torchvision.utils import make_grid

class Recon(pl.LightningModule):

    def __init__(self, args):
        super(Recon, self).__init__()

        self._init_args(args)
        self.loss_fun = torch.nn.MSELoss(reduction='sum')
        self._build_data()

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
        self.use_sigpy = args.use_sigpy
        self.Dataset = args.Dataset

    def _build_data(self):
        self.D = self.Dataset(data_file=self.data_file, stdev=self.stdev, num_data_sets=self.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, scale_data=False, fully_sampled=self.fully_sampled, data_idx=None, inverse_crime=self.inverse_crime)

    def batch(self, data):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError

    def training_step(self, batch, batch_nb):
        idx, data = batch
        idx = utils.itemize(idx)
        imgs = data['imgs']
        inp = data['out']

        self.batch(data)

        x_hat = self.forward(inp)

        try:
            num_cg = self.get_metadata()['num_cg']
        except KeyError:
            num_cg = 0

        _b = inp.shape[0]
        if _b == 1 and idx == 0:
                _idx = 0
        elif _b > 1 and 0 in idx:
            _idx = idx.index(0)
        else:
            _idx = None
        if _idx is not None:
            with torch.no_grad():
                if self.x_adj is None:
                    x_adj = self.A.adjoint(inp)
                else:
                    x_adj = self.x_adj
                cfl.writecfl('x_hat', utils.t2n(x_hat[_idx,...]))
                cfl.writecfl('x_gt', utils.t2n(imgs[_idx,...]))
                cfl.writecfl('masks', utils.t2n2(data['masks'][_idx,...]))
                cfl.writecfl('maps', utils.t2n(data['maps'][_idx,...]))
                cfl.writecfl('ksp', utils.t2n(inp[_idx,...]))
                myim = cp.zabs(torch.cat((x_adj[_idx,...], x_hat[_idx,...], imgs[_idx,...]), dim=1))[None,None,...,0]
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1)
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
