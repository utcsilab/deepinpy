
#!/usr/bin/env python

import numpy as np
import torch
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
        self.noncart = args.noncart
        self.Dataset = args.Dataset
        self.hparams = args

    def _build_data(self):
        self.D = self.Dataset(data_file=self.data_file, stdev=self.stdev, num_data_sets=self.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, scale_data=False, fully_sampled=self.fully_sampled, data_idx=None, inverse_crime=self.inverse_crime, noncart=self.noncart)

    def batch(self, data):
        raise NotImplementedError

    def forward(self, y):
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
                _x_hat = utils.t2n(x_hat[_idx,...])
                _x_gt = utils.t2n(imgs[_idx,...])
                _x_adj = utils.t2n(x_adj[_idx,...])

                myim = torch.tensor(np.stack((np.abs(_x_hat), np.angle(_x_hat)), axis=0))[:, None, ...]
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                self.logger.experiment.add_image('train_prediction', grid, self.current_epoch)

                if self.current_epoch == 0:
                    myim = torch.tensor(np.stack((np.abs(_x_gt), np.angle(_x_gt)), axis=0))[:, None, ...]
                    grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                    self.logger.experiment.add_image('ground_truth', grid, 0)

                    myim = torch.tensor(np.stack((np.abs(_x_adj), np.angle(_x_adj)), axis=0))[:, None, ...]
                    grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                    self.logger.experiment.add_image('input', grid, 0)


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
                'val_loss': 0.,
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
