
#!/usr/bin/env python

import numpy as np
import torch
import sys

import pytorch_lightning as pl

from deepinpy.utils import utils
from deepinpy import opt
import deepinpy.utils.complex as cp
from deepinpy.forwards import MultiChannelMRIDataset

from torchvision.utils import make_grid
from torch.optim import lr_scheduler 

class Recon(pl.LightningModule):

    def __init__(self, hparams):
        super(Recon, self).__init__()

        self._init_hparams(hparams)
        self._build_data()
        self.scheduler = None

    def _init_hparams(self, hparams):
        self.hparams = hparams

        self._loss_fun = torch.nn.MSELoss(reduction='sum')

        if hparams.abs_loss:
            self.loss_fun = self._abs_loss_fun
        else:
            self.loss_fun = self._loss_fun


    def _build_data(self):
        self.D = MultiChannelMRIDataset(data_file=self.hparams.data_file, stdev=self.hparams.stdev, num_data_sets=self.hparams.num_data_sets, adjoint=False, id=0, clear_cache=False, cache_data=False, scale_data=False, fully_sampled=self.hparams.fully_sampled, data_idx=None, inverse_crime=self.hparams.inverse_crime, noncart=self.hparams.noncart)

    def _abs_loss_fun(self, x_hat, imgs):
        x_hat_abs = torch.sqrt(x_hat.pow(2).sum(dim=-1))
        imgs_abs = torch.sqrt(imgs.pow(2).sum(dim=-1))
        return self._loss_fun(x_hat_abs, imgs_abs)

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
                if self.logger:
                    self.logger.experiment.add_image('2_train_prediction', grid, self.current_epoch)

                if self.current_epoch == 0:
                    myim = torch.tensor(np.stack((np.abs(_x_gt), np.angle(_x_gt)), axis=0))[:, None, ...]
                    grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                    if self.logger:
                        self.logger.experiment.add_image('1_ground_truth', grid, 0)

                    myim = torch.tensor(np.stack((np.abs(_x_adj), np.angle(_x_adj)), axis=0))[:, None, ...]
                    grid = make_grid(myim, scale_each=True, normalize=True, nrow=8, pad_value=10)
                    if self.logger:
                        self.logger.experiment.add_image('0_input', grid, 0)


        if self.hparams.self_supervised:
            pred = self.A.forward(x_hat)
            gt = inp
        else:
            pred = x_hat
            gt = imgs

        loss = self.loss_fun(pred, gt)

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
                'train_loss': _loss,
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
        if 'adam' in self.hparams.solver:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.step)
        elif 'sgd' in self.hparams.solver:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.step)
        if(self.hparams.lr_scheduler != -1):
            # doing self.scheduler will create a scheduler instance in our self object
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.lr_scheduler[0], gamma=self.hparams.lr_scheduler[1])
        if self.scheduler is None:
            return [self.optimizer]
        else:                
            return [self.optimizer], [self.scheduler]

    @pl.data_loader
    def train_dataloader(self):
        if self.hparams.distributed_training:
            sampler = torch.utils.data.distributed.DistributedSampler(self.D, shuffle=self.hparams.shuffle)
            shuffle = False
        else:
            sampler = None
            shuffle = self.hparams.shuffle
        return torch.utils.data.DataLoader(self.D, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=0, drop_last=True, sampler=sampler)
