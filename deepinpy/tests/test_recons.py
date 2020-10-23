#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
import torch.cuda
from test_tube import HyperOptArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

import argparse

from deepinpy.recons import CGSenseRecon, MoDLRecon, ResNetRecon, DeepBasisPursuitRecon
from deepinpy.forwards import MultiChannelMRIDataset

# FIXME: configs
parser = HyperOptArgumentParser(strategy='random_search')
args = parser.parse_args()
args.cg_max_iter = 2
args.eps = 1e-6
args.fully_sampled = True
args.stdev = 1e-3
args.data_file = 'deepinpy/tests/dataset_unittest.h5'
args.num_data_sets=2
args.inverse_crime = False
args.solver = 'adam'
args.step = 1e-4
args.batch_size = 1
args.shuffle = True
args.num_workers = 1
args.num_unrolls = 2
args.num_admm = 2
args.cg_eps = 1e-5
args.l2lam_init = .01
args.network = 'ResNet'
args.latent_channels = 4
args.num_blocks = 2
args.batch_norm = False
args.use_sigpy = False
args.noncart = False
args.abs_loss = False
args.self_supervised = False
args.self_supervised_adjoint = False
args.hyperopt = False
args.config = None
args.num_epochs = 2
args.lr_scheduler = -1
args.checkpoint_init = None
args.num_spatial_dimensions = 2
args.num_accumulate = 1
args.clip_grad = 0

# try to make a checkpoint logger
checkpoint_callback = ModelCheckpoint('/dev/null', 'epoch', save_top_k=-1, mode='max', verbose=False)

class TestRecon(unittest.TestCase):

    def test_recon(self):
        args.Dataset = MultiChannelMRIDataset
        for recon in [CGSenseRecon, MoDLRecon, ResNetRecon, DeepBasisPursuitRecon]:
            print('Testing Recon {}'.format(recon))

            print('  CPU:')
            M = recon(args)
            trainer = Trainer(max_epochs=args.num_epochs, gpus=None, logger=False, accumulate_grad_batches=args.num_accumulate, progress_bar_refresh_rate=1, gradient_clip_val=args.clip_grad)
            trainer.fit(M)

            if torch.cuda.device_count() > 0:
                print('  GPU:')
                M = recon(args)
                trainer = Trainer(max_epochs=args.num_epochs, gpus=[0], logger=False)
                trainer.fit(M)


if __name__ == '__main__':
    unittest.main()


