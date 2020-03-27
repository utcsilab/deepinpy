#!/usr/bin/env python

import unittest
import numpy as np
import numpy.testing as npt
from test_tube import HyperOptArgumentParser

from pytorch_lightning import Trainer

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
args.distributed_training = False
args.num_epochs = 1
args.version = 0

class TestRecon(unittest.TestCase):

    def test_recon(self):
        args.Dataset = MultiChannelMRIDataset
        for recon in [CGSenseRecon, MoDLRecon, ResNetRecon, DeepBasisPursuitRecon]:
            print('Testing Recon {}'.format(recon))
            M = recon(args)

            print('  CPU:')
            trainer = Trainer(max_epochs=args.num_epochs, gpus=None, logger=False)
            trainer.fit(M)

            print('  GPU:')
            trainer = Trainer(max_epochs=args.num_epochs, gpus=[0], logger=False)
            trainer.fit(M)

            #FIXME: check ddp

if __name__ == '__main__':
    unittest.main()


