#!/usr/bin/env python

from test_tube import Experiment, HyperOptArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger


import os
import argparse

#from deepinpy.models.mcmri import mcmri
from deepinpy.recons.cgsense.cgsense import CGSenseRecon
from deepinpy.recons.modl.modl import MoDLRecon
from deepinpy.recons.resnet.resnet import ResNetRecon

import torch
torch.backends.cudnn.enabled = True

import numpy.random

def main_train(args, gpu_ids=None):
    #print(args)
    #exp = Experiment(save_dir=os.getcwd())
    tt_logger = TestTubeLogger(save_dir="./logs", name=args.name, debug=False, create_git_tag=False, version=args.version)
    tt_logger.log_hyperparams(args)

    if args.recon == 'cgsense':
        M = CGSenseRecon(l2lam=args.l2lam_init, step=args.step, solver=args.solver, max_cg=args.max_cg)
    elif args.recon == 'modl':
        M = MoDLRecon(l2lam=args.l2lam_init, step=args.step, num_unrolls=args.num_unrolls, solver=args.solver, max_cg=args.max_cg)
    elif args.recon == 'resnet':
        M = ResNetRecon(step=args.step, solver=args.solver)

    #trainer = Trainer(experiment=exp, max_nb_epochs=1, train_percent_check=.1)
    #trainer = Trainer(experiment=exp, max_nb_epochs=100, gpus=[2, 3], distributed_backend='dp')
    print('gpu ids are', gpu_ids)
    trainer = Trainer(max_nb_epochs=args.num_epochs, gpus=gpu_ids, logger=tt_logger, default_save_path='./logs')
    #trainer = Trainer(experiment=exp, max_nb_epochs=10)

    trainer.fit(M)

if __name__ == '__main__':
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'deep inverse problems optimization'

    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='random_search')

    parser.add_argument('--name', action='store', dest='name', type=str, help='experiment name', default=1)
    parser.add_argument('--solver', action='store', dest='solver', type=str, help='optimizer/solver ("adam", "sgd")', default="sgd")
    parser.add_argument('--version', action='store', dest='version', type=int, help='version number', default=1)
    parser.add_argument('--num_unrolls', action='store', dest='num_unrolls', type=int, help='number of unrolls', default=4)
    parser.opt_range('--step', type=float, dest='step', default=.001, help='step size/learning rate', tunable=True, low=.0001, high=.1)
    parser.add_argument('--gpu', action='store', dest='gpu', type=int, help='gpu number', default=0)
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--max_cg', action='store', dest='max_cg', type=int, help='max number of conjgrad iterations', default=10)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument('--l2lam_init', action='store', type=float, dest='l2lam_init', default=.001, help='initial l2 regularization')
    parser.add_argument('--recon', action='store', type=str, dest='recon', default='cgsense', help='reconstruction method')

    args = parser.parse_args()

    #args.optimize_parallel_gpu(main_train, gpu_ids=['2', '3'], max_nb_trials=10)
    #args.optimize_parallel_cpu(main_train, nb_trials=20, nb_workers=2)
    torch.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)
    main_train(args, gpu_ids=[args.gpu])
