#!/usr/bin/env python


from test_tube import Experiment, HyperOptArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TestTubeLogger

import os
import argparse

from deepinpy.recons import CGSenseRecon, MoDLRecon, ResNetRecon, DeepBasisPursuitRecon
from deepinpy.forwards import MultiChannelMRIDataset

import torch
torch.backends.cudnn.enabled = True

import numpy.random

def main_train(args, idx=None, gpu_ids=None):
    args.Dataset = MultiChannelMRIDataset
    if args.random_name:
        ridx = int(numpy.random.rand()*1000)
        name = '{}_{}'.format(args.name, ridx)
    else:
        name = args.name

    if idx is not None:
        name = '{}_{}'.format(name, idx)

    tt_logger = TestTubeLogger(save_dir="./logs", name=name, debug=False, create_git_tag=False, version=args.version)
    tt_logger.log_hyperparams(args)

    if args.recon == 'cgsense':
        M = CGSenseRecon(args)
    elif args.recon == 'modl':
        M = MoDLRecon(args)
    elif args.recon == 'resnet':
        M = ResNetRecon(args)
    elif args.recon == 'dbp':
        M = DeepBasisPursuitRecon(args)

    if args.cpu:
        trainer = Trainer(max_epochs=args.num_epochs, logger=tt_logger, default_save_path='./logs', early_stop_callback=None, accumulate_grad_batches=args.num_accumulate)
    else:
        if args.hyperopt:
            trainer = Trainer(max_epochs=args.num_epochs, gpus=1, logger=tt_logger, default_save_path='./logs', early_stop_callback=None, distributed_backend=None, accumulate_grad_batches=args.num_accumulate)
        else:
            trainer = Trainer(max_epochs=args.num_epochs, gpus=gpu_ids, logger=tt_logger, default_save_path='./logs', early_stop_callback=None, distributed_backend='ddp', accumulate_grad_batches=args.num_accumulate)

    trainer.fit(M)


if __name__ == '__main__':
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'deep inverse problems optimization'

    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='random_search')

    parser.opt_range('--step', type=float, dest='step', default=.001, help='step size/learning rate', tunable=True, nb_samples=100, low=.0001, high=.001)
    parser.opt_range('--l2lam_init', action='store', type=float, dest='l2lam_init', default=.001, tunable=False, low=.0001, high=100, help='initial l2 regularization')
    parser.opt_list('--solver', action='store', dest='solver', type=str, tunable=False, options=['sgd', 'adam'], help='optimizer/solver ("adam", "sgd")', default="sgd")
    parser.opt_range('--cg_max_iter', action='store', dest='cg_max_iter', type=int, tunable=False, low=1, high=20, help='max number of conjgrad iterations', default=10)
    parser.opt_range('--batch_size', action='store', dest='batch_size', type=int, tunable=False, low=1, high=20, help='batch size', default=2)
    parser.opt_range('--num_unrolls', action='store', dest='num_unrolls', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of unrolls', default=4)
    parser.opt_range('--num_admm', action='store', dest='num_admm', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of ADMM iterations', default=3)
    parser.opt_list('--network', action='store', dest='network', type=str, tunable=False, options=['ResNet', 'ResNet5Block'], help='which denoiser network to use', default='ResNet')
    parser.opt_list('--latent_channels', action='store', dest='latent_channels', type=int, tunable=False, options=[16, 32, 64, 128], help='number of latent channels', default=64)
    parser.opt_range('--num_blocks', action='store', dest='num_blocks', type=int, tunable=False, low=1, high=4, nb_samples=3, help='number of ResNetBlocks', default=3)
    parser.opt_range('--dropout', action='store', dest='dropout', type=float, tunable=False, low=0., high=.5, help='dropout fraction', default=0.)
    parser.opt_list('--batch_norm', action='store_true', dest='batch_norm', tunable=False, options=[True, False], help='batch normalization', default=False)

    parser.add_argument('--num_accumulate', action='store', dest='num_accumulate', type=int, help='nunumber of batch accumulations', default=1)
    parser.add_argument('--name', action='store', dest='name', type=str, help='experiment name', default=1)
    parser.add_argument('--version', action='store', dest='version', type=int, help='version number', default=1)
    parser.add_argument('--gpu', action='store', dest='gpu', type=str, help='gpu number(s)', default=None)
    parser.add_argument('--cpu', action='store_true', dest='cpu', help='Use CPU', default=False)
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs', default=20)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument('--recon', action='store', type=str, dest='recon', default='cgsense', help='reconstruction method')
    parser.add_argument('--data_file', action='store', dest='data_file', type=str, help='data.h5', default=None)
    parser.add_argument('--data_val_file', action='store', dest='data_val_file', type=str, help='data.h5', default=None)
    parser.add_argument('--num_data_sets', action='store', dest='num_data_sets', type=int, help='number of data sets to use', default=None)
    parser.add_argument('--num_workers', action='store', type=int,  dest='num_workers', help='number of workers', default=0)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    parser.add_argument('--clip_grads', action='store', type=float, dest='clip_grads', help='clip gradients to [-val, +val]', default=False)
    parser.add_argument('--cg_eps', action='store', type=float, dest='cg_eps', help='conjgrad eps', default=1e-4)
    parser.add_argument('--stdev', action='store', type=float, dest='stdev', help='complex valued noise standard deviation', default=.01)
    parser.add_argument('--max_norm_constraint', action='store', type=float, dest='max_norm_constraint', help='norm constraint on weights', default=None)
    parser.add_argument('--fully_sampled', action='store_true', dest='fully_sampled', help='fully_sampled', default=False)
    parser.add_argument('--adam_eps', action='store', type=float, dest='adam_eps', help='adam epsilon', default=1e-8)
    parser.add_argument('--inverse_crime', action='store_true', dest='inverse_crime', help='inverse crime', default=False)
    parser.add_argument('--use_sigpy', action='store_true', dest='use_sigpy', help='use SigPy for Linops', default=False)
    parser.add_argument('--noncart', action='store_true', dest='noncart', help='NonCartesian data', default=False)
    parser.add_argument('--abs_loss', action='store_true', dest='abs_loss', help='use magnitude for loss', default=False)
    parser.add_argument('--self_supervised', action='store_true', dest='self_supervised', help='self-supervised loss', default=False)
    parser.add_argument('--random_name', action='store_true', dest='random_name', help='add random index to name', default=False)
    parser.add_argument('--hyperopt', action='store_true', dest='hyperopt', help='perform hyperparam optimization', default=False)
    parser.json_config('--config', default=None)
    

    args = parser.parse_args()


    torch.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    if args.hyperopt:
        if args.gpu is None:
            args.optimize_parallel_cpu(main_train, nb_trials=args.num_trials, nb_workers=args.num_workers)
        else:
            gpu_ids = [a.strip() for a in args.gpu.split(',')]
            args.optimize_parallel_gpu(main_train, gpu_ids=gpu_ids, max_nb_trials=args.num_trials)
    else:
        if args.gpu is not None:
            gpu_ids = [int(a) for a in args.gpu.split(',')]
        else:
            gpu_ids = None
        main_train(args, gpu_ids=gpu_ids)
