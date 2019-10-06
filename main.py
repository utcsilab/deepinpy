#!/usr/bin/env python

from test_tube import Experiment, HyperOptArgumentParser
from pytorch_lightning import Trainer

import os
import argparse

#from deepinpy.models.mcmri import mcmri
from deepinpy.recons.cgsense.cgsense import CGSense
from deepinpy.recons.modl.modl import MoDL

#torch.backends.cudnn.enabled = True

def main_train(hparams, gpu_ids=None):
    print(hparams)
    #M = CGSense(l2lam=hparams.l2lam_init, step=hparams.step)
    M = MoDL(l2lam=hparams.l2lam_init, step=hparams.step, num_unrolls=hparams.num_unrolls)
    exp = Experiment(save_dir=os.getcwd())

    #trainer = Trainer(experiment=exp, max_nb_epochs=1, train_percent_check=.1)
    #trainer = Trainer(experiment=exp, max_nb_epochs=100, gpus=[2, 3], distributed_backend='dp')
    print('gpu ids are', gpu_ids)
    trainer = Trainer(experiment=exp, max_nb_epochs=100, gpus=gpu_ids)
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

    hparams = parser.parse_args()

    #hparams.optimize_parallel_gpu(main_train, gpu_ids=['2', '3'], max_nb_trials=10)
    #hparams.optimize_parallel_cpu(main_train, nb_trials=20, nb_workers=2)
    main_train(hparams, gpu_ids=[3])
