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
    description_str = 'modl-based optimization'

    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='random_search')

    parser.add_argument('--num_unrolls', action='store', dest='num_unrolls', type=int, help='number of unrolls', default=4)
    parser.opt_range('--step', type=float, dest='step', default=.001, help='step size/learning rate', tunable=True, low=.0001, high=.1)
    parser.add_argument('--l2lam_init', action='store', type=float, dest='l2lam_init', default=.001, help='initial l2 regularization')

    hparams = parser.parse_args()

    #hparams.optimize_parallel_gpu(main_train, gpu_ids=['2', '3'], max_nb_trials=10)
    #hparams.optimize_parallel_cpu(main_train, nb_trials=20, nb_workers=2)
    main_train(hparams, gpu_ids=[3])
