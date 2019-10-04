#!/usr/bin/env python

import os

import pytorch_lightning as pl
import test_tube

#from deepinpy.models.mcmri import mcmri
from deepinpy.recons.cgsense.cgsense import CGSense

M = CGSense(l2lam=0.001)
exp = test_tube.Experiment(save_dir=os.getcwd())


#trainer = pl.Trainer(experiment=exp, max_nb_epochs=1, train_percent_check=.1)
#trainer = pl.Trainer(experiment=exp, max_nb_epochs=100, gpus=[2, 3], distributed_backend='dp')
trainer = pl.Trainer(experiment=exp, max_nb_epochs=100, gpus=[3])
trainer.fit(M)
