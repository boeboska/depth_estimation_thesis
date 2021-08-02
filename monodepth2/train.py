# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import sys

# sys.stdout = file
import os

import traceback
from trainer_experiment import Trainer
from options import MonodepthOptions

def experiment_training():


    experiment_names = ["experiment#47"]

    # for current_model_name in zip(experiment_names):

    if os.path.exists('output_during_training.txt'):
        os.remove('output_during_training.txt')
    open('output_during_training.txt', 'w')
    file = open(f'log {experiment_names[0]}.txt', 'w')


    options = MonodepthOptions()
    opts = options.parse()

    opts.model_name = experiment_names[0]
    opts.edge_loss = True

    opts.batch_size = 3
    opts.num_workers = 3
    opts.num_epochs = 6


    trainer = Trainer(opts)
    try:
        _ = trainer.train()
    except:
        traceback.print_exc(file=file)

if __name__ == "__main__":
    experiment_training()
    # options = MonodepthOptions()
    # opts = options.parse()
    # trainer = Trainer(opts)
    # # try:
    # # trainer.val_all()
    # trainer.train()
    # except:
