# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import os

import traceback
from trainer import Trainer
from options import MonodepthOptions

import sys
file = open('log.txt', 'w')


import traceback
from trainer import Trainer
from options import MonodepthOptions


def experiment_training():

    seeds = [1, 2]
    experiment_names = ["experiment#13", "experiment#14"]

    for current_model_name in experiment_names:

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open('log.txt', 'w')


        options = MonodepthOptions()
        opts = options.parse()

        if current_model_name == "experiment#13":
            opts.seed = seeds[0]
        if current_model_name == "experiment#14":
            opts.seed = seeds[1]


        opts.model_name = current_model_name
        opts.batch_size = 1
        opts.num_workers = 1
        opts.num_epochs = 6


        trainer = Trainer(opts)
        try:
            trainer.train()
        except:
            traceback.print_exc(file=file)

if __name__ == "__main__":
    experiment_training()
    # trainer = Trainer(opts)
    # # try:
    # trainer.train()
    # except:


