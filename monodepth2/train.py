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


    experiment_names = ["experiment#41", "experiment#42", "experiment#43", "experiment#44", "experiment#45", "experiment#46"]
    seeds = [5, 6, 7, 8, 9, 10]

    for current_model_name, seed in zip(experiment_names, seeds):

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open(f'log {current_model_name}.txt', 'w')


        options = MonodepthOptions()
        opts = options.parse()

        if current_model_name == "experiment#41" or current_model_name == "experiment#42" or current_model_name == "experiment#43":
            opts.attention_mask_loss = True
        else:
            opts.attention_mask_loss = False


        opts.model_name = current_model_name
        opts.self_attention = True

        opts.seed = seed

        opts.batch_size = 6
        opts.num_workers = 6
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
    # # except:
