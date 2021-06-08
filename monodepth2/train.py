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


    experiment_names = ["experiment#26", "experiment#27", "experiment#28"]
    seeds = [5, 5, 6]
    reduce_attention_weights = [0.9, 0.8, 0.8]


    for current_model_name, seed, reduce_attention_weight in zip(experiment_names, seeds, reduce_attention_weights):

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open(f'log {current_model_name}.txt', 'w')

        print("settings:", current_model_name, seed, reduce_attention_weight)


        options = MonodepthOptions()
        opts = options.parse()

        opts.attention_mask_loss = True
        opts.model_name = current_model_name
        opts.seed = seed
        opts.reduce_attention_weight = reduce_attention_weight
        opts.batch_size = 8
        opts.num_workers = 8
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
    # try:
    # trainer.val_all()
    # trainer.train()
    # except:
