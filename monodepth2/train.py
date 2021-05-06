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

    # seeds = [0]
    experiment_names = ["experiment#17", "experiment#18", "experiment#19"]
    attention_mask_thresholds = [1.05]
    reduce_attention_weights = [1, 0.9, 0]


    for reduce_attention_weight, current_model_name in zip(reduce_attention_weights, experiment_names):

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open('log{}.txt', 'w'.format(experiment_names))


        options = MonodepthOptions()
        opts = options.parse()


        opts.attention_mask_threshold = attention_mask_thresholds[0]
        opts.model_name = current_model_name
        opts.reduce_attention_weight = reduce_attention_weight
        opts.batch_size = 8
        opts.num_workers = 8
        opts.num_epochs = 6
        opts.attention_mask_loss = True

        trainer = Trainer(opts)
        # try:
        trainer.train()
        # except:
        #     traceback.print_exc(file=file)

if __name__ == "__main__":
    experiment_training()
    # options = MonodepthOptions()
    # opts = options.parse()
    # trainer = Trainer(opts)
    # # # try:
    # trainer.train()
    # # except:


