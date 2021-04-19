# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import sys

# sys.stdout = file
import os




import traceback
from trainer import Trainer
from options import MonodepthOptions

def experiment_training():

    experiment_names = ["experiment#1", "experiment#2", "experiment#3", "experiment#4", "experiment#5", "experiment#6", "experiment#7"]
    attention_losses = [1.0, 1.03, 1.10, 1.15, 1.20, 1.30, 1.45]

    for attention_number, current_model_name in zip(attention_losses, experiment_names):

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open('log.txt', 'w')


        options = MonodepthOptions()
        opts = options.parse()

        opts.attention_weight = attention_number
        opts.model_name = current_model_name
        opts.batch_size = 8
        opts.num_workers = 8
        opts.num_epochs = 6
        opts.attention_mask_loss = True



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


