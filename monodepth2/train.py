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

    seeds = [1, 2]
    experiment_names = ["experiment#10", "experiment#11", "experiment#12"]
    attention_losses = [1.3, 1.3, 50]

    for attention_number, current_model_name in zip(attention_losses, experiment_names):

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open('log.txt', 'w')




        options = MonodepthOptions()
        opts = options.parse()

        if current_model_name == "experiment#10":
            opts.seed = seeds[0]
        if current_model_name == "experiment#11":
            opts.seed = seeds[1]

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


