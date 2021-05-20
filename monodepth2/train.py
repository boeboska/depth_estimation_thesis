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


    experiment_names = ["experiment#23", "experiment#24", "experiment#25"]
    edge_weights = [3e-5, 2e-4, 2e-3]


    for current_model_name, edge_weight in zip(experiment_names, edge_weights):

        if os.path.exists('output_during_training.txt'):
            os.remove('output_during_training.txt')
        open('output_during_training.txt', 'w')
        file = open(f'log {current_model_name}.txt', 'w')


        options = MonodepthOptions()
        opts = options.parse()

        opts.edge_loss = True
        opts.edge_weight = edge_weight
        opts.model_name = current_model_name
        opts.batch_size = 4
        opts.num_workers = 4
        opts.num_epochs = 10



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
