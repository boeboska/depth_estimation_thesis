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


    experiment_names = ["experiment#33", "experiment#36", "experiment#37", "experiment#34", "experiment#38", "experiment#39"]
    weights = ["weights_0_batch_idx9999", "weights_1_batch_idx9999", "weights_2_batch_idx9999", "weights_3_batch_idx9999", "weights_4_batch_idx9999", "weights_5_batch_idx9999"]

    for current_model_name in experiment_names:

        for epoch in weights:

            if os.path.exists('output_during_training.txt'):
                os.remove('output_during_training.txt')
            open('output_during_training.txt', 'w')
            file = open(f'log {current_model_name}.txt', 'w')


            options = MonodepthOptions()
            opts = options.parse()

            print(f'test_all/{current_model_name}/{epoch}')


            # breakpoint()

            opts.load_weights_folder = f'../../experiment_weights/all/monodepth_models/{current_model_name}/models/{epoch}'
            opts.batch_size = 1
            opts.num_workers = 1

            opts.attention_mask_loss = True
            opts.self_attention = True

            # opts.self_attention == True
            # opts.attention_mask_loss = True

            dict_name = current_model_name + '_' + epoch


            trainer = Trainer(opts)
            trainer.test_all(dict_name)

if __name__ == "__main__":
    experiment_training()
    # options = MonodepthOptions()
    # opts = options.parse()
    # trainer = Trainer(opts)
    # # try:
    # trainer.val_all()
    # trainer.train()
    # except:


