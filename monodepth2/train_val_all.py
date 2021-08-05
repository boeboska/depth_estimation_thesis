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


    experiment_names = ["experiment#13", "experiment#14", "experiment#20"]
    weights = ["weights_0_batch_idx9999", "weights_1_batch_idx9999", "weights_2_batch_idx9999", "weights_3_batch_idx9999", "weights_4_batch_idx9999", "weights_5_batch_idx9999"]

    for current_model_name in experiment_names:

        for epoch in weights:

            if os.path.exists('output_during_training.txt'):
                os.remove('output_during_training.txt')
            open('output_during_training.txt', 'w')
            file = open(f'log {current_model_name}.txt', 'w')

            options = MonodepthOptions()
            opts = options.parse()

            print(f'validation_all/{current_model_name}/{epoch}')


            # opts.load_weights_folder = f'validation_all/{current_model_name}/models/{epoch}'
            opts.load_weights_folder = f'monodepth_models/{current_model_name}/models/{epoch}'
            opts.batch_size = 1
            opts.num_workers = 1

            # opts.attention_mask_loss = True
            # opts.self_attention = True



            dict_name = current_model_name + '_' + epoch


            trainer = Trainer(opts)
            trainer.val_all(dict_name)

            # try:
            #     _ = trainer.train()
            # except:
            #     traceback.print_exc(file=file)

if __name__ == "__main__":
    experiment_training()
    # options = MonodepthOptions()
    # opts = options.parse()
    # trainer = Trainer(opts)
    # # try:
    # trainer.val_all()
    # trainer.train()
    # except:


