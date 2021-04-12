# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import sys
file = open('log.txt', 'w')
# sys.stdout = file
import os
if os.path.exists('output_during_training.txt'):
    os.remove('output_during_training.txt')
open('output_during_training.txt', 'w')



import traceback
from trainer import Trainer
from options import MonodepthOptions



options = MonodepthOptions()
opts = options.parse()


if __name__ == "__main__":
    trainer = Trainer(opts)
    # try:
    trainer.train()
    # except:
    # traceback.print_exc(file=file)

