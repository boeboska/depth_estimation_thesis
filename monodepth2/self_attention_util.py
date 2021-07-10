import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import seaborn as sns
import copy
import itertools
import matplotlib
from random import randrange



def save_self_attention_masks(inputs, outputs, batch_idx, epoch_nr, model_name, log_dir):

    path = log_dir + model_name + "/" + "self_attention_maps/"
    if not os.path.exists(path):
        os.makedirs(path)

    path = f'{path}/epoch_nr_batch_idx {epoch_nr}_{batch_idx}'
    if not os.path.exists(path):
        os.makedirs(path)

    # batch, 512, 6, 20
    self_attention_maps = outputs['self_attention_maps'][0]

    # self_attention_maps_test = outputs['self_attention_maps_test'][0]

    original_img = inputs["color_aug", 0, 0]

    original_img = np.array(original_img[0].squeeze().cpu().detach().permute(1, 2, 0).numpy())

    for i in range(10):
        # select one attention map randomly and save it
        rand_nr = randrange(self_attention_maps.shape[0])

        curr  = self_attention_maps[rand_nr]

        fig, axis = plt.subplots(3, 1, figsize=(10 , 15))


        # axis[0, 0].title.set_text('Kitti image')
        axis[0].set_title('Kitti image', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        axis[0].axis('off')
        axis[0].imshow(original_img)

        axis[1].set_title('Attention map', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        sns.heatmap(self_attention_maps[rand_nr].cpu().detach().numpy(), ax=axis[1], vmin=0, vmax=1, cmap = "YlOrBr")
        axis[1].axis('off')

        axis[2].set_title('Attention map', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        sns.heatmap(self_attention_maps[rand_nr].cpu().detach().numpy(), ax=axis[2], vmin=0, vmax=3, cmap="YlOrBr")
        axis[2].axis('off')

        fig.savefig(f'{path}/rand_nr{rand_nr}.png')
        plt.close(fig)


    return None