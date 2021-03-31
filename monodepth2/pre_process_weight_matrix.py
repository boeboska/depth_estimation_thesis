import os
# from tq import tqdm
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision import transforms
import numpy as np


import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import time
import seaborn as sns
import copy
import itertools
from tqdm import tqdm

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def pil_loader_attention(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def calculate_weight_per_mask(attention_masks, threshold):
    # set values again to their normal values. First their were casted to negative such that there was no torch.eq outside the mask
    attention_masks[attention_masks >= 0.8] = 1
    attention_masks[attention_masks < 0.8] = 0

    # batch x amount of attention. sum every attention tensor within the batch size
    attention_sum = attention_masks.sum(-1).sum(-1)
    #     print("ATTENTION SUM", attention_sum)

    # within a batch you have multiple attention masks. Decide how much weight each attention mask will receive.
    # The smaller the mask the more weight it'll receive.

    v = attention_sum / (attention_masks.shape[1] * attention_masks.shape[2])

    v = 1 / v
    # remove inf number because 1 / 0 = inf
    v[v == float('inf')] = 0
    v[v != v] = 0

    avg_weight_per_threshold = {
        0.4: 1053,
        0.5: 980,
        0.6: 905,
        0.7: 818,
        0.8: 688
    }

    attention_weight_matrix = v / avg_weight_per_threshold[threshold]

    # remove nan
    attention_weight_matrix[attention_weight_matrix != attention_weight_matrix] = 0

    return attention_weight_matrix





device = "cuda"
torch.cuda.empty_cache()

# PAD AANPASSEN
paths = os.walk('../data/attention_masks')

methods = ['avg', 'min', 'max']
thresholds = [0.5, 0.6, 0.7, 0.8]

for threshold in thresholds:
    print("threshold", threshold)
    for method in methods:
        print("method", method)
        for i, (path, directories, files) in enumerate(tqdm(paths)):

            if i % 100 == 0:
                print(i)

            # voor deze kitti plaatjes heb je al de weight mask berekend
            weight_path = path.replace("attention_masks", "weight_mask")

            # open the attention masks per kitti image
            if len(files) == 100 and files[0].endswith('jpg'):
                start = time.time()

                attention_masks_per_kitti_img = torch.empty(size=(0, 192, 640))

                for file in files:

                    # check the attention mask probability
                    attention_mask_prob = float(file.split("_")[1].split(".jpg")[0])

                    #             print("PROB", attention_mask_prob)
                    if attention_mask_prob >= threshold:
                        # create path to attention mask
                        curr_path = path + "/" + file
                        attention_mask_from_pil = pil_loader_attention(curr_path)
                        attention_mask_from_pil = transforms.ToTensor()(attention_mask_from_pil)

                        #                # stack the attention masks which belong to the same kitti image in one big tensor
                        attention_masks_per_kitti_img = torch.cat(
                            (attention_masks_per_kitti_img, attention_mask_from_pil), dim=0)

                attention_masks_per_kitti_img[attention_masks_per_kitti_img >= 0.8] = 1
                attention_masks_per_kitti_img[attention_masks_per_kitti_img < 0.8] = 0

                weight_per_mask = calculate_weight_per_mask(attention_masks_per_kitti_img, threshold)

                # initialize the weight matrix per kitti image
                weight_mask = copy.deepcopy(attention_masks_per_kitti_img.clone().detach())
                divide_mask = copy.deepcopy(attention_masks_per_kitti_img.clone().detach())

                for i, current_mask in enumerate(weight_mask):
                    # overal waar een attention pixel is, zet daar de weight neer
                    current_mask[current_mask == 1] = weight_per_mask[i]
                    weight_mask[i] = current_mask

                if method == 'avg':

                    for i, curr_mask in enumerate(divide_mask):
                        # overal waar een pixel is zet daar een 1 neer zodat je later weer hoeveel overlappende masks per pixels er zijn
                        curr_mask[curr_mask == 1] = 1
                        divide_mask[i] = curr_mask

                    weight_mask = weight_mask.sum(0)
                    divide_mask = divide_mask.sum(0)

                    end_mask = (weight_mask / divide_mask)

                    # replace nan
                    end_mask[end_mask != end_mask] = 0

                    # add 1 everywhere becasue otherwise the ssim loss , l1 loss will shrink
                    end_mask = end_mask + 1

                if method == 'min':
                    # set all zero element on 999 such that the torch. min doesn't find 0 values for overlapping pixels
                    weight_mask[weight_mask == 0] = 99997

                    # find lowest value and now the 0 values are skipped because they are cased to 999
                    min_values = torch.min(weight_mask, dim=0)[0]

                    # but for some pixels there are only 0 values so cast back to zero after torch. min
                    min_values[min_values == 99997] = 0

                    # add 1 everywhere becasue otherwise the ssim loss , l1 loss will shrink
                    end_mask = min_values + 1

                if method == 'max':
                    end_mask = weight_mask.max(dim=0)[0]
                    end_mask = end_mask + 1

                if not os.path.exists(weight_path):
                    os.makedirs(weight_path)

                torch.save(end_mask, weight_path + "/" + "threshold_" + str(threshold) + "_method_" + method + ".pt")

                print(time.time() - start)

