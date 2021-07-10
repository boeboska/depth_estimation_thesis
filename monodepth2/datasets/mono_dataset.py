# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms


def pil_loader(path):

    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader_attention(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class MonoDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 convolution_experiment,
                 top_k,
                 seed,
                 weight_mask_method,
                 weight_matrix_path,
                 attention_mask_loss,
                 edge_loss,
                 data_path,
                 attention_path,
                 attention_threshold,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',):
        super(MonoDataset, self).__init__()

        self.convolution_experiment = convolution_experiment
        self.top_k = top_k
        self.seed = seed
        self.weight_mask_method = weight_mask_method
        self.weight_matrix_path = weight_matrix_path
        self.attention_mask_loss = attention_mask_loss
        self.edge_loss = edge_loss
        self.attention_threshold = attention_threshold
        self.attention_path = attention_path
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS
        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.attention_loader = pil_loader_attention
        self.to_tensor = transforms.ToTensor()

        random.seed(self.seed)

        amount_of_masks = {
            0.9: 12,
            0.8: 15,
            0.7: 18,
            0.6: 23,
            0.5: 30,
            0.4: 35,
            0.3: 40,
            0.2: 50
        }

        self.mask_amount = amount_of_masks[self.attention_threshold]


        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        ['or
            "s" for the opposite image in the stereo pair.']

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        # print("GET ITEM")

        # seed = random.Random(self.seed)
        # print("SEED", seed())

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None


        # print("hidde", self.frame_idxs)
        for i in self.frame_idxs:

            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:

                # print("hier", frame_index)

                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

                if i == 0:
                    weight_matrix = self.get_weight_matrix(folder, frame_index, side, do_flip)
                    inputs[("weight_matrix")] = weight_matrix

            # only add the attention masks for the target frame (frame 0)
                if self.edge_loss:
                    if i == 0:

                        # look up attention masks for target frame
                        # attention mask is a dict with key = file name value = (size, mask)
                        attention_masks_dict = self.get_attention(folder, frame_index, side, do_flip)

                        # # attention_masks_dict['68_0_858.jpg']: (235, M)
                        mask_sizes = np.array([attention_masks_dict[key][0] for key in attention_masks_dict])
                        masks = torch.from_numpy(np.vstack([attention_masks_dict[key][1] for key in attention_masks_dict]))

                        mask_order = mask_sizes.argsort()

                        # mask_sizes_sorted = mask_sizes[mask_order]
                        masks_sorted = masks[mask_order]

                        # 100, 192, 640
                        zeros = torch.zeros(100, self.height, self.width)
                        # check how many dimension are missing to create a 100, 192, 640 tensor such then every image got same dimensions
                        diff = zeros.shape[0] - masks_sorted.shape[0]

                        masks_sorted = torch.cat([masks_sorted, torch.zeros(diff, self.height, self.width)])

                        # print(masks_sorted.shape)

                        inputs[("attention")] = masks_sorted

                if self.attention_mask_loss:
                    if i == 0:
                        weight_matrix = self.get_weight_matrix(folder, frame_index, side, do_flip)
                        inputs[("weight_matrix")] = weight_matrix

                if self.convolution_experiment:
                    if i == 0:

                        # look up 100 attention masks for target frame
                        attention_masks, mask_amount = self.get_attention_top_k(folder, frame_index, side, do_flip)

                        assert mask_amount == 100, "There should be 100 attention masks saved for this kitti image. its now {}".format(mask_amount)

                        loop_count = 0
                        a = torch.empty(self.top_k, self.height, self.width)

                        # loop over the dict with keys the prob and value a list of attention mask which have that prob
                        for key, value in sorted(attention_masks.items(), reverse = True):

                            # if you have found enough attention masks based on the top-k number
                            if loop_count == self.top_k:
                                break

                            # if there is only 1 attention mask for this prob
                            if len(value) == 1:

                                a[loop_count] = transforms.ToTensor()(value[0])
                                loop_count += 1

                            # if there are more attention mask with the same prob
                            else:

                                for mask in value:

                                    # if you have found enough attention masks based on the top-k number
                                    if loop_count == self.top_k:
                                        break

                                    a[loop_count] = transforms.ToTensor()(mask)
                                    loop_count +=1

                        inputs[("top_k_masks")] = a







        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = torch.from_numpy(stereo_T)

        # print(inputs.keys())
        # print(inputs[('color', 0, 0)].size())

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
