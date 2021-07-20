# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo

from networks.asp_oc_block import ASP_OC_Module
from networks.prepare_self_attention import PREPARE_SELF_ATTENTION_MODULE

from networks.util import conv3x3, Bottleneck

import matplotlib.pyplot as plt
import matplotlib
import os
from random import randrange
import seaborn as sns
from random import randrange

affine_par = True

class ResNetMultiImageInputSelfAttention(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1, top_k = 0):
        print("self attention resnet ")
        super(ResNetMultiImageInputSelfAttention, self).__init__(block, layers)

        self.inplanes = 64

        # add extra channels because the input have more channels becuase of the attention masks
        self.conv1 = nn.Conv2d(
        num_input_images * 3 + top_k, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2]) # removed stride
        self.layer4 = self._make_layer(block, 512, layers[3]) # removed stride

        # self.prepare_self_attention = nn.Sequential(
        # self.prepate_self_attention_1 = self._make_layer(block, 128, layers[0])
        # self.prepate_self_attention_2 = self._make_layer(block, 256, layers[0])
        # self.prepate_self_attention_3 = self._make_layer(block, 512, layers[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




def resnet_multiimage_inputSelfAttention(num_layers, pretrained=False, num_input_images=1, top_k = 0):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInputSelfAttention(block_type, blocks, num_input_images=num_input_images, top_k = top_k)

    if pretrained:
        print("NUM LAYERS", num_layers)
        # breakpoint()
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoderSelfAttention(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1, top_k=0):
        super(ResnetEncoderSelfAttention, self).__init__()

        # self.batch_idx = batch_idx

        # self.num_ch_enc = np.array([128, 256, 512, 1024, 2048])
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        # self.resnet_model = ResNet(Bottleneck, [3, 4, 23, 3])

        self.context = nn.Sequential(

            # go from 2048 channels to 512 channels
            # nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1),
            # # ABN_module(512),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=False),

            ASP_OC_Module(512, 256)
        )

        # self.prepare_self_attention = nn.Sequential(
        #     PREPARE_SELF_ATTENTION_MODULE(64)
        # )

        # num classes 128 ?? komt uit de andere resnet
        self.cls = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=True)

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        self.num_input_images = num_input_images
        self.top_k = top_k

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_inputSelfAttention(num_layers, pretrained, num_input_images, top_k)
        else:
            self.encoder = resnet_multiimage_inputSelfAttention(num_layers, pretrained, num_input_images, top_k)
            # self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    #     layers = [3, 4, 23, 3]
    #
    #     self.inplanes = 128
    #     self.conv1 = conv3x3(3, 64, stride=1, dilation=2, padding=2)
    #     self.bn1 = nn.BatchNorm2d(64)
    #     self.relu1 = nn.ReLU(inplace=False)
    #     self.conv2 = conv3x3(64, 64, stride=2)
    #     self.bn2 = nn.BatchNorm2d(64)
    #     self.relu2 = nn.ReLU(inplace=False)
    #     self.conv3 = conv3x3(64, 128)
    #     self.bn3 = nn.BatchNorm2d(128)
    #     self.relu3 = nn.ReLU(inplace=False)
    #
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
    #     self.relu = nn.ReLU(inplace=False)
    #     self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # change
    #     self.layer1 = self._make_layer(Bottleneck, 64, layers[0])
    #     self.layer2 = self._make_layer(Bottleneck, 128, layers[1], stride=2)
    #     self.layer3 = self._make_layer(Bottleneck, 256, layers[2], stride=1, dilation=2)
    #     self.layer4 = self._make_layer(Bottleneck, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))
    #
    # def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
    #
    #     layers = []
    #     generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
    #     layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
    #                         multi_grid=generate_multi_grid(0, multi_grid)))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(
    #             block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))
    #     return nn.Sequential(*layers)

    def forward(self, input_image, batch_idx, inputs, hist_dict, masks = None):
        self.features = []
        x = (input_image - 0.45) / 0.225


        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x))   # 1, 64, 96, 320
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1]))) # 1, 64, 48, 160
        self.features.append(self.encoder.layer2(self.features[-1])) # 1, 128, 24, 80 >> 24, 80
        self.features.append(self.encoder.layer3(self.features[-1])) # 1, 256, 12, 40 >> 24, 80
        self.features.append(self.encoder.layer4(self.features[-1])) # 1, 512, 6, 20 >> 24, 80


        output, feat1, visualize_query, visualize_key, visualize_value, sim_map = self.context(self.features[-1])


        self.features[-1] = output
        attention_maps = output

        # only during training .. not during valdiation
        if hist_dict != None:

            for x in range(15):


                rand_nr = randrange(attention_maps.shape[1])

                for i in range(len(list(hist_dict.keys()))):

                    # breakpoint()

                    print("ii", i)

                    # make sure you also get items > 4
                    if list(hist_dict.keys())[i + 1] == list(hist_dict.keys())[-1]:
                        print("BIJ DE LAATSTE")

                        list(hist_dict.keys())[i + 1] = np.inf


                    # take current attention map and filter on nonzero values
                    temp = attention_maps.squeeze()[rand_nr][attention_maps.squeeze()[rand_nr] > 0].cpu().clone()

                    # sort the attention mask in bins
                    temp = temp[temp > list(hist_dict.keys())[i]]
                    temp = temp[temp < list(hist_dict.keys())[i + 1]]


                    curr = hist_dict[list(hist_dict.keys())[i]]

                    # breakpoint()
                    # how many % of the whole image contains such high value
                    curr.append( temp.shape[0] / (attention_maps.shape[2] + attention_maps.shape[3]))

                    hist_dict[list(hist_dict.keys())[i]] = curr



                    if list(hist_dict.keys())[i + 1] == np.inf:
                        print("IK GA BREAKEN")
                        break

            # if x.item() == 0:
            #     print("HAAA")
            # divider = np.floor(x.item() / 0.05)
            # pos_in_dict = divider * 0.05


            # hist_dict[pos_in_dict] += 1

        # self.features[-1] = self.context(self.features[-1])  # attention maps > 1, 256, 48, 160

        # breakpoint()
        # attention_maps = self.features[-1] # 1, 512, 24, 80

        # during label checking None is returned
        # if inputs is not None and batch_idx is not None:
        #     path = 'monodepth_models/14_07/vis_query'
        #     # path = self.log_dir + self.model_name + "/" + "vis_query/"
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #
        #     path = f'{path}/{batch_idx}'
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #
        #     for x in range(10):
        #         rand_nr = randrange(visualize_query.shape[1])
        #
        #         fig, axis = plt.subplots(7, 1, figsize=(10, 15))
        #
        #         original_img = inputs["color_aug", 0, 0]
        #
        #         original_img = np.array(original_img[0].squeeze().cpu().detach().permute(1, 2, 0).numpy())
        #         axis[0].set_title('Kitti image', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[0].axis('off')
        #         axis[0].imshow(original_img)
        #
        #
        #
        #         axis[1].set_title('query ', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[1].axis('off')
        #         sns.heatmap(visualize_query[0][rand_nr].cpu().detach().numpy(), ax=axis[1], vmin=0, vmax=0.5, cmap='Greens',
        #                     center=0.25)
        #         # np.save(os.path.join(path, 'query.npy'), visualize_query[0][rand_nr].cpu().detach().numpy())
        #
        #         axis[2].set_title('key ', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[2].axis('off')
        #         sns.heatmap(visualize_key[0][rand_nr].cpu().detach().numpy(), ax=axis[2], vmin=0, vmax=0.5, cmap='Greens',
        #                     center=0.25)
        #         # np.save(os.path.join(path, 'key.npy'), visualize_key[0][rand_nr].cpu().detach().numpy())
        #
        #         axis[3].set_title('value ', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[3].axis('off')
        #         sns.heatmap(visualize_value[0][rand_nr].cpu().detach().numpy(), ax=axis[3], vmin=0, vmax=0.5, cmap='Greens',
        #                     center=0.25)
        #         # np.save(os.path.join(path, 'value.npy'), visualize_value[0][rand_nr].cpu().detach().numpy())
        #
        #         axis[4].set_title('sim map ', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[4].axis('off')
        #         sns.heatmap(sim_map[0].cpu().detach().numpy(), ax=axis[4], vmin=0, vmax=0.01, cmap='Greens',
        #                     center=0.005)
        #         # np.save(os.path.join(path, 'sim_map.npy'), sim_map[0].cpu().detach().numpy())
        #
        #         axis[5].set_title('attention map w/o dial ', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[5].axis('off')
        #         sns.heatmap(feat1[0][rand_nr].cpu().detach().numpy(), ax=axis[5], vmin=0, vmax=0.1, cmap='Greens',
        #                     center=0.05)
        #         # np.save(os.path.join(path, 'feat_1.npy'), feat1[0][rand_nr].cpu().detach().numpy())
        #
        #         axis[6].set_title('attention map w dialiation ', fontdict={'fontsize': 20, 'fontweight': 'bold'})
        #         axis[6].axis('off')
        #         sns.heatmap(output[0][rand_nr].cpu().detach().numpy(), ax=axis[6], vmin=0, vmax=2, cmap='Greens',
        #                     center=1)
        #         # np.save(os.path.join(path, 'attention_map.npy'), output[0][rand_nr].cpu().detach().numpy())
        #
        #
        #
        #
        #
        #
        #
        #         fig.savefig(f'{path}/batch_idx_{batch_idx}_rand{rand_nr}.png')
        #         plt.close(fig)


        # all features[-1] dit is nu de self attention ding ... 1, 512, 24, 80 PAPER (h/8, w/8 , 512)
        # deze is 1, 512, 6, 20
        # breakpoint()

        # self.features[-1] = self.cls(self.features[-1])
        # breakpoint()
        # paper =  # 1, 2048, 24, 80

        # deze 1, 2048 << deze kan ik kiezen?, 6, 20
        # breakpoint()

        # self.features = []
        # x = (input_image - 0.45) / 0.225
        # x = self.relu1(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        # self.features.append(x)
        # x = self.maxpool(x)
        # x = self.layer1(x)
        # self.features.append(x)
        # x = self.layer2(x)
        #
        # self.features.append(x)
        # # breakpoint()
        # x = self.layer3(x)
        # self.features.append(x)
        # # x_dsn = self.dsn(x)
        # x = self.layer4(x)
        # self.features.append(x)
        #
        # #0 1, 128, 96, 320
        # # 1, 2048, 24, 80
        #
        # breakpoint()

        return self.features, attention_maps, hist_dict


        # return self.features
