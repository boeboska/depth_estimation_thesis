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
        self.layer2 = self._make_layer(block, 128, layers[1])
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

        self.prepare_self_attention = nn.Sequential(
            PREPARE_SELF_ATTENTION_MODULE(64)
        )

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

    def forward(self, input_image, masks = None):
        self.features = []
        x = (input_image - 0.45) / 0.225

        # print("num input img", self.num_input_images) print("top k  en masks", self.top_k, masks.shape)
        if self.top_k > 0 and self.num_input_images != 2 and masks is not None:
            x = torch.cat((x, masks), dim = 1)

        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)

        self.features.append(self.encoder.relu(x))   # 1, 64, 96, 320
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1]))) # 1, 64, 48, 160
        self.features.append(self.encoder.layer2(self.features[-1])) # 1, 128, 24, 80 >> 48, 160
        self.features.append(self.encoder.layer3(self.features[-1])) # 1, 256, 12, 40 >> 24, 80
        self.features.append(self.encoder.layer4(self.features[-1])) # 1, 512, 6, 20 >> 24, 80


        # pre_self_attention_features = self.prepare_self_attention(self.features[1]) # 1, 512, 48, 160

        self.features[-1], attention_maps_test = self.context(self.features[-1])  # attention maps > 1, 256, 48, 160

        attention_maps = self.features[-1] # 1, 512, 24, 80


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

        return self.features, attention_maps, attention_maps_test


        # return self.features
