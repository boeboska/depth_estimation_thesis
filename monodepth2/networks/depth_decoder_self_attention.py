# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoderSelfAttention(nn.Module):
    def __init__(self, num_ch_enc, self_attention, scales=range(4) , num_output_channels=1, use_skips=True):
        super(DepthDecoderSelfAttention, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.self_attention = self_attention

        self.num_ch_enc = num_ch_enc

         # 1, 64, 96, 320
         # 1, 64, 48, 160
         # 1, 128, 24, 80
         # 1, 256, 12, 40
         # 1, 512, 6, 20


        # self.num_ch_dec= np.array([64, 64, 128, 256, 2048])
        # self.num_ch_enc = np.array([64, 64, 128, 256, 2048]) # dit zijn de shapes die de encoder heeft

        self.num_ch_dec = np.array([64, 32, 64, 128, 256])


        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # breakpoint()
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            # print("eerst", num_ch_in)
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
                # print("extra en daarna",self.num_ch_enc[i - 1], num_ch_in )
            num_ch_out = self.num_ch_dec[i]

            # print("iii, IN, OUT", i, num_ch_in, num_ch_out)

            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def upsample(x):
        """Upsample input tensor by a factor of 2
        """
        return F.interpolate(x, scale_factor=2, mode="nearest")

    def forward(self, input_features, attention_maps, attention_maps_test):
        self.outputs = {}

        x = input_features[-1]

        for i in range(4, -1, -1):

            x = self.convs[("upconv", i, 0)](x)

            if i < 3:
                x = [upsample(x)]
            else:
                x = [x]

            if self.use_skips and i > 0:

                x += [input_features[i - 1]]

            x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 1)](x)

            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        self.outputs['self_attention_maps'] = attention_maps
        self.outputs['self_attention_maps_test'] = attention_maps_test

        # dict_keys([
        # ('disp', 3) : 1, 1, 24, 80
        # ('disp', 2) : 1, 1, 48, 160
        # ('disp', 1) :1, 1, 96, 320
        # ('disp', 0) : 1, 1, 192, 640

        return self.outputs
