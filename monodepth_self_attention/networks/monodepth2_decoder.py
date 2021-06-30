from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc

        #all_features[0] batch, 128, 96, 320
        #all_features[1] batch, 256, 48, 160
        #all_features[2] batch, 512, 24, 80
        #all_features[3] batch, 1024, 24, 80
        #all_features[4] batch, 2048, 24, 80


        # self.num_ch_enc = np.array([128, 256, 512, 1024, 2048]) # dit zijn de channels vd encoder
        self.num_ch_dec = np.array([64, 128, 256, 512, 1024])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            print("iii, IN, OUT", i, num_ch_in, num_ch_out)
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # input features [-1] = 1, 2048, 24, 80
        
        # decoder
        x = input_features[-1]

        # i = [4, 3, 2, 1, 0]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)   # 1, 2024, 24, 80

            # print("eerst", x.shape)
            if i < 3:
                x = [upsample(x)]
            else:
                x = [x]

            # print("2e ", x[0].shape)

            # use_skips = true
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]

            # print("3e", x[0].shape, x[1].shape)

            # breakpoint()
            x = torch.cat(x, 1) # 1, 2048, 48, 80
            # print("4e", x.shape)
            # breakpoint()
            x = self.convs[("upconv", i, 1)](x) # 1, 2014, 24, 80
            # print("5e ", x.shape)
            # breakpoint()
            # breakpoint()
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))
            
        
        # breakpoint()
        # dict_keys([('disp', 3),  # 1, 1, 24, 80
        #            ('disp', 2),  # 1, 1, 48, 160
        #            ('disp', 1),  # 1, 1, 96, 320
        #            ('disp', 0)]) # 1, 1, 192, 640
            
            
            
        return self.outputs
