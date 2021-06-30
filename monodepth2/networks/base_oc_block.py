import torch
import os
import sys
import pdb
import numpy as np
from torch import nn
from torch.nn import functional as F
import functools
# from inplace_abn.bn import InPlaceABNSync, InPlaceABN

# ABN_module = InPlaceABN
# BatchNorm2d = functools.partial(ABN_module, activation='none')


class _SelfAttentionBlock(nn.Module):
    """
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Args:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    """

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(_SelfAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels is None:
            self.out_channels = in_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_key = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                      kernel_size=1, stride=1, padding=0),

            # ABN_module(self.key_channels),
            nn.BatchNorm2d(self.key_channels),
            nn.ReLU(inplace=False),
        )
        self.f_query = self.f_key

        # 256 > 256
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels,
                                 kernel_size=1, stride=1, padding=0)

        # 256 > 256
        self.W = nn.Conv2d(in_channels=self.value_channels, out_channels=self.out_channels,
                           kernel_size=1, stride=1, padding=0)

        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):

        # 1, 6, 20
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        # scale = 2
        if self.scale > 1:

            # 1, 256, 3, 10
            x = self.pool(x)

        # 1, 256, 30
        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        # 1, 30, 256
        value = value.permute(0, 2, 1)

        # breakpoint()

        # f_query(x)  = 1, 128, 3, 10 >>>  .view() >> 1, 128, 30
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        # 1, 30, 128
        query = query.permute(0, 2, 1)

        # breakpoint()
        # f_key(x) = 1, 128, 3, 10 >> .view() >>> 1, 128, 30
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        # breakpoint()
        # query 1, 30, 128    key 1, 128, 30
        # breakpoint()
        sim_map = torch.matmul(query, key) # 1, 30, 30

        # 0.088 * sim map
        sim_map = (self.key_channels ** -.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1) # 1, 30, 30

        # breakpoint()
        # sim_map 1, 30, 30   value 1, 30, 256
        context = torch.matmul(sim_map, value) # 1, 30, 256
        context = context.permute(0, 2, 1).contiguous() # 1, 256, 30
        # breakpoint()
        context = context.view(batch_size, self.value_channels, *x.size()[2:]) # 1, 256, 3, 10
        # breakpoint()
        context = self.W(context)

        # breakpoint()
        # print( "SCLAE ", self.scale)
        # print("eerst", context.shape)
        # self.scale = 2
        if self.scale > 1:
            context = F.upsample(input=context, size=(h, w), mode='bilinear', align_corners=True)
        # print("NA UPSAMPLE", context.shape)
        return context


class SelfAttentionBlock2D(_SelfAttentionBlock):
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(SelfAttentionBlock2D, self).__init__(in_channels,
                                                   key_channels,
                                                   value_channels,
                                                   out_channels,
                                                   scale)


class BaseOC_Module(nn.Module):
    """
    Implementation of the BaseOC module
    Args:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: we choose 0.05 as the default value.
        size: you can apply multiple sizes. Here we only use one size.
    Return:
        features fused with Object context information.
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            # ABN_module(out_channels),
            nn.Dropout2d(dropout)
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):
        priors = [stage(feats) for stage in self.stages]
        context = priors[0]
        for i in range(1, len(priors)):
            context += priors[i]
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


# THIS SELF ATTENTION BLOCK IS PERFORMED
class BaseOC_Context_Module(nn.Module):
    """
    Output only the context features.
    Args:
        in_features / out_features: the channels of the input / output feature maps.
        dropout: specify the dropout ratio
        fusion: We provide two different fusion method, "concat" or "add"
        size: we find that directly learn the attention weights on even 1/8 feature maps is hard.
    Return:
        features after "concat" or "add"
    """

    def __init__(self, in_channels, out_channels, key_channels, value_channels, dropout, sizes=([1])):
        super(BaseOC_Context_Module, self).__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, out_channels, key_channels, value_channels, size) for size in sizes])
        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
            # ABN_module(out_channels),
        )

    def _make_stage(self, in_channels, output_channels, key_channels, value_channels, size):
        return SelfAttentionBlock2D(in_channels,
                                    key_channels,
                                    value_channels,
                                    output_channels,
                                    size)

    def forward(self, feats):

        # breakpoint()
        # print("FORWARD HIER")
        # feats = 1, 256, 6, 20

        priors = [stage(feats) for stage in self.stages]

        # breakpoint()
        context = priors[0]  # 1, 256, 24, 80

        for i in range(1, len(priors)):

            context += priors[i]

        # contecxt = 1, 256, 24, 80

        output = self.conv_bn_dropout(context)

        # output 1, 256, 24, 80

        return output
