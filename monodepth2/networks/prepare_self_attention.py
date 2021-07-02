import torch.nn as nn
from torch.nn import functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import os
import sys
import pdb
import numpy as np
from torch.autograd import Variable
import functools
# from inplace_abn.bn import InPlaceABNSync, InPlaceABN
# ABN_module = InPlaceABN
# BatchNorm2d = functools.partial(ABN_module, activation='none')


class PREPARE_SELF_ATTENTION_MODULE(nn.Module):
    """
    Network to perform Atrous Spatial Pyramid Pooling (ASPP).
    """

    def __init__(self, features):
        super(PREPARE_SELF_ATTENTION_MODULE, self).__init__()

        self.out_features = [128, 256, 512]

        self.prep_self_attention = nn.Sequential(
            nn.Conv2d(features, self.out_features[0], kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.out_features[0]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_features[0], self.out_features[1], kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.out_features[1]),
            nn.ReLU(inplace=True),

            nn.Conv2d(self.out_features[1], self.out_features[2], kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(self.out_features[2]),
            nn.ReLU(inplace=True)

        )

    def forward(self, features):

        feat1 = self.prep_self_attention(features)

        return feat1
