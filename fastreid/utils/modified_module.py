# encoding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastreid.layers.batch_norm import AdaBatchNorm1d, AdaBatchNorm2d, AdaIBN


class AdaBNSequential(nn.Sequential):
    def __init__(self, *args):
        super(AdaBNSequential, self).__init__(*args)

    def forward(self, input, momentum=0.1):
        for module in self:
            if isinstance(module, AdaBatchNorm1d) or isinstance(module, AdaBatchNorm2d) or isinstance(module, AdaIBN):
                input = module(input, momentum)
            else:
                input = module(input)
        return input


class AdaBNIdentity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AdaBNIdentity, self).__init__()

    def forward(self, input, momentum=None):
        return input
