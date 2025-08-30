#
# @Date: 2024-11-12 10:58:16
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-11-12 10:58:35
# @FilePath: /hy_bak_test_delete_after_used/yolo_hy_complex_network_for_complex_input/models/complexPart/complexDropout.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch
from torch.nn.functional import (
    avg_pool2d,
    dropout,
    dropout2d,
    interpolate,
    max_pool2d,
    relu,
    sigmoid,
    tanh,
)

import torch.nn as nn

from torch.nn.functional import max_pool2d, avg_pool2d, dropout, dropout2d, interpolate
from torch import tanh, relu, sigmoid

def complex_dropout(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp


def complex_dropout2d(inp, p=0.5, training=True):
    # need to have the same dropout mask for real and imaginary part,
    # this not a clean solution!
    mask = torch.ones(*inp.shape, dtype=torch.float32, device=inp.device)
    mask = dropout2d(mask, p, training) * 1 / (1 - p)
    mask.type(inp.dtype)
    return mask * inp

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, input):
        if self.training:
            return complex_dropout(input, self.p)
        else:
            return input


class ComplexDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout2d, self).__init__()
        self.p = p

    def forward(self, inp):
        if self.training:
            return complex_dropout2d(inp, self.p)
        else:
            return inp