#
# @Date: 2024-08-27 15:54:45
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-09-23 15:33:21
# @FilePath: /hy_bak_test_delete_after_used/complex_gru_single/model/complexPart/linearComplex.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#

import torch.nn.functional as F
import torch
import torch.nn as nn
# from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import math
from torch.nn import init as init

class LinearComplex(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None) -> None:
        super(LinearComplex, self).__init__()
        self.lr = nn.Linear(in_features, out_features, bias, device=device)
        self.li = nn.Linear(in_features, out_features, bias, device=device)        

    def forward(self, input: Tensor) -> Tensor:
        # bs, ch, h, w = input.shape
        real = input[:input.size(0)//2,...]
        imag = input[input.size(0)//2:,...]
        a = self.lr(real) - self.li(imag)
        b = self.lr(imag) + self.li(real)
        return torch.cat((a,b),0)
