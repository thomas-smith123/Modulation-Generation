#
# @Date: 2024-06-17 16:39:47
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-10-24 11:49:24
# @FilePath: /hy_bak_test_delete_after_used/complex_gru_single_radioml/model/complexPart/convComplex.py
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
from torch.autograd import Function
import os
__all__ = ['Conv2dComplex']

def apply_complex(fr, fi, input, dtype=torch.float32):
    if input.dim() > 2:
        input_real = input[:, :input.size(1) // 2, ...]
        input_imag = input[:, input.size(1) // 2:, ...]
        dim = 1
    else:
        input_real = input[:,:input.size(1) // 2]
        input_imag = input[:,input.size(1) // 2:]
        dim = 1
    
    return torch.concat(((fr(input_real)-fi(input_imag)).type(dtype) \
        ,(fr(input_imag)+fi(input_real)).type(dtype)),dim)

class Conv2dComplex(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode='zeros',
    ):
        super(Conv2dComplex, self).__init__()
        self.conv_r = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.conv_i = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )

    def forward(self, inp):
        # return self.conv(inp)
        return apply_complex(self.conv_r, self.conv_i, inp, self.conv_i.weight.dtype)

# class Conv2dComplex2real(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=1,
#         padding=0,
#         dilation=1,
#         groups=1,
#         bias=True,
#     ):
#         super(Conv2dComplex2real, self).__init__()
#         self.conv_r = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             dilation,
#             groups,
#             bias,
#         )
#         self.conv_i = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             padding,
#             dilation,
#             groups,
#             bias,
#         )

#     def forward(self, input):
#         input_real = input[:, :input.size(1) // 2, ...]
#         input_imag = input[:, input.size(1) // 2:, ...]
#         return apply_complex(self.conv_r, self.conv_i, inp, self.conv_i.weight.dtype)

if __name__=="__main__":
    ## For test only
    class ComplexNet(nn.Module):
        def __init__(self):
            super(ComplexNet, self).__init__()
            self.conv1 = Conv2dComplex(2, 2, 3, bias=False)  # 第1层卷积
            # self.conv2 = Conv2dComplex(12, 12, 3)  # 第2层卷积

        def forward(self, x_real):
            x_real = self.conv1(x_real)

            # x_real = self.conv2(x_real)
            return x_real
    a = ComplexNet()
    # m = Conv2dComplex(16, 34, 3, stride=1)
    a.conv1.weight_r = nn.Parameter(torch.ones_like(a.conv1.weight_r))
    a.conv1.weight_i = nn.Parameter(torch.ones_like(a.conv1.weight_i)*2)
    
    input1 = torch.ones(1, 2, 9, 9, dtype=torch.float32,requires_grad=True)*2
    
    p = a(input1)
    print(p)
    loss = p.sum()
    p.backward(torch.randn(p.shape))
    print(input1.grad)
    # p = m(input)
    # print(p.shape)
    # loss = p.sum()
    # loss.backward()
    pass
