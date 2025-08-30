#
# @Date: 2024-08-27 15:54:45
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-11-10 10:23:43
# @FilePath:
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

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 device=None) -> None:
        super(LinearComplex, self).__init__()
        self.lr = nn.Linear(in_features, out_features, bias, device=device)
        self.li = nn.Linear(in_features, out_features, bias, device=device)        
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
    def forward(self, input: Tensor) -> Tensor:
        bs, ch = input.shape
        real = input[:,:ch//2]
        imag = input[:,ch//2:]
        if self.bias_imag is None:
            a = self.lr(real) - self.li(imag)
            b = self.lr(imag) + self.li(real)
        else:
            a = self.lr(real) - self.li(imag) + self.bias_real
            b = self.lr(imag) + self.li(real) + self.bias_imag
        return torch.cat((a,b),1)

class ComplexLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_real, input_imag, weight_real, weight_imag, bias_real=None, bias_imag=None):
        # 保存前向传播的变量以供反向传播使用
        ctx.save_for_backward(input_real, input_imag, weight_real, weight_imag, bias_real, bias_imag)

        # 计算复数矩阵乘法的实部和虚部
        output_real = input_real.mm(weight_real.t()) - input_imag.mm(weight_imag.t())
        output_imag = input_real.mm(weight_imag.t()) + input_imag.mm(weight_real.t())
        
        # 添加偏置（如果有）
        if bias_real is not None:
            output_real += bias_real
            output_imag += bias_imag

        return output_real, output_imag

    @staticmethod
    def backward(ctx, grad_output_real, grad_output_imag):
        # 从上下文中取出保存的变量
        input_real, input_imag, weight_real, weight_imag, bias_real, bias_imag = ctx.saved_tensors

        # 计算输入和权重的梯度
        grad_input_real = grad_input_imag = grad_weight_real = grad_weight_imag = grad_bias_real = grad_bias_imag = None

        # 计算输入的梯度
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            grad_input_real = grad_output_real.mm(weight_real) + grad_output_imag.mm(weight_imag)
            grad_input_imag = grad_output_imag.mm(weight_real) - grad_output_real.mm(weight_imag)

        # 计算权重的梯度
        if ctx.needs_input_grad[2] or ctx.needs_input_grad[3]:
            grad_weight_real = grad_output_real.t().mm(input_real) + grad_output_imag.t().mm(input_imag)
            grad_weight_imag = grad_output_imag.t().mm(input_real) - grad_output_real.t().mm(input_imag)

        # 计算偏置的梯度
        if bias_real is not None and ctx.needs_input_grad[4]:
            grad_bias_real = grad_output_real.sum(dim=0)
            grad_bias_imag = grad_output_imag.sum(dim=0)

        return grad_input_real, grad_input_imag, grad_weight_real, grad_weight_imag, grad_bias_real, grad_bias_imag
    
class ComplexLinear(torch.nn.Module):
    def __init__(self, input_features, output_features):
        super(ComplexLinear, self).__init__()
        # 初始化权重和偏置
        self.weight_real = torch.nn.Parameter(torch.randn(output_features, input_features))
        self.weight_imag = torch.nn.Parameter(torch.randn(output_features, input_features))
        self.bias_real = torch.nn.Parameter(torch.randn(output_features))
        self.bias_imag = torch.nn.Parameter(torch.randn(output_features))

    def forward(self, inp, dim_ri = 0):
        # 调用 ComplexLinearFunction 的前向和反向传播
        if dim_ri == 0:
            input_real = inp[:inp.size(0) // 2, ...]
            input_imag = inp[inp.size(0) // 2:, ...]
        elif dim_ri == 1:
            input_real = inp[:, :inp.size(1) // 2]
            input_imag = inp[:, inp.size(1) // 2:]
        tmp_r, tmp_i = ComplexLinearFunction.apply(input_real, input_imag, self.weight_real, self.weight_imag, self.bias_real, self.bias_imag)
        return torch.cat((tmp_r, tmp_i), dim_ri)
    