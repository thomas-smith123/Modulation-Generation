#
# @Date: 2024-06-17 16:39:47
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-11-10 17:06:48
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
        )

    def forward(self, inp):
        # return self.conv(inp)
        return apply_complex(self.conv_r, self.conv_i, inp, self.conv_i.weight.dtype)


class MyConv2d(Function):
    @staticmethod
    def forward(ctx, input, weight, stride=1, padding=0, dilation=1, groups=1, bias=None):
        """
        执行卷积操作，并保存所需的中间值（输入、卷积核、偏置等）用于反向传播。
        
        :param ctx: 用于保存变量的上下文
        :param input: 输入张量 (batch_size, channels, height, width)
        :param weight: 卷积核 (out_channels, in_channels, kernel_height, kernel_width)
        :param stride: 步长
        :param padding: 填充
        :param dilation: 膨胀
        :param groups: 分组卷积
        :param bias: 偏置
        :return: 输出张量
        """
        # 保存必要的参数
        ctx.save_for_backward(input, weight, bias)  # 保存输入和卷积核以及偏置
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        # 进行卷积操作
        input_real = input[:, :input.size(1) // 2, ...]
        input_imag = input[:, input.size(1) // 2:, ...]
        weight_real = weight[:, :weight.size(1) // 2, ...]
        weight_imag = weight[:, weight.size(1) // 2:, ...]
        if bias is not None:
            bias_real = bias[:bias.size(0)//2]
            bias_imag = bias[bias.size(0)//2:]      
          
            output = torch.concat((F.conv2d(input_real, weight_real, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)-F.conv2d(input_imag, weight_imag, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)+bias_real[None,:,None,None], F.conv2d(input_imag, weight_real, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)+F.conv2d(input_real, weight_imag, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)+bias_imag[None,:,None,None]),1)
        else:
            output = torch.concat((F.conv2d(input_real, weight_real, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)-F.conv2d(input_imag, weight_imag, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups), F.conv2d(input_imag, weight_real, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)+F.conv2d(input_real, weight_imag, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)),1)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        手动计算卷积的梯度：dL/dX 和 dL/dK
        
        :param ctx: 用于获取保存的变量
        :param grad_output: 损失相对于输出的梯度 (dL/dY)
        :return: 输入和卷积核的梯度
        """
        input, weight, bias = ctx.saved_tensors  # 获取保存的输入、卷积核和偏置
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        
        input_real = input[:, :input.size(1) // 2, ...]
        input_imag = input[:, input.size(1) // 2:, ...]
        weight_real = weight[:, :weight.size(1) // 2, ...]
        weight_imag = weight[:, weight.size(1) // 2:, ...]
        grad_output_real = grad_output[:, :grad_output.size(1) // 2, ...]
        grad_output_imag = grad_output[:, grad_output.size(1) // 2:, ...]
        

        ##CHECK ? # 计算 dL/dX（输入的梯度）
        
        grad_input_real = 0.5 * (F.grad.conv2d_input(input_real.shape, weight_real.to(grad_output.dtype), grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups) + \
                                 F.grad.conv2d_input(input_imag.shape, weight_imag.to(grad_output.dtype), grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups))
        
        grad_input_imag = 0.5 * (F.grad.conv2d_input(input_imag.shape, weight_real.to(grad_output.dtype), grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups) - \
                                 F.grad.conv2d_input(input_real.shape, weight_imag.to(grad_output.dtype), grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups))
                          
        grad_input = torch.concat((grad_input_real, grad_input_imag),1)/2
        # 计算 dL/dX（输入的梯度）
        # grad_input = F.conv_transpose2d(grad_output, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

        # 计算 dL/dK（卷积核的梯度）
        grad_weight_real = 0.5 * (F.grad.conv2d_weight(input_real.to(grad_output.dtype), weight_real.shape, grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups) + \
                                  F.grad.conv2d_weight(input_imag.to(grad_output.dtype), weight_imag.shape, grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups))
                           
        grad_weight_imag = 0.5 * (-F.grad.conv2d_weight(input_imag.to(grad_output.dtype), weight_real.shape, grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups) + \
                                  F.grad.conv2d_weight(input_real.to(grad_output.dtype), weight_imag.shape, grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups))
                           
        grad_weight = torch.concat((grad_weight_real, grad_weight_imag),1)/2
        # grad_weight = F.conv2d(input, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
        # 计算 dL/dbias（偏置的梯度）
        grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None  # 对每个输出通道求和计算偏置的梯度

        return grad_input, grad_weight, None, None, None, None, grad_bias  # 返回梯度和没有用的额外值
    # def backward(ctx, grad_output):
    #     """
    #     手动计算卷积的梯度：dL/dX 和 dL/dK
        
    #     :param ctx: 用于获取保存的变量
    #     :param grad_output: 损失相对于输出的梯度 (dL/dY)
    #     :return: 输入和卷积核的梯度
    #     """
    #     input, weight, bias = ctx.saved_tensors  # 获取保存的输入、卷积核和偏置
    #     stride = ctx.stride
    #     padding = ctx.padding
    #     dilation = ctx.dilation
    #     groups = ctx.groups
        
    #     input_real = input[:, :input.size(1) // 2, ...]
    #     input_imag = input[:, input.size(1) // 2:, ...]
    #     weight_real = weight[:, :weight.size(1) // 2, ...]
    #     weight_imag = weight[:, weight.size(1) // 2:, ...]
    #     grad_output_real = grad_output[:, :grad_output.size(1) // 2, ...]
    #     grad_output_imag = grad_output[:, grad_output.size(1) // 2:, ...]
        

    #     ##CHECK ? # 计算 dL/dX（输入的梯度）
        
    #     grad_input_real = F.grad.conv2d_input(input_real.shape, weight_real.to(grad_output.dtype), grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups)+\
    #                       F.grad.conv2d_input(input_real.shape, weight_imag.to(grad_output.dtype), grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
    #     grad_input_imag = F.grad.conv2d_input(input_imag.shape, weight_real.to(grad_output.dtype), grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups)-\
    #                       F.grad.conv2d_input(input_imag.shape, weight_imag.to(grad_output.dtype), grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups)
                          
    #     grad_input = torch.concat((grad_input_real, grad_input_imag),1)/2
    #     # 计算 dL/dX（输入的梯度）
    #     # grad_input = F.conv_transpose2d(grad_output, weight, stride=stride, padding=padding, dilation=dilation, groups=groups)

    #     # 计算 dL/dK（卷积核的梯度）
    #     grad_weight_real = F.grad.conv2d_weight(input_real.to(grad_output.dtype), weight_real.shape, grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups)+\
    #                        F.grad.conv2d_weight(input_imag.to(grad_output.dtype), weight_real.shape, grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups)
                           
    #     grad_weight_imag =-F.grad.conv2d_weight(input_imag.to(grad_output.dtype), weight_imag.shape, grad_output_real, stride=stride, padding=padding, dilation=dilation, groups=groups)+\
    #                        F.grad.conv2d_weight(input_real.to(grad_output.dtype), weight_imag.shape, grad_output_imag, stride=stride, padding=padding, dilation=dilation, groups=groups)
                           
    #     grad_weight = torch.concat((grad_weight_real, grad_weight_imag),1)/2
    #     # grad_weight = F.conv2d(input, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)
        
    #     # 计算 dL/dbias（偏置的梯度）
    #     grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None  # 对每个输出通道求和计算偏置的梯度

    #     return grad_input, grad_weight, None, None, None, None, grad_bias  # 返回梯度和没有用的额外值

class Conv2dComplex_(nn.Module):
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
    ):
        super(Conv2dComplex_,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        in_channels = in_channels*2
        
        # self.bias = bias
        self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *self.kernel_size)))
        if bias:
            self.bias = Parameter(torch.empty(out_channels*2))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.conv = MyConv2d.apply
        
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
    def forward(self, inp):
        return self.conv(inp,self.weight,self.stride,self.padding,self.dilation,self.groups,self.bias)
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
                
def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))

if __name__=="__main__":
    ## For test only
    class ComplexNet(nn.Module):
        def __init__(self):
            super(ComplexNet, self).__init__()
            self.conv1 = Conv2dComplex(2, 4, 3, bias=True)  # 第1层卷积
            # self.conv2 = Conv2dComplex(12, 12, 3)  # 第2层卷积

        def forward(self, x_real):
            x_real = self.conv1(x_real)

            # x_real = self.conv2(x_real)
            return x_real
    a = ComplexNet()
    # m = Conv2dComplex(16, 34, 3, stride=1)
    
    input1 = torch.ones(5, 4, 9, 9, dtype=torch.float32,requires_grad=True)*2
    
    p = a(input1)
    # print(p)
    target = torch.ones_like(p)  # 目标张量
    loss = (p - target) ** 2
    loss = loss.sum()  # 求和得到标量损失

    # 反向传播
    loss.backward()

    # 打印梯度
    print("输入的梯度 (dL/dX):")
    print(input1.grad)
    print("卷积核的梯度 (dL/dK):")
    print(a.conv1.weight.grad)
    print("偏置的梯度 (dL/dbias):")
    print(a.conv1.bias.grad)
