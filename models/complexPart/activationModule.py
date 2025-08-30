#
# @Date: 2024-06-17 16:48:08
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-11-07 17:26:52
# @FilePath: 
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: A Branch of activation function.
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
from torch.autograd import Function
import torch.nn.functional as F
import torch,os

"""
存在点疑问 #@todo
grad_output是复数吗？————来自于chatgpt：input和output都是复数，grad_output也是复数
那么在反向传的过程中可能要做复数乘法的问题了
"""

def complex_relu(inp):
    input_real = inp[:, :inp.size(1) // 2, ...]
    input_imag = inp[:, inp.size(1) // 2:, ...]
    return torch.concat((F.relu(input_real).type(inp.dtype), F.relu(input_imag).type(inp.dtype)),1)

def complex_leakyrelu(inp): 
    input_real = inp[:, :inp.size(1) // 2, ...]
    input_imag = inp[:, inp.size(1) // 2:, ...]
    return torch.concat((F.leaky_relu(input_real).type(inp.dtype), F.leaky_relu(input_imag).type(inp.dtype)),1)

def complex_sigmoid(inp):
    input_real = inp[:, :inp.size(1) // 2, ...]
    input_imag = inp[:, inp.size(1) // 2:, ...]
    return torch.concat((F.sigmoid(input_real).type(inp.dtype), F.sigmoid(input_imag).type(inp.dtype)),1)


def complex_tanh(inp):
    input_real = inp[:, :inp.size(1) // 2, ...]
    input_imag = inp[:, inp.size(1) // 2:, ...]
    return torch.concat((F.tanh(input_real).type(inp.dtype), F.tanh(input_imag).type(inp.dtype)),1)
        
def complex_silu(inp):
    input_real = inp[:, :inp.size(1) // 2, ...]
    input_imag = inp[:, inp.size(1) // 2:, ...]
    return torch.concat((F.silu(input_real).type(inp.dtype), F.silu(input_imag).type(inp.dtype)),1)

def complex_sigmoid2(inp):
    """
    map from complex number to real number
    """
    input_real = inp[:, :inp.size(1) // 2,...]
    input_imag = inp[:, inp.size(1) // 2:,...]
    return F.sigmoid((input_real+input_imag)/1.41421356237)
    
        