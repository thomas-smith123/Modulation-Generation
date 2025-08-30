#
# @Date: 2024-06-17 16:49:24
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2025-08-30 16:13:08
# @FilePath: \MG-orphan\models\complexPart\activationComplex.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch,os
import torch.nn as nn
from models.complexPart.activationModule import *
import torch.nn.functional as F
__all__ = ['ComplexReLu', 'ComplexSiLu', 'ComplexCartHardSigmoid', 'ModReLU', 'ZReLU', 'ComplexLeakyReLU_','ComplexLeakyReLU']

# os.environ['DEBUG'] = 'True'
class ComplexReLU(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_relu(inp)

class ComplexLeakyReLU(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_leakyrelu(inp)

class ComplexSigmoid(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid(inp)
      
class ComplexPReLU(nn.Module):
    def __init__(self):
        super(ComplexPReLU,self).__init__()
        self.r_prelu = torch.nn.PReLU()        
        self.i_prelu = torch.nn.PReLU()

    # @staticmethod
    def forward(self, inp):

        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        
        return torch.concat((self.r_prelu(input_real),self.i_prelu(input_imag)),1)

class ComplexTanh(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)

class ComplexSiLu(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_silu(inp)
    
class ComplexSiLu_v2(nn.Module):
    pass

class ComplexSigmoid2(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid2(inp)

class ZReLU(nn.Module):
    def __init__(self):
        super(ZReLU, self).__init__()
    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        real = torch.nn.functional.relu(input_real)
        imag = torch.nn.functional.relu(input_imag)
        mask = (real > 0) & (imag > 0)
        output = torch.concat((real * mask,imag * mask),1)
        return output
    
class ModReLU(nn.Module):
    def __init__(self, num_features):
        super(ModReLU, self).__init__()
        # 可学习的偏置参数
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, inp):
        # 计算复数的模
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        
        modulus = torch.sqrt(input_real ** 2 + input_imag ** 2)
        # ReLU激活
        activated_modulus = torch.nn.functional.silu(modulus - self.bias[None,:,None,None])
        
        output = torch.concat((activated_modulus*input_real/modulus.clamp(min=1e-6), activated_modulus*input_imag/modulus.clamp(min=1e-6)), 1)
        return output

class iGaussian(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(iGaussian, self).__init__(*args, **kwargs)
        self.sigma = torch.tensor(2.0)
        
    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        modulus = torch.sqrt(input_real ** 2 + input_imag ** 2)
        real = (1-torch.exp(-input_real**2/(2*self.sigma**2)))*input_real/modulus
        imag = (1-torch.exp(-input_imag**2/(2*self.sigma**2)))*input_imag/modulus
        return torch.concat((real,imag),1)
        
class cardioid(nn.Module):
    def __init__(self) -> None:
        super(cardioid, self).__init__()
        self.sigma = torch.tensor(2.0)
        
    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        
        angle = torch.atan2(input_imag,input_real)
        return torch.concat((0.5*(1+torch.cos(angle))*input_real,0.5*(1+torch.cos(angle))*input_imag),1)
    
