#
# @Date: 2024-06-17 16:49:24
# @LastEditors: jiangrd3 thomas-smith@live.cn
# @LastEditTime: 2025-08-14 17:36:16
# @FilePath: /complex_gru_single_radioml/model/complexPart/activationComplex.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch,os
import torch.nn as nn
try:
    from model.complexPart.activationModule import *
    from model.complexPart.stableAtan2 import StableAtan2
except:
    from complexPart.activationModule import *
    from complexPart.stableAtan2 import StableAtan2
import torch.nn.functional as F


__all__ = ['ComplexReLu', 'ComplexSiLu', 'ModReLU', 'ZReLU', 'ComplexLeakyReLU', 'ComplexSigmoid2']
"""
激活函数三种形式
1. 实部虚部分开，以非解析性保持有界
2. 非解析的相位幅度形式
3. 全复形式的解析有界函数，但是存在奇异值
第一种：
$$
\sigma(z)=\sigma(x,y)=u(x,y)+iv(x,y)\\
z=x+iy
$$
后两种：
$$
\sigma(z,\bar{z})
$$
求导最后都变为对实部和虚部的求导形式.
对第一种，使用wirtinger derivatives算子求导
wirtinger derivatives算子
$$
\nabla_w=\frac{\partial}{\partial z}=\frac{1}{2}\left(\frac{\partial}{\partial x}-i\frac{\partial}{\partial y}\right)
$$
计算得到导数为
$$
\nabla_w\sigma(z)=\frac{1}{2}\left(\frac{\partial u(x,y)}{\partial x}+
\frac{\partial v(x,y)}{\partial y}\right)-
\frac{i}{2}\left(\frac{\partial u(x,y)}{\partial y}-
\frac{\partial v(x,y)}{\partial x}\right)
$$
左边一项是实部，右边一项是虚部
对后两种，直接算对z的导数，和前面基本是一样的，只是表达形式不同罢了
"""
# os.environ['DEBUG'] = 'True'
class ComplexReLu(nn.Module):
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

class ComplexReLU_A(nn.Module):
    def __init__(self):
        super(ComplexReLU_A,self).__init__()
        self.relu = torch.nn.ReLU()

    # @staticmethod
    def forward(self, inp):

        input_real = inp[:inp.size(0) // 2, ...]
        input_imag = inp[inp.size(0) // 2:, ...]
        
        return self.relu((input_real+input_imag)/torch.sqrt(torch.tensor(2)))

class ComplexSigmoid2(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid2(inp)
    
class ComplexTanh(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_tanh(inp)

class ComplexTanhModified(nn.Module):
    @staticmethod
    def forward(inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        m = torch.sqrt(input_real**2+input_imag**2 + 1e-6)
        r = torch.tanh((input_real+input_imag)/torch.sqrt(torch.tensor(2.0)))*input_real/(m + 1e-6)
        i = torch.tanh((input_real+input_imag)/torch.sqrt(torch.tensor(2.0)))*input_imag/(m + 1e-6)
        return torch.concat((r,i),1)

class ComplexSiLu(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_silu(inp)
    
class ComplexSigmoid2(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid2(inp)
    
class ComplexSigmoid1(nn.Module):
    @staticmethod
    def forward(inp):
        return complex_sigmoid1(inp)

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
        
        modulus = torch.sqrt(input_real ** 2 + input_imag ** 2 + 1e-6)
        # ReLU激活
        activated_modulus = torch.nn.functional.silu(modulus - self.bias[None,:,None,None])
        
        output = torch.concat((activated_modulus*input_real/modulus.clamp(min=1e-6), activated_modulus*input_imag/modulus.clamp(min=1e-6)), 1)
        return output

class iGaussian(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(iGaussian, self).__init__(*args, **kwargs)
        self.sigma = torch.tensor(2.0)
        self.eps = 1e-4
    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        modulus = torch.sqrt(input_real ** 2 + input_imag ** 2 + self.eps)
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
    
class silu_modified(nn.Module):
    def __init__(self) -> None:
        super(silu_modified, self).__init__()
        self.eps = 1e-4
    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        modulus = torch.sqrt(input_real ** 2 + input_imag ** 2 + self.eps)
        tmp = (input_real+input_imag + self.eps)/1.414213562373095

        real = 1/(1+torch.exp(-tmp))*tmp*input_real/(modulus + self.eps)
        imag = 1/(1+torch.exp(-tmp))*tmp*input_imag/(modulus + self.eps)
        return torch.concat((real,imag),1)

class tanh_modified(nn.Module):
    def __init__(self) -> None:
        super(tanh_modified, self).__init__()
        self.eps = 1e-4
    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        modulus = torch.sqrt(input_real ** 2 + input_imag ** 2 + self.eps)
        tmp = (input_real+input_imag)/1.414213562373095

        real = torch.tanh(tmp)*input_real/(modulus + self.eps)
        imag = torch.tanh(tmp)*input_imag/(modulus + self.eps)
        return torch.concat((real,imag),1)

# class ExpActivation(nn.Module):
#     def __init__(self):
#         super(ExpActivation, self).__init__()
# class ModReLU(nn.Module):
#     def __init__(self, num_features):
#         super(ModReLU, self).__init__()
#     def forward(self, inp):
#         input_real = inp[:, :inp.size(1) // 2, ...]
#         input_imag = inp[:, inp.size(1) // 2:, ...]
#         # 对复数张量进行指数激活
#         # 通过 torch.exp 对复数输入进行指数运算
#         real = torch.nn.functional.silu(input_real)
#         imag = torch.nn.functional.silu(input_imag)
#         result = torch.concat((real,imag), 1)
#         return result