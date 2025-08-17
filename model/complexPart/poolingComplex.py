#
# @Date: 2024-08-09 16:22:40
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-09-09 11:37:20
# @FilePath: /hy_bak_test_delete_after_used/complex_gru single/model/complexPart/poolingComplex.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
'''
Date: 2024-04-10 10:08:07
LastEditors: jiangrd3 jiangrd3@mail2.sysu.edu.cn
LastEditTime: 2024-06-12 19:40:54
FilePath: /complex_net/poolingComplex.py
'''

from typing import List, Optional
from torch import Tensor
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple, _quadruple
import torch.nn.functional as F
import math,torch,os
from torch.nn.common_types import (_size_any_t, _size_1_t, _size_2_t, _size_3_t,
                            _ratio_3_t, _ratio_2_t, _size_any_opt_t, _size_2_opt_t, _size_3_opt_t)

__all__ = ['ComplexMaxPool2d']

def _retrieve_elements_from_indices(tensor, indices):
    flattened_tensor = tensor.flatten(start_dim=-2)
    output = flattened_tensor.gather(
        dim=-1, index=indices.flatten(start_dim=-2)
    ).view_as(indices)
    return output


def complex_max_pool2d(
    inp,
    kernel_size,
    stride=None,
    padding=0,
    dilation=1,
    ceil_mode=False,
    return_indices=False,
):
    """
    Perform complex max pooling by selecting on the absolute value on the complex values.
    """
    input_real = inp[:,:inp.size(1)//2,...]
    input_imag = inp[:,inp.size(1)//2:,...]
    
    absolute_value, indices = F.max_pool2d(
        torch.sqrt(input_real**2+input_imag**2 + 1e-4),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )
    # performs the selection on the absolute values
    # retrieve the corresponding phase value using the indices
    # unfortunately, the derivative for 'angle' is not implemented
    # get only the phase values selected by max pool
    input_real = _retrieve_elements_from_indices(input_real, indices)
    input_imag = _retrieve_elements_from_indices(input_imag, indices)
    if return_indices:
        return torch.cat((input_real,input_imag),dim=1), indices
    else:
        return torch.cat((input_real,input_imag),dim=1)

class ComplexMaxPool2d(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False,
    ):
        super(ComplexMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices

    def forward(self, inp):
        return complex_max_pool2d(
            inp,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=self.return_indices,
        )

class ComplexAveragePool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, ceil_mode=False, padding=0):
        super(ComplexAveragePool2d, self).__init__()
        self.real_pool = nn.AvgPool2d(kernel_size, stride, ceil_mode=ceil_mode, padding=padding)
        self.imag_pool = nn.AvgPool2d(kernel_size, stride, ceil_mode=ceil_mode, padding=padding)

    def forward(self, x):
        real = x[:,:x.size(1)//2,...]
        imag = x[:,x.size(1)//2:,...]
        real = self.real_pool(real)
        imag = self.imag_pool(imag)
        return torch.concat((real, imag), 1)

if __name__ == '__main__':
    torch.random.manual_seed(0)
    import torch,time
    import torch.nn as nn
    import torch.nn.functional as F

    # pool = AdaptiveMaxPool2dComplex((3,3))
    # input = torch.randn(20, 16, 50, 32,dtype=torch.complex64).cuda()
    # s=time.perf_counter()
    # output = pool(input)
    # e = time.perf_counter()
    # print(time.perf_counter()-s)
    # print(output.size())
    input = torch.randn(1, 2, 9, 9,requires_grad=True)#.cuda()#.requires_grad_(True)
    print('input:',input)
    p = ComplexMaxPool2d(4, stride=1, padding=2)
    output = p(input)
    print('output:',output)
    grad_output_real = torch.ones_like(output)
    output.backward(grad_output_real)
    # output.backward(grad_output_real,retain_graph=True)
    print(input.grad)
    pass
    
    