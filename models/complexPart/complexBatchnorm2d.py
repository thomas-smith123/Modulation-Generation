#
# @Date: 2024-06-17 16:35:24
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2025-08-30 16:16:05
# @FilePath: \MG-orphan\models\complexPart\complexBatchnorm2d.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch
from torch.autograd import Function
import torch.nn as nn
from typing import Optional
import os
# from utils.plot_sth import plot_distribution

DEBUG = os.getenv('DEBUG', 'False')

__all__ = ['ComplexBatchNorm2d']

class _ComplexBatchNorm(nn.Module):
    running_mean: Optional[torch.Tensor]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features//2, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features//2, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, dtype=torch.float32)
            )
            self.register_buffer("running_covar", torch.zeros(num_features//2, 3))
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_covar", None)
            self.register_parameter("num_batches_tracked", None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar.zero_()
            self.running_covar[:, 0] = 1.4142135623730951
            self.running_covar[:, 1] = 1.4142135623730951
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            nn.init.constant_(self.weight[:, :2], 1.4142135623730951)
            nn.init.zeros_(self.weight[:, 2])
            nn.init.zeros_(self.bias)
            

class ComplexBatchNorm2d(_ComplexBatchNorm):
    """
    使用Schur补计算协方差矩阵的逆
    """
    def forward(self, inp):
    
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            # mean does not support automatic differentiation for outputs with complex dtype.
            mean_r = input_real.mean([0, 2, 3]).type(torch.float32)
            mean_i = input_imag.mean([0, 2, 3]).type(torch.float32)
            # mean = mean_r + 1j * mean_i
            mean = torch.concat((mean_r, mean_i),0)
        else:
            mean = self.running_mean

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        # inp = inp - mean[None, :, None, None]
        inp = inp - mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1)
            Crr = 1.0 / n * inp[:,:inp.size(1)//2,...].pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cii = 1.0 / n * inp[:,inp.size(1)//2:,...].pow(2).sum(dim=[0, 2, 3]) + self.eps
            Cri = (inp[:,:inp.size(1)//2,...].mul(inp[:,inp.size(1)//2:,...])).mean(dim=[0, 2, 3])
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]  # +self.eps

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)  #
                    + (1 - exponential_average_factor) * self.running_covar[:, 0]
                )

                self.running_covar[:, 1] = (
                    exponential_average_factor * Cii * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 1]
                )

                self.running_covar[:, 2] = (
                    exponential_average_factor * Cri * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_covar[:, 2]
                )

        # calculate the inverse square root the covariance matrix
        det = Crr * Cii - Cri.pow(2)
        s = torch.sqrt(det)
        t = torch.sqrt(Cii + Crr + 2 * s)
        inverse_st = 1.0 / (s * t)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        inp = torch.concat(((
            Rrr[None, :, None, None] * inp[:,:inp.size(1)//2,...] + Rri[None, :, None, None] * inp[:,inp.size(1)//2:,...]
        ).type(inp.dtype), (
            Rii[None, :, None, None] * inp[:,inp.size(1)//2:,...] + Rri[None, :, None, None] * inp[:,:inp.size(1)//2,...]
        ).type(
            inp.dtype
        )),1)

        if self.affine:
            inp = torch.concat(((
                self.weight[None, :, 0, None, None] * inp[:,:inp.size(1)//2,...]
                + self.weight[None, :, 2, None, None] * inp[:,inp.size(1)//2:,...]
                + self.bias[None, :, 0, None, None]
            ).type(inp.dtype), (
                self.weight[None, :, 2, None, None] * inp[:,:inp.size(1)//2,...]
                + self.weight[None, :, 1, None, None] * inp[:,inp.size(1)//2:,...]
                + self.bias[None, :, 1, None, None]
            ).type(
                inp.dtype
            )),1)
        return inp


if __name__ == '__main__':
    os.environ['Train_status'] = 'True'
    os.environ['OnlyReal'] = 'True'
    os.environ['Origin'] = 'True'
    # Example usage
    batch_size = 4
    num_features = 8
    height = width = 16
    input = torch.randn(batch_size, num_features, height, width)  # Complex input tensor with separate real and imaginary parts
    input[:,input.size(1)//2:,...] = torch.zeros_like(input[:,input.size(1)//2:,...])  # Random imaginary part
    
    complex_bn = ComplexBatchNorm2d(num_features)
    
    # torch.nn.functional.batch_norm(input[:,:input.size(1)//2,...], running_mean=None, running_var=None, weight=complex_bn.weight_r[:,0], bias=complex_bn.bias_r[:,0], training=True, momentum=0.1, eps=1e-5)
    
    output = complex_bn(input)
    grad_output = torch.randn_like(output)  # Random gradient for testing
    output.backward(grad_output)

