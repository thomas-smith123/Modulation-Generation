#
# @Date: 2024-08-09 20:38:50
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-08-13 10:33:08
# @FilePath: 
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

__all__ = ['ComplexBatchNorm1d']

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
class ComplexBatchNorm1d(_ComplexBatchNorm):
    def forward(self, inp):
        
        input_real = inp[:,:inp.size(1) // 2]
        input_imag = inp[:,inp.size(1) // 2:]
        
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
            mean_r = input_real.mean(dim=0).type(inp.dtype)
            mean_i = input_imag.mean(dim=0).type(inp.dtype)
            mean = torch.concat((mean_r, mean_i),0)
        else:
            mean = self.running_mean.type(inp.dtype)

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )

        # inp = inp - mean[None, ...]
        inp = torch.concat((input_real-mean[:mean.size(0)//2], input_imag-mean[mean.size(0)//2:]),1)

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = inp.numel() / inp.size(1) /2
            Crr = inp[:,:inp.size(1)//2,...].var(dim=0, unbiased=False) + self.eps
            Cii = inp[:,inp.size(1)//2:,...].var(dim=0, unbiased=False) + self.eps
            Cri = (inp[:,:inp.size(1)//2,...].mul(inp[:,inp.size(1)//2:,...])).mean(dim=0)
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2]

        if self.training and self.track_running_stats:
            with torch.no_grad():
                self.running_covar[:, 0] = (
                    exponential_average_factor * Crr * n / (n - 1)
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

        inp = torch.concat(((Rrr[None, :] * inp[:,:inp.size(1)//2,...] + Rri[None, :] * inp[:,inp.size(1)//2:,...]).type(
            inp.dtype
        ), (Rii[None, :] * inp[:,inp.size(1)//2:,...] + Rri[None, :] * inp[:,:inp.size(1)//2,...]).type(
            inp.dtype
        )),1)

        if self.affine:
            inp = torch.concat(((
                self.weight[None, :, 0] * inp[:,:inp.size(1)//2,...]
                + self.weight[None, :, 2] * inp[:,inp.size(1)//2:,...]
                + self.bias[None, :, 0]
            ).type(inp.dtype), (
                self.weight[None, :, 2] * inp[:,:inp.size(1)//2,...]
                + self.weight[None, :, 1] * inp[:,inp.size(1)//2:,...]
                + self.bias[None, :, 1]
            ).type(
                inp.dtype
            )),1)

        del Crr, Cri, Cii, Rrr, Rii, Rri, det, s, t
        return inp