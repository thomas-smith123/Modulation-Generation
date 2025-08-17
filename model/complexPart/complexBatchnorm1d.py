#
# @Date: 2024-08-09 20:38:50
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-10-22 20:13:05
# @FilePath: /hy_bak_test_delete_after_used/complex_gru_single_radioml/model/complexPart/complexBatchnorm1d.py
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
    running_covar: Optional[torch.Tensor]

    def __init__(
        self,
        num_features,
        eps=1e-4,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_ComplexBatchNorm, self).__init__()
        num_features = num_features//2
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features, 3))
            self.bias = nn.Parameter(torch.Tensor(num_features, 2))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if self.track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(2, num_features, dtype=torch.float32)
            )
            self.register_buffer("running_covar", torch.zeros(num_features, 3))
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
    """
    使用Schur补计算协方差矩阵的逆
    """
    def forward(self, inp):
        input_real = inp[:inp.size(0) // 2, :]
        input_imag = inp[inp.size(0) // 2:, :]
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.track_running_stats):
            # calculate mean of real and imaginary part
            mean_r = input_real.mean([0]).type(torch.float32)
            mean_i = input_imag.mean([0]).type(torch.float32)
        else:
            mean_r = self.running_mean[0]
            mean_i = self.running_mean[1]

        if self.training and self.track_running_stats:
            # update running mean
            with torch.no_grad():
                self.running_mean[0] = (
                    exponential_average_factor * mean_r
                    + (1 - exponential_average_factor) * self.running_mean[0]
                )
                self.running_mean[1] = (
                    exponential_average_factor * mean_i
                    + (1 - exponential_average_factor) * self.running_mean[1]
                )

        input_real = input_real - mean_r
        input_imag = input_imag - mean_i

        if self.training or (not self.track_running_stats):
            # Elements of the covariance matrix (biased for train)
            n = input_real.numel() / input_real.size(1)
            Crr = 1.0 / n * input_real.pow(2).sum(dim=[0]) + self.eps
            Cii = 1.0 / n * input_imag.pow(2).sum(dim=[0]) + self.eps
            Cri = (input_real.mul(input_imag)).mean(dim=[0]) + self.eps
        else:
            Crr = self.running_covar[:, 0] + self.eps
            Cii = self.running_covar[:, 1] + self.eps
            Cri = self.running_covar[:, 2] + self.eps

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
        s = torch.sqrt(det + self.eps)
        t = torch.sqrt(Cii + Crr + 2 * s + self.eps)
        inverse_st = 1.0 / (s * t + self.eps)
        Rrr = (Cii + s) * inverse_st
        Rii = (Crr + s) * inverse_st
        Rri = -Cri * inverse_st

        input_real = Rrr[None, :] * input_real + Rri[None, :] * input_imag
        input_imag = Rii[None, :] * input_imag + Rri[None, :] * input_real

        inp = torch.cat((input_real, input_imag), 0)
        
        if self.affine:
            input_real = (
                self.weight[None, :, 0] * input_real
                + self.weight[None, :, 2] * input_imag
                + self.bias[None, :, 0]
            )
            input_imag = (
                self.weight[None, :, 2] * input_real
                + self.weight[None, :, 1] * input_imag
                + self.bias[None, :, 1]
            )
            inp = torch.cat((input_real, input_imag), 0)
        return inp


if __name__ == '__main__':
    # 设置环境变量
    os.environ['Train_status'] = 'True'
    os.environ['OnlyReal'] = 'True'
    os.environ['Origin'] = 'True'
    
    # 示例使用
    batch_size = 20
    num_features = 8  # 复数特征数，实部和虚部各占一半
    input = torch.randn(batch_size, num_features)  # 复数输入张量，实部和虚部分别占据 num_features 的一半
    input[:, input.size(1) // 2:] = torch.zeros_like(input[:, input.size(1) // 2:])  # 随机虚部

    complex_bn = ComplexBatchNorm1d(num_features)
    
    # 前向传播
    output = complex_bn(input)
    print("Output after batch normalization:")
    print(output)
    
    # 反向传播测试
    grad_output = torch.randn_like(output)  # 随机梯度用于测试
    output.backward(grad_output)
    print("Backward pass completed successfully.")