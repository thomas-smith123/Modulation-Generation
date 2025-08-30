#
# @Date: 2024-09-03 16:25:01
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-09-03 16:25:03
# @FilePath: 
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#

import torch

class StableAtan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2):
        ctx.save_for_backward(x1, x2)
        return torch.atan2(x1, x2)
    
    @staticmethod
    def backward(ctx, grad_output):
        x1, x2 = ctx.saved_tensors
        grad_x1 = grad_x2 = None

        if ctx.needs_input_grad[0]:
            grad_x1 = grad_output * (x2 / (x1 ** 2 + x2 ** 2 + 1e-4))
        if ctx.needs_input_grad[1]:
            grad_x2 = -grad_output * (x1 / (x1 ** 2 + x2 ** 2 + 1e-4))
        
        return grad_x1, grad_x2