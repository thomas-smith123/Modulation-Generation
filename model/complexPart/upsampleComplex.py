#
# @Date: 2024-06-17 16:45:54
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-09-27 14:39:27
# @FilePath: /hy_bak_test_delete_after_used/complex_gru_single/model/complexPart/upsampleComplex.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch,os
import torch.nn as nn
import torch.nn.functional as F
try:
    from model.complexPart.stableAtan2 import StableAtan2
except:
    from complexPart.stableAtan2 import StableAtan2

__all__ = ['ComplexUpsample']

def _phase_unwrap(phase):
    phase_diff = torch.diff(phase, dim=-1)
    phase_diff_unwrapped = phase_diff - 2 * torch.pi * torch.round(phase_diff / (2 * torch.pi))
    phase_unwrapped = torch.cat((phase[..., :1], phase[..., :1] + torch.cumsum(phase_diff_unwrapped, dim=-1)), dim=-1)
    return phase_unwrapped

def complex_upsample(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the real and imaginary part and recombining
    """
    input_real = inp[:,:inp.size(1)//2,...]
    input_imag = inp[:,inp.size(1)//2:,...]
    outp_real = F.interpolate(
        input_real,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    outp_imag = F.interpolate(
        input_imag,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return torch.concat((outp_real,outp_imag),dim=1)

def complex_upsample2(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the amplitude and phase part and recombining
    """
    input_real = inp[:,:inp.size(1)//2,...]
    input_imag = inp[:,inp.size(1)//2:,...]
    input_abs = torch.sqrt(input_real**2+input_imag**2+1e-4)
    outp_abs = F.interpolate(
        input_abs,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    angle = torch.atan2(input_imag, input_real)
    outp_angle = F.interpolate(
        angle,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_abs * torch.concat((
        torch.cos(outp_angle).type(torch.complex64),
        torch.sin(outp_angle).type(torch.complex64)
    ), dim=1)

def complex_upsample3(
    inp,
    size=None,
    scale_factor=None,
    mode="nearest",
    align_corners=None,
    recompute_scale_factor=None,
):
    """
    Performs upsampling by separately interpolating the amplitude and phase part and recombining
    """
    input_real = inp[:,:inp.size(1)//2,...]
    input_imag = inp[:,inp.size(1)//2:,...]
    outp_abs = F.interpolate(
        inp.abs(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )
    angle = torch.atan2(inp.imag, inp.real)
    outp_angle = F.interpolate(
        angle,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
    )

    return outp_abs * (
        torch.cos(outp_angle).type(torch.complex64)
        + 1j * torch.sin(outp_angle).type(torch.complex64)
    )

class ComplexUpsample_(nn.Module):
    """
    PhaseUnwrapUpsample
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super(ComplexUpsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        
        amplitudes = torch.sqrt(input_real**2 + input_imag**2)
        phases = torch.atan2(input_imag, input_real)
        
        phases_unwrapped = _phase_unwrap(phases)

        upsampled_amplitudes = F.interpolate(amplitudes, size=self.size, scale_factor=self.scale_factor,
                                             mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)
        
        upsampled_phases_unwrapped = F.interpolate(phases_unwrapped, size=self.size, scale_factor=self.scale_factor,
                                                   mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)

        return torch.concat((upsampled_amplitudes*torch.cos(upsampled_phases_unwrapped), upsampled_amplitudes*torch.sin(upsampled_phases_unwrapped)), dim=1)
    
class ComplexUpsample(nn.Module):
    """
    PolarUpsample
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super(ComplexUpsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor

    def forward(self, inp):
        input_real = inp[:, :inp.size(1) // 2, ...]
        input_imag = inp[:, inp.size(1) // 2:, ...]
        
        amplitudes = torch.sqrt(input_real**2 + input_imag**2 + 1e-4)
        phases = StableAtan2.apply(input_imag, input_real)

        phases_unwrapped = _phase_unwrap(phases)

        upsampled_amplitudes = F.interpolate(amplitudes, size=self.size, scale_factor=self.scale_factor,
                                             mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)
        
        upsampled_phases_unwrapped = F.interpolate(phases_unwrapped, size=self.size, scale_factor=self.scale_factor,
                                                   mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)
        
        return torch.concat((upsampled_amplitudes*torch.cos(upsampled_phases_unwrapped), upsampled_amplitudes*torch.sin(upsampled_phases_unwrapped)), dim=1)

class ComplexUpsample_(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
        super(ComplexUpsample_, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
    
    def forward(self, inp):
        input_real = inp[:,:inp.size(1)//2,...]
        input_imag = inp[:,inp.size(1)//2:,...]
        outp_abs = F.interpolate(
            input_real,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )
        outp_angle = F.interpolate(
            input_imag,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            recompute_scale_factor=self.recompute_scale_factor,
        )

        return torch.concat((outp_abs, outp_angle),1)
# 测试复数上采样层
if __name__ == "__main__":
    ## 这玩意好像没有什么办法验证
    # 输入数据
    input_tensor_real = torch.randn(1, 3, 16, 16)  # batch size 1, 3 channels, 16x16
    input_tensor_imag = torch.randn(1, 3, 16, 16)  # batch size 1, 3 channels, 16x16
    input_tensor = torch.complex(input_tensor_real, input_tensor_imag)
    scale_factor = 2
    input_tensor = torch.cat([input_tensor_real, input_tensor_imag], dim=1)
    print(input_tensor.shape) 
    # 使用自定义的复数上采样函数
    complex_upsample = ComplexUpsample(scale_factor)
    output = complex_upsample(input_tensor)
    print("Output using ComplexUpsample:")
    print(output.shape)  # 应为 (1, 3, 32, 32)

    import numpy as np
    import plotly.graph_objects as go
    import tensorboard as tb
    from io import BytesIO
    import PIL.Image
    # 定义复数函数
    def complex_function(z):
        return z**2  # 示例：复数的平方

    # 生成输入复数的网格
    re_range = np.linspace(-2, 2, 100)
    im_range = np.linspace(-2, 2, 100)
    re, im = np.meshgrid(re_range, im_range)
    z_in = re + 1j * im

    # 计算输出复数
    z_out = complex_function(z_in)

    # 分别获取输出复数的模和相角
    magnitude = np.abs(z_out)
    phase = np.angle(z_out)

    # 创建绘图数据
    fig = go.Figure(data=[go.Surface(x=re, y=im, z=magnitude, surfacecolor=phase, colorscale='Viridis')])

    # 设置图形的布局
    fig.update_layout(
        title='Complex to Complex Function Visualization',
        scene=dict(
            xaxis_title='Real Part (Input)',
            yaxis_title='Imaginary Part (Input)',
            zaxis_title='Magnitude (Output)'
        ),
        coloraxis_colorbar=dict(
            title='Phase'
        )
    )

    # 显示图形
    img_bytes = fig.to_image(format="png")
    img = PIL.Image.open(BytesIO(img_bytes))
    img.save("complex_function.png")