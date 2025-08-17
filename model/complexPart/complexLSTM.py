#
# @Date: 2024-08-27 21:23:07
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2024-08-28 10:06:30
# @FilePath: /hy_bak_test_delete_after_used/yolo_hy_complex_network_for_complex_input_with_gru/models/complexPart/complexLSTM.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
import torch
import torch.nn as nn
import torch.optim as optim
from models.complexPart.linearComplex import LinearComplex
from models.complexPart.activationComplex import ComplexSigmoid2, tanh_modified
# from models.complexPart.activationModule import complex_sigmoid
from models.complexPart.convComplex import Conv2dComplex

def normalize(num,type = 'abs'):
	# numpy里complex可改为numpy.complex128
    if type == 'abs': ##FIXME 不确定在这里好不好使
        max = torch.max(torch.abs(num))
        return num/max/torch.sqrt(2)
    elif type == 'real_imag':
        real = num.real
        imag = num.imag
        real_min = torch.min(real)
        imag_min = torch.min(imag)
        num = num - (real_min+1j*imag_min)
        real = num.real
        imag = num.imag
        real_max = torch.max(real)
        imag_max = torch.max(imag)
        return real/real_max+1j*imag/imag_max

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # 输入门
        self.W_i = LinearComplex(input_size + hidden_size, hidden_size)
        # 遗忘门
        self.W_f = LinearComplex(input_size + hidden_size, hidden_size)
        # 输出门
        self.W_o = LinearComplex(input_size + hidden_size, hidden_size)
        # 候选记忆
        self.W_c = LinearComplex(input_size + hidden_size, hidden_size)
        self.sigmoid = ComplexSigmoid2()
        self.activation = tanh_modified() ##FIXME 要换的

    def forward(self, x, hidden):
        h_prev, c_prev = hidden  # 上一个时间步的隐藏状态和细胞状态
        tmp_r = torch.concat((x[:, :x.size(1) // 2, ...], h_prev[:, :h_prev.size(1) // 2, ...]),1)
        tmp_i = torch.concat((x[:, x.size(1) // 2:, ...], h_prev[:, h_prev.size(1) // 2:, ...]),1)
        combined = torch.concat((tmp_r, tmp_i),1)
        # 拼接输入和上一时间步的隐藏状态
        # combined = torch.cat([x, h_prev], dim=1)

        # 计算门和候选状态
        i = self.sigmoid(self.W_i(combined))  # 输入门
        f = self.sigmoid(self.W_f(combined))  # 遗忘门
        o = self.sigmoid(self.W_o(combined))  # 输出门
        c_tilde = self.activation(self.W_c(combined))  # 候选记忆

        # 更新细胞状态
        # c_new = f * c_prev + i * c_tilde
        f_r = x[:, :x.size(1) // 2, ...]
        f_i = x[:, x.size(1) // 2:, ...]
        c_prev_r = c_prev[:, :c_prev.size(1) // 2, ...]
        c_prev_i = c_prev[:, c_prev.size(1) // 2:, ...]
        i_r = i[:, :i.size(1) // 2, ...]
        i_i = i[:, i.size(1) // 2:, ...]
        c_tilde_r = c_tilde[:, :c_tilde.size(1) // 2, ...]
        c_tilde_i = c_tilde[:, c_tilde.size(1) // 2:, ...]
        c_new_r = f_r*c_prev_r - f_i*c_prev_i + i_r*c_tilde_r - i_i*c_tilde_i
        c_new_i = f_r*c_prev_i + f_i*c_prev_r + i_r*c_tilde_i + i_i*c_tilde_r
        c_new = torch.concat((c_new_r, c_new_i),1)
        
        # 更新隐藏状态
        o_r = o[:, :o.size(1) // 2, ...]
        o_i = o[:, o.size(1) // 2:, ...]
        tmp = self.activation(c_new)
        tmp_r = tmp[:, :tmp.size(1) // 2, ...]
        tmp_i = tmp[:, tmp.size(1) // 2:, ...]
        h_new = torch.concat((o_r*tmp_r - o_i*tmp_i, o_r*tmp_i + o_i*tmp_r),1)
        # h_new = o * self.activation(c_new)

        return h_new, c_new

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # 创建多个LSTM层
        self.layers = nn.ModuleList([LSTMCell(input_size if i == 0 else hidden_size, hidden_size) 
                                     for i in range(num_layers)])

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            batch_size = x.size(0)
            hidden_states = [(torch.zeros(batch_size, layer.hidden_size, device=x.device),
                              torch.zeros(batch_size, layer.hidden_size, device=x.device))
                             for layer in self.layers]

        # 对每一层逐步处理
        for i, layer in enumerate(self.layers):
            h, c = hidden_states[i]
            h_new, c_new = layer(x, (h, c))
            x = h_new  # 下一层的输入为当前层的输出
            hidden_states[i] = (h_new, c_new)

        return x, hidden_states

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_channels = hidden_channels
        
        padding = kernel_size // 2  # 保持输入和输出的空间大小一致
        
        # 输入门、遗忘门、输出门、候选细胞状态，分别使用卷积层代替线性层
        self.conv_i = Conv2dComplex(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_f = Conv2dComplex(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_o = Conv2dComplex(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.conv_c = Conv2dComplex(input_channels + hidden_channels, hidden_channels, kernel_size, padding=padding)
        self.sigmoid = ComplexSigmoid2()
        self.activation = nn.Tanh()

    def forward(self, x, hidden):
        h_prev, c_prev = hidden  # 上一个时间步的隐藏状态和细胞状态
        tmp_r = torch.concat((x[:, :x.size(1) // 2, ...], h_prev[:, :h_prev.size(1) // 2, ...]),1)
        tmp_i = torch.concat((x[:, x.size(1) // 2:, ...], h_prev[:, h_prev.size(1) // 2:, ...]),1)
        combined = torch.concat((tmp_r, tmp_i),1)
        # 拼接输入和隐藏状态，沿通道维度堆叠
        # combined = torch.cat([x, h_prev], dim=1)
        
        # 计算门和候选状态（使用卷积操作）
        i = self.sigmoid(self.conv_i(combined))  # 输入门
        f = self.sigmoid(self.conv_f(combined))  # 遗忘门
        o = self.sigmoid(self.conv_o(combined))  # 输出门
        c_tilde = self.activation(self.conv_c(combined))  # 候选记忆

        # 更新细胞状态
        c_new = f * c_prev + i * c_tilde
        f_r = x[:, :x.size(1) // 2, ...]
        f_i = x[:, x.size(1) // 2:, ...]
        c_prev_r = c_prev[:, :c_prev.size(1) // 2, ...]
        c_prev_i = c_prev[:, c_prev.size(1) // 2:, ...]
        i_r = i[:, :i.size(1) // 2, ...]
        i_i = i[:, i.size(1) // 2:, ...]
        c_tilde_r = c_tilde[:, :c_tilde.size(1) // 2, ...]
        c_tilde_i = c_tilde[:, c_tilde.size(1) // 2:, ...]
        c_new_r = f_r*c_prev_r - f_i*c_prev_i + i_r*c_tilde_r - i_i*c_tilde_i
        c_new_i = f_r*c_prev_i + f_i*c_prev_r + i_r*c_tilde_i + i_i*c_tilde_r
        c_new = torch.concat((c_new_r, c_new_i),1)
        
        # 更新隐藏状态
        o_r = o[:, :o.size(1) // 2, ...]
        o_i = o[:, o.size(1) // 2:, ...]
        tmp = self.activation(c_new)
        tmp_r = tmp[:, :tmp.size(1) // 2, ...]
        tmp_i = tmp[:, tmp.size(1) // 2:, ...]
        h_new = torch.concat((o_r*tmp_r - o_i*tmp_i, o_r*tmp_i + o_i*tmp_r),1)
        # h_new = o * torch.tanh(c_new)

        return h_new, c_new

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # 创建多个ConvLSTM层
        self.layers = nn.ModuleList([ConvLSTMCell(input_channels if i == 0 else hidden_channels, hidden_channels, kernel_size) 
                                     for i in range(num_layers)])

    def forward(self, x, hidden_states=None):
        if hidden_states is None:
            batch_size, _, height, width = x.size()
            hidden_states = [(torch.zeros(batch_size, layer.hidden_channels, height, width, device=x.device),
                              torch.zeros(batch_size, layer.hidden_channels, height, width, device=x.device))
                             for layer in self.layers]

        # 对每一层逐步处理
        for i, layer in enumerate(self.layers):
            h, c = hidden_states[i]
            h_new, c_new = layer(x, (h, c))
            x = h_new  # 下一层的输入为当前层的输出
            hidden_states[i] = (h_new, c_new)

        return x, hidden_states

# 示例用法
if __name__ == "__main__":
    # 初始化一个两层的ConvLSTM
    conv_lstm = ConvLSTM(input_channels=3, hidden_channels=16, kernel_size=3, num_layers=2)
    
    # 输入示例 (batch_size=4, channels=3, height=64, width=64)
    x = torch.randn(4, 3, 64, 64)
    
    # 前向传播
    output, hidden = conv_lstm(x)
    
    print("Output shape:", output.shape)
    print("Hidden state shape:", hidden[0][0].shape)



# 示例用法
if __name__ == "__main__":
    # 初始化一个两层的LSTM
    lstm = LSTM(input_size=10, hidden_size=20, num_layers=2)
    
    # 输入示例 (batch_size=4, input_size=10)
    x = torch.randn(4, 10)
    
    # 前向传播
    output, hidden = lstm(x)
    
    print("Output shape:", output.shape)
    print("Hidden state shape:", hidden[0][0].shape)
