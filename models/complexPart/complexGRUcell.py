#
# @Date: 2024-08-27 15:33:42
# @LastEditors: thomas-smith123 thomas-smith@live.cn
# @LastEditTime: 2025-08-30 16:15:23
# @FilePath: \MG-orphan\models\complexPart\complexGRUcell.py
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

__all__ = ['CustomGRUModel','ConvGRU']

def normalize(num,type = 'abs'):
	# numpy里complex可改为numpy.complex128
    if type == 'abs': 
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

# 定义手动实现的GRU单元
class CustomGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.Wz = LinearComplex(input_size + hidden_size, hidden_size)
        self.Wr = LinearComplex(input_size + hidden_size, hidden_size)
        self.Wh = LinearComplex(input_size + hidden_size, hidden_size)
        self.sigmoid = ComplexSigmoid2()
        self.activation = tanh_modified() 

    def forward(self, x, h):
        tmp_r = torch.concat((x[:, :x.size(1) // 2, ...], h[:, :h.size(1) // 2, ...]),1)
        tmp_i = torch.concat((x[:, x.size(1) // 2:, ...], h[:, h.size(1) // 2:, ...]),1)
        combined = torch.concat((tmp_r, tmp_i),1)
        z = self.sigmoid(self.Wz(combined))
        r = self.sigmoid(self.Wr(combined))
        tmp = r*h
        tmp_r = torch.concat((x[:, :x.size(1) // 2, ...], tmp[:, :tmp.size(1) // 2, ...]),1)
        tmp_i = torch.concat((x[:, x.size(1) // 2:, ...], tmp[:, tmp.size(1) // 2:, ...]),1)
        combined_r = torch.concat((tmp_r, tmp_i),1)
        # combined_r = torch.cat((x, r * h), dim=1)
        h_tilde = self.activation(self.Wh(combined_r))
        h_next = (1 - z) * h + z * h_tilde
        return h_next

# 定义GRU模型
class CustomGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru_cells = nn.ModuleList([CustomGRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.fc = LinearComplex(hidden_size, output_size)
        self.normalize = False
    def forward(self, x):
        ## 归一化
        if self.normalize:
            x = normalize(x)
        h = [torch.zeros(x.size(0), self.hidden_size).to(x.device) for _ in range(self.num_layers)]
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            for i, gru_cell in enumerate(self.gru_cells):
                h[i] = gru_cell(x_t, h[i])
                x_t = h[i]
        out = self.fc(h[-1])
        return out

class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, padding=1):
        super(ConvGRUCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Update gate
        self.conv_z = Conv2dComplex(in_channels=(input_channels + hidden_channels),
                                out_channels=hidden_channels, 
                                kernel_size=kernel_size, 
                                padding=padding)
        
        # Reset gate
        self.conv_r = Conv2dComplex(in_channels=(input_channels + hidden_channels),
                                out_channels=hidden_channels, 
                                kernel_size=kernel_size, 
                                padding=padding)
        
        # New memory content
        self.conv_h = Conv2dComplex(in_channels=(input_channels + hidden_channels),
                                out_channels=hidden_channels, 
                                kernel_size=kernel_size, 
                                padding=padding)

        self.sigmoid = ComplexSigmoid2()
        self.activation = tanh_modified()
        
    def forward(self, x, h_prev):
        # Concatenate the input and the previous hidden state along the channel dimension
        tmp_r = torch.concat((x[:, :x.size(1) // 2, ...], h_prev[:, :h_prev.size(1) // 2, ...]),1)
        tmp_i = torch.concat((x[:, x.size(1) // 2:, ...], h_prev[:, h_prev.size(1) // 2:, ...]),1)
        combined = torch.concat((tmp_r, tmp_i),1)
        # combined = torch.cat([x, h_prev], dim=1)
        
        # Compute update gate and reset gate
        z = self.sigmoid(self.conv_z(combined))
        r = self.sigmoid(self.conv_r(combined))
        z = torch.concat((z,z),1)
        r = torch.concat((r,r),1)
        tmp = r*h_prev 
        
        tmp_r = torch.concat((x[:, :x.size(1) // 2, ...], tmp[:, :tmp.size(1) // 2, ...]),1)
        tmp_i = torch.concat((x[:, x.size(1) // 2:, ...], tmp[:, tmp.size(1) // 2:, ...]),1)
        combined_r = torch.concat((tmp_r, tmp_i),1)
        # Compute new hidden state candidate
        # combined_r = torch.cat([x, r * h_prev], dim=1)
        h_tilde = self.activation(self.conv_h(combined_r))
        
        # Compute the new hidden state
        h_new = (1 - z) * h_prev + z * h_tilde
        return h_new

class ConvGRU(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, padding=1):
        super(ConvGRU, self).__init__()
        self.num_layers = num_layers
        
        # Create a list of ConvGRU layers
        self.layers = nn.ModuleList([ConvGRUCell(input_channels if i == 0 else hidden_channels,
                                                 hidden_channels, kernel_size, padding)
                                     for i in range(num_layers)])

    def forward(self, x, hidden_states=None):
        
        if hidden_states is None:
            batch_size, _, height, width = x.size()
            hidden_states = [torch.zeros(batch_size, layer.hidden_channels*2, height, width, device=x.device) 
                             for layer in self.layers]

        # Forward propagate through each layer
        for i, layer in enumerate(self.layers):
            h = hidden_states[i]
            
            h_new = layer(x, h)
            if torch.isnan(h_new).any():
                print('tmp has nan')
                if torch.isnan(x).any():
                    print('X has nan')
                else:
                    print('X has no nan')
            x = h_new  # input to the next layer is the output of the current layer
            hidden_states[i] = h_new

        return x, hidden_states

if __name__ == '__main__':
    # 示例参数
    input_size = 10
    hidden_size = 20
    num_layers = 2
    output_size = 1

    # 创建模型
    model = CustomGRUModel(input_size, hidden_size, num_layers, output_size)

    # 打印模型结构
    print(model)
    # 生成一些示例数据
    def generate_data(seq_length, num_samples):
        X = torch.randn(num_samples, seq_length, input_size)
        y = torch.randn(num_samples, output_size)
        return X, y

    # 示例参数
    seq_length = 5
    num_samples = 100

    # 生成数据
    X, y = generate_data(seq_length, num_samples)

    # 创建模型
    model = CustomGRUModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # 测试模型
    model.eval()
    with torch.no_grad():
        test_X, test_y = generate_data(seq_length, 10)
        test_outputs = model(test_X)
        print("Test Outputs:", test_outputs)
        print("Test Targets:", test_y)