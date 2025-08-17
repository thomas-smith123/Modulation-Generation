#
# @Date: 2024-10-24 17:27:07
# @LastEditors: jiangrd3 thomas-smith@live.cn
# @LastEditTime: 2025-08-15 17:53:15
# @FilePath: /complex_gru_single_radioml/model/modelv0_7.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
try:
    from model.complexPart.poolingComplex import *
    from model.complexPart.complexBatchnorm2d import ComplexBatchNorm2d
    from model.complexPart.upsampleComplex import ComplexUpsample
    from model.complexPart.complexGRUcell import ConvGRU, CustomGRUModel
    from model.complexPart.activationModule import complex_sigmoid2
    from model.complexPart.activationComplex import ComplexReLu, ComplexSiLu, ModReLU, ZReLU, ComplexLeakyReLU
    from model.complexPart.linearComplex import LinearComplex
    from model.complexPart.convComplex import Conv2dComplex
    from model.complexPart.complexBatchnorm1d import ComplexBatchNorm1d
except:
    from complexPart.poolingComplex import *
    from complexPart.complexBatchnorm2d import ComplexBatchNorm2d
    from complexPart.upsampleComplex import ComplexUpsample
    from complexPart.complexGRUcell import ConvGRU, CustomGRUModel
    from complexPart.activationModule import complex_sigmoid2
    from complexPart.activationComplex import ComplexReLu, ComplexSiLu, ModReLU, ZReLU, ComplexLeakyReLU
    from complexPart.linearComplex import LinearComplex
    from complexPart.convComplex import Conv2dComplex
    from complexPart.complexBatchnorm1d import ComplexBatchNorm1d
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

class ComplexFlatten(nn.Module):
    def __init__(self,start_dim=1, end_dim=-1) -> None:
        super(ComplexFlatten, self).__init__()
        self.real_flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.imag_flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
    def forward(self,x):
        real = x[:,:x.size(1)//2,...]
        imag = x[:,x.size(1)//2:,...]
        real = self.real_flatten(real)
        imag = self.imag_flatten(imag)
        return torch.concat((real,imag),0)

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,padding) -> None:
        super(Conv, self).__init__()
        self.conv = Conv2dComplex(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn = ComplexBatchNorm2d(out_channels*2)
        self.activation = ComplexSiLu()
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

class ResStruct(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ResStruct, self).__init__()
        self.conv1 = Conv(in_channels,out_channels,kernel_size=(1,3),stride=(1,1),padding='same')
        self.conv2 = Conv(out_channels,out_channels,kernel_size=(1,3),stride=(1,1),padding='same')
        self.conv3 = Conv(in_channels,out_channels,kernel_size=(1,1),stride=(1,1),padding='same')
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x)+x2
        return x3

class resnet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(resnet,self).__init__()
        self.r1 = ResStruct(in_channels,(in_channels+out_channels)//2)
        self.r2 = ResStruct((in_channels+out_channels)//2,out_channels)
        self.mp1 = Conv2dComplex(out_channels,out_channels,kernel_size=(1,3),stride=(1,2))
        self.rp3 = ResStruct(in_channels,out_channels)
    def forward(self,x):
        x1 = self.r1(x)
        x1 = self.r2(x1)
        x = self.rp3(x)+x1
        x = self.mp1(x)
        return x

class ComplexConcat(nn.Module):
    def __init__(self) -> None:
        super(ComplexConcat, self).__init__()
    def forward(self,x1,x2):
        x1_real = x1[:x1.size(0)//2,...]
        x1_imag = x1[x1.size(0)//2:,...]
        x2_real = x2[:x2.size(0)//2,...]
        x2_imag = x2[x2.size(0)//2:,...]
        return torch.concat((torch.concat((x1_real,x2_real),0),torch.concat((x1_imag,x2_imag),0)),1)

class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Net, self).__init__(*args, **kwargs)
        self.r1 = resnet(1,32)
        self.r2 = resnet(32,32)
        self.r3 = resnet(32,32)
        self.r4 = resnet(32,4)
        # self.mp = Conv2dComplex(1,32,kernel_size=(1,4),stride=(1,4))
        
        self.flatten = ComplexFlatten()
        self.gru = CustomGRUModel(1024,128,2,128)
        self.classify = nn.Sequential(
            nn.Linear(380,64),
            nn.BatchNorm1d(64),
            # nn.LeakyReLU(),
            nn.SiLU(),
            # nn.Linear(256,128),
            # nn.BatchNorm1d(128),
            # nn.SiLU(),
            nn.Linear(64,15),
            # nn.BatchNorm1d(15),
            # nn.LeakyReLU(),
            # nn.SiLU(),
            )
        # self.concat = ComplexConcat()
        self.softmax = nn.Softmax()
        # self.bn = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(p=0.3)  # 50% 的概率丢弃神经元
    def forward(self,x):
        """N,C,2,1024"""
        x1 = self.r1(x)
        x1 = self.r2(x1)
        x1 = self.dropout(x1)
        # x1 = self.mp(x)+x1
        x1 = self.r3(x1)
        x1 = self.r4(x1)
        x1 = self.flatten(x1)
        
        x = self.flatten(x)
        x = x.unsqueeze(1)
        x = self.gru(x)
        x = torch.concat((x,x1),1)
        x = (x[:x.size(0)//2,...]+x[x.size(0)//2:,...])/1.414213562373095
        x = self.dropout(x)
        x = self.classify(x)
        
        # x = self.bn(x)
        # x = self.softmax(x)
        return x,None,None

if __name__ == "__main__":
    model = Net()
    print(model)
    data = torch.randn(4,2,1,1024,dtype=torch.float32)
    data = model(data)
    print(data[0].shape)
    # model = Net()
    # x = torch.randn(2,4,2,1024)
    # y = model(x)
    # print(y.shape)
