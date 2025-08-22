import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleConv(nn.Module):
    """
    Multi-scale convolutional block for extracting features at different scales
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        
        # 恢复到3个分支的设计
        branch_channels = out_channels // 3
        remaining_channels = out_channels - 2 * branch_channels
        
        self.conv1 = nn.Conv1d(in_channels, branch_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels, branch_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels, remaining_channels, kernel_size=7, padding=3)
        
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=0.2)  # 降低dropout率
        
    def forward(self, x):
        # Multi-scale convolution - 3个分支
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        
        # Concatenate multi-scale features
        out = torch.cat([out1, out2, out3], dim=1)
        out = F.relu(self.bn(out))
        out = self.dropout(out)
        
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with skip connections
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(p=0.2)  # 较低的dropout
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        # Skip connection
        out += residual
        out = F.relu(out)
        
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism
    """
    def __init__(self, channels, reduction=8):  # 从16减少到8，减少中间层参数
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # 确保reduction后至少有8个通道
        reduced_channels = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)
        
    def forward(self, x):
        b, c, _ = x.size()
        
        # Average pooling and max pooling
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        
        # Shared MLP
        avg_out = F.relu(self.fc1(avg_out))
        avg_out = self.fc2(avg_out)
        
        max_out = F.relu(self.fc1(max_out))
        max_out = self.fc2(max_out)
        
        # Channel attention weights
        out = F.sigmoid(avg_out + max_out).unsqueeze(2)
        
        return x * out


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        
        # 减小卷积核大小从7到5
        self.conv = nn.Conv1d(2, 1, kernel_size=5, padding=2)
        
    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        out = torch.cat([avg_out, max_out], dim=1)
        out = F.sigmoid(self.conv(out))
        
        return x * out


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    """
    def __init__(self, channels, reduction=8):  # 增加reduction参数
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class AMCNet(nn.Module):
    """
    AMC-Net: An Effective Network for Automatic Modulation Classification
    
    Architecture:
    1. Multi-scale convolutional feature extraction
    2. Residual blocks with skip connections
    3. CBAM attention mechanism
    4. Global average pooling and classification
    """
    def __init__(self, input_shape, modulation_num=15):
        super(AMCNet, self).__init__()
        
        # Input shape: [batch_size, seq_len, input_dim] (e.g., [batch, 1024, 2])
        self.input_shape = input_shape
        
        # Initial convolution - 恢复到较大的通道数
        self.init_conv = nn.Conv1d(in_channels=input_shape[0], out_channels=64, 
                                  kernel_size=3, padding=1)  # 恢复到64通道
        self.init_bn = nn.BatchNorm1d(64)
        self.init_drop = nn.Dropout(p=0.2)  # 降低dropout以保持学习能力
        
        # Multi-scale feature extraction - 恢复到原始通道数
        self.multiscale1 = MultiScaleConv(64, 128)  # 恢复到64->128
        self.multiscale2 = MultiScaleConv(128, 256)  # 恢复到128->256
        
        # 恢复多个残差块
        self.res_block1 = ResidualBlock(256)
        self.res_block2 = ResidualBlock(256) 
        self.res_block3 = ResidualBlock(256)
        
        # CBAM attention - 恢复到原始配置
        self.cbam = CBAM(256, reduction=16)  # 恢复到256通道，reduction=16
        
        # Pooling layers
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 恢复到更强的分类器
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),  # 降低dropout
            nn.Linear(256, 128), # 恢复到256->128
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128), # 添加BatchNorm稳定训练
            nn.Dropout(p=0.2),   # 更低的dropout
            nn.Linear(128, 64),  # 添加中间层
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),   # 最低的dropout
            nn.Linear(64, modulation_num)  # 最终输出层
        )
        
    def forward(self, x):
        # Input: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.init_drop(x)
        
        # Multi-scale feature extraction
        x = self.multiscale1(x)
        x = self.maxpool1(x)
        
        x = self.multiscale2(x)
        x = self.maxpool2(x)
        
        # 使用所有残差块
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # CBAM attention
        x = self.cbam(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # 使用简化的分类器
        x = self.classifier(x)
        
        return x


if __name__ == '__main__':
    import time
    from thop import profile

    # Test with different input sizes
    print("Testing AMC-Net model...")
    
    # Test case 1: Single sample
    input_tensor = torch.randn(1, 1024, 2).to("cuda")
    model = AMCNet(input_shape=[1, 2, 1024], modulation_num=11).to("cuda")
    
    print(f"Model output shape: {model(input_tensor).shape}")
    print(f"Model output: {model(input_tensor)}")
    
    # Test case 2: Batch processing and timing
    input_batch = torch.randn(32, 1024, 2).cuda()
    
    # Warm up
    for _ in range(5):
        _ = model(input_batch)
    
    # Timing test
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_batch)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Batch size: {input_batch.shape[0]}")
    print(f"Elapsed time: {elapsed_time:.4f} seconds")
    print(f"Average time per sample: {elapsed_time/input_batch.shape[0]:.6f} seconds")
    
    # Model complexity analysis
    test_input = torch.randn(1, 1024, 2).cuda()
    macs, params = profile(model, inputs=(test_input,))
    
    print(f"Parameters: {params / (1000 ** 2):.2f}M")
    print(f"FLOPs: {macs / (1000 ** 3):.3f}G")
    
    # Model size estimation
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # 32-bit float
    total_size_megabytes = total_size_bytes / (1024 * 1024)
    
    print(f"Model size: {total_size_megabytes:.2f} MB")
    
    # Architecture summary
    print("\nModel Architecture Summary:")
    print(model)
