import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialFeatureExtractor(nn.Module):
    """
    Spatial feature extraction module for I/Q channel processing - Lightweight version
    """
    def __init__(self, input_channels=2, output_channels=32):  # 从64减少到32
        super(SpatialFeatureExtractor, self).__init__()
        
        # 减少中间层通道数
        self.spatial_conv1 = nn.Conv1d(input_channels, 16, kernel_size=3, padding=1)  # 从32减少到16
        self.spatial_bn1 = nn.BatchNorm1d(16)
        
        self.spatial_conv2 = nn.Conv1d(16, output_channels, kernel_size=3, padding=1)  # 从32->64减少到16->32
        self.spatial_bn2 = nn.BatchNorm1d(output_channels)
        
        self.spatial_drop = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # x: (batch_size, input_channels, seq_len)
        x = F.relu(self.spatial_bn1(self.spatial_conv1(x)))
        x = self.spatial_drop(x)
        
        x = F.relu(self.spatial_bn2(self.spatial_conv2(x)))
        x = self.spatial_drop(x)
        
        return x


class TemporalFeatureExtractor(nn.Module):
    """
    Temporal feature extraction module using multi-scale convolutions - Lightweight version
    """
    def __init__(self, input_channels=32, output_channels=64):  # 从64->128减少到32->64
        super(TemporalFeatureExtractor, self).__init__()
        
        # 减少多尺度分支，只保留2个
        self.temp_conv_3 = nn.Conv1d(input_channels, output_channels//2, kernel_size=3, padding=1)  # 从//4改为//2
        self.temp_conv_5 = nn.Conv1d(input_channels, output_channels//2, kernel_size=5, padding=2)
        # 删除7和9的卷积分支
        
        self.temp_bn = nn.BatchNorm1d(output_channels)
        self.temp_drop = nn.Dropout(p=0.3)
        
        # Temporal pooling
        self.temp_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # 只使用两个尺度的卷积
        out_3 = self.temp_conv_3(x)
        out_5 = self.temp_conv_5(x)
        
        # Concatenate multi-scale features
        x = torch.cat([out_3, out_5], dim=1)  # 只连接两个分支
        x = F.relu(self.temp_bn(x))
        x = self.temp_drop(x)
        
        # Temporal pooling
        x = self.temp_pool(x)
        
        return x


class ChannelAttentionModule(nn.Module):
    """
    Channel attention module for multi-channel feature fusion - Lightweight version
    """
    def __init__(self, channels, reduction=8):  # 从16减少到8，增加压缩比例
        super(ChannelAttentionModule, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 删除max pooling，只保留avg pooling
        
        # 简化的MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 4)),  # 最少4个神经元
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 4), channels)
        )
        
    def forward(self, x):
        b, c, _ = x.size()
        
        # 只使用平均池化
        avg_out = self.avg_pool(x).view(b, c)
        
        # MLP processing
        avg_out = self.mlp(avg_out)
        
        # Attention weights
        attention = F.sigmoid(avg_out).unsqueeze(2)
        
        return x * attention


class SpatioTemporalBlock(nn.Module):
    """
    Spatiotemporal processing block combining spatial and temporal features - Lightweight version
    """
    def __init__(self, input_channels, output_channels):
        super(SpatioTemporalBlock, self).__init__()
        
        # 简化为单一卷积，去掉分支结构
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(output_channels)
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        # 简化的单层卷积
        x = F.relu(self.bn(self.conv(x)))
        x = self.dropout(x)
        
        return x


class DLCNN(nn.Module):
    """
    Deep Learning CNN with Spatiotemporal Multi-Channel Learning Framework
    for Automatic Modulation Recognition
    
    Architecture:
    1. Spatial feature extraction for I/Q channels
    2. Temporal feature extraction with multi-scale convolutions
    3. Spatiotemporal blocks for feature fusion
    4. Channel attention for multi-channel learning
    5. Deep classification layers
    """
    def __init__(self, input_shape, modulation_num=15):
        super(DLCNN, self).__init__()
        
        # Input shape: [batch_size, seq_len, input_dim] (e.g., [batch, 1024, 2])
        self.input_shape = input_shape
        
        # 简化的特征提取 - 大幅减少参数
        self.spatial_extractor = SpatialFeatureExtractor(input_channels=input_shape[1], 
                                                        output_channels=32)
        
        # 简化的时序特征提取
        self.temporal_extractor = TemporalFeatureExtractor(input_channels=32, 
                                                          output_channels=64)
        
        # 单个时空块，减少复杂度
        self.st_block = SpatioTemporalBlock(64, 32)
        
        # 简化的注意力机制
        self.channel_attn = ChannelAttentionModule(32)
        
        # 单个池化层
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全局池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # 简化的分类器 - 减少层数和参数
        self.clf_dense_1 = nn.Linear(32, 64)
        self.clf_bn_1 = nn.BatchNorm1d(64)
        self.clf_drop_1 = nn.Dropout(p=0.5)
        
        self.clf_dense_2 = nn.Linear(64, modulation_num)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 简化的特征提取流程
        x = self.spatial_extractor(x)      # 输出: (batch, 32, seq_len)
        x = self.temporal_extractor(x)     # 输出: (batch, 64, seq_len)
        
        # 单个时空块 + 注意力
        x = self.st_block(x)               # 输出: (batch, 32, seq_len)
        x = self.channel_attn(x)           # 注意力增强
        x = self.maxpool(x)                # 下采样
        
        # 全局平均池化
        x = self.global_avg_pool(x).squeeze(2)  # (batch_size, 32)
        
        # 简化的分类器
        x = F.relu(self.clf_dense_1(x))
        if batch_size > 1:
            x = self.clf_bn_1(x)
        x = self.clf_drop_1(x)
        
        x = self.clf_dense_2(x)
        
        return x


if __name__ == '__main__':
    import time
    from thop import profile

    # Test with different input sizes
    print("Testing DLCNN model for Spatiotemporal Multi-Channel Learning...")
    
    # Test case 1: Single sample
    input_tensor = torch.randn(1, 1024, 2).to("cuda")
    model = DLCNN(input_shape=[1, 2, 1024], modulation_num=11).to("cuda")
    
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
