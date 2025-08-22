import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise Separable Convolution for efficient feature extraction
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  padding=padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class MultiHeadAttention1D(nn.Module):
    """
    Multi-head attention mechanism for 1D signals
    """
    def __init__(self, d_model, num_heads=8):
        super(MultiHeadAttention1D, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights
    
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, _ = self.scaled_dot_product_attention(q, k, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model)
        
        # Final linear transformation
        output = self.w_o(attn_output)
        return output


class ICAMC(nn.Module):
    """
    Lightweight Improved CNN-based Automatic Modulation Classification
    
    Simplified architecture to reduce overfitting:
    1. Reduced channel dimensions significantly
    2. Simplified attention mechanism  
    3. Fewer layers and parameters
    4. Efficient depthwise separable convolutions
    """
    def __init__(self, input_shape, modulation_num=15):
        super(ICAMC, self).__init__()
        self.input_shape = input_shape
        input_dim = input_shape[0]  # Usually 2 for I/Q data
        num_classes = modulation_num
        
        # Initial convolution to expand channels - 增加初始通道数
        self.init_conv = nn.Conv1d(input_dim, 64, kernel_size=7, padding=3)  # 32->64
        self.init_bn = nn.BatchNorm1d(64)
        self.init_drop = nn.Dropout(0.1)
        
        # Depthwise separable convolutions with more channels
        self.dw_conv1 = DepthwiseSeparableConv1d(64, 128, kernel_size=5, padding=2)  # 64->128
        self.maxpool1 = nn.MaxPool1d(2)
        
        self.dw_conv2 = DepthwiseSeparableConv1d(128, 256, kernel_size=3, padding=1)  # 128->256
        self.maxpool2 = nn.MaxPool1d(2)
        
        self.dw_conv3 = DepthwiseSeparableConv1d(256, 384, kernel_size=3, padding=1)  # 192->384
        self.maxpool3 = nn.MaxPool1d(2)
        
        # 添加第四层depthwise conv增加容量
        self.dw_conv4 = DepthwiseSeparableConv1d(384, 512, kernel_size=3, padding=1)
        self.maxpool4 = nn.MaxPool1d(2)
        
        # Multi-head attention with more complexity
        d_model = 512  # 192->512
        self.attention = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)  # 6->8 heads
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Global pooling (both avg and max)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Enhanced multi-layer classifier
        self.clf_dense_1 = nn.Linear(1024, 256)  # 384->1024 (512*2), 128->256
        self.clf_bn_1 = nn.BatchNorm1d(256)
        self.clf_drop_1 = nn.Dropout(0.3)
        
        self.clf_dense_2 = nn.Linear(256, 128)  # 增加这一层的容量
        self.clf_bn_2 = nn.BatchNorm1d(128)
        self.clf_drop_2 = nn.Dropout(0.2)
        
        self.clf_dense_3 = nn.Linear(128, 64)  # 64->128->64，增加中间层
        self.clf_bn_3 = nn.BatchNorm1d(64)
        self.clf_drop_3 = nn.Dropout(0.1)
        
        self.clf_dense_4 = nn.Linear(64, num_classes)  # 最终输出层
        
        # Initialize weights
        self._initialize_weights()
    

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: (batch_size, seq_len, input_dim) = (batch_size, 1024, 2)
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch_size, input_dim, seq_len) = (batch_size, 2, 1024)
        x = x.transpose(1, 2)
        
        # Initial convolution
        x = F.relu(self.init_bn(self.init_conv(x)))
        x = self.init_drop(x)
        
        # Depthwise separable convolution blocks
        x = self.dw_conv1(x)
        x = self.maxpool1(x)
        
        x = self.dw_conv2(x)
        x = self.maxpool2(x)
        
        x = self.dw_conv3(x)
        x = self.maxpool3(x)
        
        # 新增的第四层depthwise conv
        x = self.dw_conv4(x)
        x = self.maxpool4(x)
        
        # Transpose for attention: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(x, x, x)  # Self-attention: query=key=value=x
        x = self.attn_norm(x + attn_output)
        
        # Transpose back for pooling: (batch_size, features, seq_len)
        x = x.transpose(1, 2)
        
        # Global pooling (combine avg and max pooling)
        avg_pool = self.global_avg_pool(x).squeeze(2)  # (batch_size, 512)
        max_pool = self.global_max_pool(x).squeeze(2)  # (batch_size, 512)
        x = torch.cat([avg_pool, max_pool], dim=1)     # (batch_size, 1024)
        
        # Enhanced multi-layer classification
        x = F.relu(self.clf_dense_1(x))
        if batch_size > 1:
            x = self.clf_bn_1(x)
        x = self.clf_drop_1(x)
        
        x = F.relu(self.clf_dense_2(x))
        if batch_size > 1:
            x = self.clf_bn_2(x)
        x = self.clf_drop_2(x)
        
        x = F.relu(self.clf_dense_3(x))
        if batch_size > 1:
            x = self.clf_bn_3(x)
        x = self.clf_drop_3(x)
        
        x = self.clf_dense_4(x)
        
        return x
def create_icamc_model(weights=None, input_shape=[1, 2, 1024], modulation_num=15, **kwargs):
    """Create ICAMC model factory function for API compatibility"""
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    model = ICAMC(input_shape=input_shape, modulation_num=modulation_num)
    
    # Load weights
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    
    return model


if __name__ == '__main__':
    import time
    from thop import profile

    # Test with different input sizes
    print("Testing ICAMC model for Beyond 5G Communications...")
    
    # Test case 1: Single sample
    input_tensor = torch.randn(1, 1024, 2).to("cuda")
    model = ICAMC(input_shape=[1, 2, 1024], modulation_num=11).to("cuda")
    
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