"""
MCNet PyTorch Implementation

This is a PyTorch implementation of the MCNet architecture originally designed in MATLAB.

Key differences from the original MATLAB version:
1. Uses 1D convolutions instead of 2D for signal processing compatibility
2. Simplifies the complex branching structure while preserving multi-scale feature extraction
3. Replaces complex mix blocks with Inception-style multi-scale blocks
4. Adds residual connections for better gradient flow
5. Includes attention mechanism for feature refinement

Architecture mapping:
- MATLAB's imageInputLayer -> PyTorch's initial conv1d
- MATLAB's complex mix blocks -> PyTorch's MultiScaleBlock (Inception-style)
- MATLAB's depthConcatenationLayer -> PyTorch's torch.cat
- MATLAB's additionLayer -> PyTorch's residual connections
- MATLAB's global pooling -> PyTorch's AdaptiveAvgPool1d/AdaptiveMaxPool1d
- MATLAB's classification layers -> PyTorch's fully connected classifier

The network maintains the multi-scale feature extraction philosophy of the original
while being more suitable for PyTorch and signal processing tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleBlock(nn.Module):
    """
    Multi-scale convolution block inspired by MATLAB MCNet
    Uses 1D convolutions suitable for signal processing
    """
    def __init__(self, in_channels, out_channels):
        super(MultiScaleBlock, self).__init__()
        
        # Branch 1: 1x1 conv (equivalent to 1 in 1D)
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: small scale conv
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: medium scale conv
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels//4, out_channels//4, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Branch 4: max pool + conv
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels//4, kernel_size=1),
            nn.BatchNorm1d(out_channels//4),
            nn.ReLU(inplace=True)
        )
        
        # Output projection
        self.output_conv = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.output_bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        # Concatenate all branches
        concat = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        
        # Output projection
        output = F.relu(self.output_bn(self.output_conv(concat)))
        
        return output


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out


class MCNet(nn.Module):
    """
    MCNet: Multi-scale CNN for Automatic Modulation Classification
    PyTorch implementation using 1D convolutions for signal processing
    """
    def __init__(self, input_shape, modulation_num=24):
        super(MCNet, self).__init__()
        
        # Initial convolution
        self.init_conv = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3)
        self.init_bn = nn.BatchNorm1d(64)
        self.init_relu = nn.ReLU(inplace=True)
        self.init_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Multi-scale blocks (representing complex mix blocks from MATLAB)
        self.ms_block1 = MultiScaleBlock(64, 128)
        self.res_block1 = ResidualBlock(128)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.ms_block2 = MultiScaleBlock(128, 256)
        self.res_block2 = ResidualBlock(256)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.ms_block3 = MultiScaleBlock(256, 512)
        self.res_block3 = ResidualBlock(512)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.ms_block4 = MultiScaleBlock(512, 512)
        self.res_block4 = ResidualBlock(512)
        
        # Global feature aggregation
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(1024, 512),  # 512 + 512 from avg and max pooling
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, modulation_num)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: [batch_size, seq_len, input_dim] -> [batch_size, input_dim, seq_len]
        if len(x.shape) == 3:
            x = x.transpose(1, 2)  # [B, 2, 1024]
        
        # Initial feature extraction
        x = self.init_relu(self.init_bn(self.init_conv(x)))
        x = self.init_pool(x)
        
        # Multi-scale blocks with residual connections
        x = self.ms_block1(x)
        x = self.res_block1(x)
        x = self.pool1(x)
        
        x = self.ms_block2(x)
        x = self.res_block2(x)
        x = self.pool2(x)
        
        x = self.ms_block3(x)
        x = self.res_block3(x)
        x = self.pool3(x)
        
        x = self.ms_block4(x)
        x = self.res_block4(x)
        
        # Global pooling
        avg_pool = self.global_avg_pool(x).view(x.size(0), -1)
        max_pool = self.global_max_pool(x).view(x.size(0), -1)
        
        # Concatenate pooled features
        features = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply attention
        attention_weights = self.attention(features)
        features = features * attention_weights
        
        # Classification
        output = self.classifier(features)
        
        return output

if __name__ == '__main__':
    import time
    
    print("Testing MCNet model for Automatic Modulation Classification...")
    
    # Test case 1: Single sample
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = MCNet(input_shape=[1024, 2], modulation_num=11).to(device)
    
    # Test with batch
    input_tensor = torch.randn(4, 1024, 2).to(device)
    print(f"Input shape: {input_tensor.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test timing
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    print(f"Average inference time: {avg_time:.4f} seconds per batch")
    print(f"Average time per sample: {avg_time/input_tensor.shape[0]:.6f} seconds")
