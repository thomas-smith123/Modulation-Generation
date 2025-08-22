import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary
from thop import profile
import time
import os


class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.max_pool(x)

        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet_CNN(nn.Module):
    def __init__(self, in_channels, hid_dim, z_dim):
        super(ProtoNet_CNN, self).__init__()

        self.in_channels = in_channels
        self.hid_dim = hid_dim
        self.z_dim = z_dim
        self.conv_block_first = Conv_block(in_channels, hid_dim)
        self.conv_blocks = nn.ModuleList([Conv_block(hid_dim, hid_dim) for _ in range(6)])
        self.conv_block_last = Conv_block(hid_dim, z_dim)
        self.flatten = Flatten()

    def forward(self, x):
        x = self.conv_block_first(x)
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        x = self.conv_block_last(x)
        x = self.flatten(x)

        return x


class ProtoNet(nn.Module):
    """
    ProtoNet for Automatic Modulation Classification
    
    Complete ProtoNet model with classification head for RadioML dataset.
    Input format: (batch_size, seq_len, features) -> (batch_size, 1, features, seq_len)
    """
    def __init__(self, input_shape=[1024, 2], modulation_num=15):
        super(ProtoNet, self).__init__()
        self.input_shape = input_shape
        self.modulation_num = modulation_num
        
        # Feature extractor (original ProtoNet_CNN)
        self.feature_extractor = ProtoNet_CNN(in_channels=1, hid_dim=32, z_dim=24)
        
        # Calculate the output size of feature extractor
        # For input (1, 1, 2, 1024), after 8 conv blocks with maxpool
        # Size reduces by factor of 2^8 = 256 for each dimension
        # So (2, 1024) -> (2//256, 1024//256) = (1, 4) for feature maps
        # With 24 channels: 24 * 1 * 4 = 96 features
        feature_size = self._get_feature_size()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, modulation_num)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_feature_size(self):
        """Calculate the output size of feature extractor"""
        with torch.no_grad():
            # Create dummy input to calculate feature size
            dummy_input = torch.randn(1, 1, 2, 1024)
            features = self.feature_extractor(dummy_input)
            return features.shape[1]
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass of ProtoNet
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features) for I/Q data
            
        Returns:
            output: Classification logits of shape (batch_size, modulation_num)
        """
        batch_size = x.size(0)
        
        # Reshape input from (batch_size, seq_len, features) to (batch_size, 1, features, seq_len)
        # ProtoNet expects 4D input: (batch, channels, height, width)
        if x.dim() == 3:  # (batch_size, seq_len, features)
            x = x.permute(0, 2, 1)  # (batch_size, features, seq_len)
            x = x.unsqueeze(1)      # (batch_size, 1, features, seq_len)
        
        # Extract features
        features = self.feature_extractor(x)
        
        # Classification
        output = self.classifier(features)
        
        return output


def create_protonet_model(weights=None, input_shape=[1024, 2], modulation_num=15, **kwargs):
    """Create ProtoNet model factory function for API compatibility"""
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    model = ProtoNet(input_shape=input_shape, modulation_num=modulation_num)
    
    # Load weights
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    
    return model


if __name__ == '__main__':
    print("Testing ProtoNet models...")
    
    # Test original ProtoNet_CNN
    print("\n1. Testing original ProtoNet_CNN:")
    model_cnn = ProtoNet_CNN(1, 32, 24).to("cuda")
    input_cnn = torch.randn(1, 1, 2, 1024).cuda()
    
    start_time = time.time()
    outputs_cnn = model_cnn(input_cnn)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"ProtoNet_CNN output shape: {outputs_cnn.shape}")
    print(f"Elapsed time: {elapsed_time:.3f}s")
    
    # Calculate parameters and FLOPs for CNN
    macs, params = profile(model_cnn, inputs=(input_cnn,))
    print(f"ProtoNet_CNN - Param: {params / (1000 ** 2):.2f}M | FLOPs: {macs / (1000 ** 3):.3f}G")
    
    # Test complete ProtoNet model
    print("\n2. Testing complete ProtoNet for classification:")
    model = ProtoNet(input_shape=[1024, 2], modulation_num=15).to("cuda")
    
    # Test with RadioML format input
    input_radioml = torch.randn(8, 1024, 2).cuda()  # Batch of 8 samples
    
    start_time = time.time()
    outputs = model(input_radioml)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"ProtoNet input shape: {input_radioml.shape}")
    print(f"ProtoNet output shape: {outputs.shape}")
    print(f"Elapsed time: {elapsed_time:.3f}s")
    
    # Calculate parameters and FLOPs for complete model
    test_input = torch.randn(1, 1024, 2).cuda()
    macs, params = profile(model, inputs=(test_input,))
    print(f"ProtoNet - Param: {params / (1000 ** 2):.2f}M | FLOPs: {macs / (1000 ** 3):.3f}G")
    
    # Test factory function
    print("\n3. Testing factory function:")
    factory_model = create_protonet_model(input_shape=[1024, 2], modulation_num=11)
    factory_output = factory_model(torch.randn(4, 1024, 2))
    print(f"Factory model output shape: {factory_output.shape}")
    
    # Model size estimation
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # 32-bit float
    total_size_megabytes = total_size_bytes / (1024 * 1024)
    
    print(f"\nModel summary:")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_size_megabytes:.2f} MB")

