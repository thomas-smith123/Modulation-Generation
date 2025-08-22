import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PETCGDNN(nn.Module):
    """
    PETCGDNN: Phase Enhancement and Temporal Convolutional GRU Deep Neural Network
    
    PyTorch implementation of the PETCGDNN model for automatic modulation classification.
    This model uses phase enhancement and combines 2D CNN with GRU for feature extraction.
    """
    
    def __init__(self, input_shape=[1024, 2], classes=24):
        super(PETCGDNN, self).__init__()
        self.input_shape = input_shape
        self.classes = classes
        
        # Phase enhancement parameters
        self.phase_fc = nn.Linear(1024 * 2, 1)  # For phase calculation
        
        # Spatial feature extraction (2D CNN)
        self.conv1_1 = nn.Conv2d(1, 75, kernel_size=(8, 2), padding=0)
        self.conv1_2 = nn.Conv2d(75, 25, kernel_size=(5, 1), padding=0)
        
        # Temporal feature extraction (GRU)
        self.gru = nn.GRU(input_size=25, hidden_size=128, batch_first=True)
        
        # Classification layer
        self.classifier = nn.Linear(128, classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        nn.init.xavier_uniform_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)
    
    def l2_normalize(self, x):
        """L2 normalization function"""
        norm = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True))
        norm = torch.clamp(norm, min=1e-8)  # Avoid division by zero
        return x / norm
    
    def forward(self, x):
        """
        Forward pass of PETCGDNN
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 2) for I/Q data
            
        Returns:
            output: Classification logits of shape (batch_size, classes)
        """
        batch_size = x.size(0)
        
        # Extract I and Q components
        # x shape: (batch_size, 1024, 2)
        input_i = x[:, :, 0]  # I component: (batch_size, 1024)
        input_q = x[:, :, 1]  # Q component: (batch_size, 1024)
        
        # Phase enhancement
        # Flatten input for phase calculation
        x_flat = x.view(batch_size, -1)  # (batch_size, 1024*2)
        phase_param = self.phase_fc(x_flat)  # (batch_size, 1)
        
        # Calculate cos and sin of phase parameter
        cos_phase = torch.cos(phase_param)  # (batch_size, 1)
        sin_phase = torch.sin(phase_param)  # (batch_size, 1)
        
        # Apply phase rotation
        # y1 = I*cos(θ) + Q*sin(θ)
        # y2 = Q*cos(θ) - I*sin(θ)
        y1 = input_i * cos_phase + input_q * sin_phase  # (batch_size, 1024)
        y2 = input_q * cos_phase - input_i * sin_phase  # (batch_size, 1024)
        
        # Combine enhanced I/Q components
        enhanced_iq = torch.stack([y1, y2], dim=2)  # (batch_size, 1024, 2)
        
        # Reshape for 2D convolution: (batch_size, channels, height, width)
        # Add channel dimension and reshape to (batch_size, 1, 1024, 2)
        x_conv = enhanced_iq.unsqueeze(1)  # (batch_size, 1, 1024, 2)
        
        # Spatial feature extraction
        x_conv = F.relu(self.conv1_1(x_conv))  # (batch_size, 75, 1017, 1)
        x_conv = F.relu(self.conv1_2(x_conv))  # (batch_size, 25, 1013, 1)
        
        # Reshape for GRU: (batch_size, seq_len, features)
        x_conv = x_conv.squeeze(-1)  # Remove last dimension: (batch_size, 25, 1013)
        x_conv = x_conv.transpose(1, 2)  # (batch_size, 1013, 25)
        
        # Temporal feature extraction with GRU
        gru_out, _ = self.gru(x_conv)  # (batch_size, 1013, 128)
        
        # Use the last output of GRU
        gru_final = gru_out[:, -1, :]  # (batch_size, 128)
        
        # Classification
        output = self.classifier(gru_final)  # (batch_size, classes)
        
        # Apply softmax for probability distribution (optional, usually done in loss function)
        # output = F.softmax(output, dim=1)
        
        return output


def create_petcgdnn_model(weights=None, input_shape=[1024, 2], classes=24, **kwargs):
    """Create PETCGDNN model factory function for API compatibility"""
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), '
                         'or the path to the weights file to be loaded.')
    
    model = PETCGDNN(input_shape=input_shape, classes=classes)
    
    # Load weights
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    
    return model


if __name__ == '__main__':
    import time
    from thop import profile

    print("Testing PETCGDNN model...")
    
    # Test case 1: Single sample
    input_tensor = torch.randn(1, 1024, 2)
    model = PETCGDNN(input_shape=[1024, 2], classes=10)
    
    print(f"Model output shape: {model(input_tensor).shape}")
    print(f"Model output: {model(input_tensor)}")
    
    # Test case 2: Batch processing and timing
    input_batch = torch.randn(32, 1024, 2)
    
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
    try:
        test_input = torch.randn(1, 1024, 2)
        macs, params = profile(model, inputs=(test_input,))
        
        print(f"Parameters: {params / (1000 ** 2):.2f}M")
        print(f"FLOPs: {macs / (1000 ** 3):.3f}G")
    except Exception as e:
        print(f"Could not calculate FLOPs: {e}")
    
    # Model size estimation
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = total_params * 4  # 32-bit float
    total_size_megabytes = total_size_bytes / (1024 * 1024)
    
    print(f"Model size: {total_size_megabytes:.2f} MB")
    print(f"Total parameters: {total_params:,}")
    
    # Architecture summary
    print("\nModel Architecture Summary:")
    print(model)