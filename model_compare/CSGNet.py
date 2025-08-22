import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    Self-attention mechanism for temporal feature aggregation
    """
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention(lstm_output), dim=1)  # (batch_size, seq_len, 1)
        attn_output = torch.sum(attn_weights * lstm_output, dim=1)  # (batch_size, hidden_size)
        return attn_output, attn_weights


class CSGNet(nn.Module):
    """
    CSGNet: A network with attention mechanism for automatic modulation recognition
    
    Architecture:
    1. Convolutional layers for local feature extraction
    2. GRU layers for temporal modeling  
    3. Attention mechanism for feature aggregation
    4. Dense layers for classification
    """
    def __init__(self, input_shape, modulation_num=15):
        super(CSGNet, self).__init__()
        
        # Input shape: [batch_size, seq_len, input_dim] (e.g., [batch, 1024, 2])
        self.input_shape = input_shape
        
        # Convolutional layers for local feature extraction
        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=64, 
                              kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv1_drop = nn.Dropout(p=0.2)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, 
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv2_drop = nn.Dropout(p=0.2)
        
        # Max pooling to reduce sequence length
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # GRU layers for temporal modeling
        self.gru1 = nn.GRU(input_size=128, hidden_size=64, 
                          num_layers=1, batch_first=True, bidirectional=True)
        self.gru1_drop = nn.Dropout(p=0.3)
        
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, 
                          num_layers=1, batch_first=True, bidirectional=True)
        self.gru2_drop = nn.Dropout(p=0.3)
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size=128)
        
        # Classification layers
        self.clf_dense_1 = nn.Linear(128, 128)
        self.clf_bn_1 = nn.BatchNorm1d(128)
        self.clf_drop_1 = nn.Dropout(p=0.4)
        
        self.clf_dense_2 = nn.Linear(128, 64)
        self.clf_bn_2 = nn.BatchNorm1d(64)
        self.clf_drop_2 = nn.Dropout(p=0.3)
        
        self.clf_dense_3 = nn.Linear(64, modulation_num)
        
    def forward(self, x):
        # Input: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # Transpose for Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv1_drop(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv2_drop(x)
        
        # Max pooling
        x = self.maxpool(x)
        
        # Transpose back for GRU: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # First GRU layer
        gru1_out, _ = self.gru1(x)
        gru1_out = self.gru1_drop(gru1_out)
        
        # Second GRU layer
        gru2_out, _ = self.gru2(gru1_out)
        gru2_out = self.gru2_drop(gru2_out)
        
        # Attention mechanism for temporal aggregation
        attn_output, attn_weights = self.attention(gru2_out)
        
        # Classification layers
        x = F.relu(self.clf_dense_1(attn_output))
        # Handle batch norm for single sample case
        if batch_size > 1:
            x = self.clf_bn_1(x)
        x = self.clf_drop_1(x)
        
        x = F.relu(self.clf_dense_2(x))
        # Handle batch norm for single sample case
        if batch_size > 1:
            x = self.clf_bn_2(x)
        x = self.clf_drop_2(x)
        
        x = self.clf_dense_3(x)
        
        return x


if __name__ == '__main__':
    import time
    from thop import profile

    # Test with different input sizes
    print("Testing CSGNet model...")
    
    # Test case 1: Single sample
    input_tensor = torch.randn(1, 1024, 2).to("cuda")
    model = CSGNet(input_shape=[1, 2, 1024], modulation_num=11).to("cuda")
    
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
