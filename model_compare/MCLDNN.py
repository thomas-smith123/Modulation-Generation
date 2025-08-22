import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class MCLDNN(nn.Module):
    """
    PyTorch implementation of MCLDNN (Multi-Channel LSTM Deep Neural Network)
    
    Original Keras model converted to PyTorch
    
    Args:
        input_shape1: Shape for I/Q channel input [2, 1024]
        input_shape2: Shape for I and Q channel inputs [1024, 1] 
        classes: Number of output classes (default: 11)
        dropout_rate: Dropout rate (default: 0.5)
    """
    
    def __init__(self, input_shape1=[2, 1024], input_shape2=[1024, 1], classes=11, dropout_rate=0.5):
        super(MCLDNN, self).__init__()
        
        self.input_shape1 = input_shape1  # [2, 1024] for I/Q channel
        self.input_shape2 = input_shape2  # [1024, 1] for I and Q channels separately
        self.classes = classes
        self.dropout_rate = dropout_rate
        
        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping Section
        
        # Conv2D for I/Q channel input [batch, 1, 2, 1024] -> [batch, 50, 2, 1024]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=50, kernel_size=(2, 8), 
                              padding='same', bias=True)
        
        # Conv1D for I channel [batch, 1, 1024] -> [batch, 50, 1024]
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8, 
                              padding=7, bias=True)  # causal padding
        
        # Conv1D for Q channel [batch, 1, 1024] -> [batch, 50, 1024]
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=50, kernel_size=8, 
                              padding=7, bias=True)  # causal padding
        
        # Conv2D after concatenation [batch, 50, 2, 1024] -> [batch, 50, 2, 1024]
        self.conv4 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=(1, 8), 
                              padding='same', bias=True)
        
        # Conv2D after final concatenation [batch, 100, 2, 1024] -> [batch, 100, 1, 1020]
        self.conv5 = nn.Conv2d(in_channels=100, out_channels=100, kernel_size=(2, 5), 
                              padding=0, bias=True)
        
        # Part-B: Temporal Characteristics Extraction Section
        
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=100, hidden_size=128, batch_first=True, bias=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True, bias=True)
        
        # DNN (Fully Connected layers)
        self.fc1 = nn.Linear(128, 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Glorot uniform initialization (similar to Keras)"""
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, input1, input2, input3):
        """
        Forward pass
        
        Args:
            input1: I/Q channel input [batch, 2, 1024]
            input2: I channel input [batch, 1024]  
            input3: Q channel input [batch, 1024]
        
        Returns:
            output: Classification probabilities [batch, classes]
        """
        
        # Prepare inputs
        # input1: [batch, 2, 1024] -> [batch, 1, 2, 1024]
        x1 = input1.unsqueeze(1)
        
        # input2, input3: [batch, 1024] -> [batch, 1, 1024]
        x2 = input2.unsqueeze(1)
        x3 = input3.unsqueeze(1)
        
        # Part-A: Multi-channel Inputs and Spatial Characteristics Mapping
        
        # Conv2D for I/Q channel
        x1 = F.relu(self.conv1(x1))  # [batch, 50, 2, 1024]
        
        # Conv1D for I and Q channels
        x2 = F.relu(self.conv2(x2))  # [batch, 50, 1024+7] 
        x2 = x2[:, :, :1024]  # Causal padding - keep only first 1024 timesteps
        x2 = x2.unsqueeze(2)  # [batch, 50, 1, 1024]
        
        x3 = F.relu(self.conv3(x3))  # [batch, 50, 1024+7]
        x3 = x3[:, :, :1024]  # Causal padding - keep only first 1024 timesteps  
        x3 = x3.unsqueeze(2)  # [batch, 50, 1, 1024]
        
        # Concatenate x2 and x3 along spatial dimension
        x_concat1 = torch.cat([x2, x3], dim=2)  # [batch, 50, 2, 1024]
        
        # Conv2D after concatenation
        x_concat1 = F.relu(self.conv4(x_concat1))  # [batch, 50, 2, 1024]
        
        # Concatenate with x1
        x = torch.cat([x1, x_concat1], dim=1)  # [batch, 100, 2, 1024]
        
        # Final Conv2D
        x = F.relu(self.conv5(x))  # [batch, 100, 1, 1020]
        
        # Part-B: Temporal Characteristics Extraction
        
        # Reshape for LSTM: [batch, 100, 1, 1020] -> [batch, 1020, 100]
        x = x.squeeze(2).transpose(1, 2)  # [batch, 1020, 100]
        
        # LSTM layers
        x, _ = self.lstm1(x)  # [batch, 1020, 128]
        x, (hidden, _) = self.lstm2(x)  # [batch, 1020, 128], take only final hidden state
        x = hidden[-1]  # [batch, 128] - last hidden state
        
        # DNN (Fully Connected layers)
        x = F.selu(self.fc1(x))  # [batch, 128]
        x = self.dropout1(x)
        x = F.selu(self.fc2(x))  # [batch, 128]
        x = self.dropout2(x)
        x = self.fc3(x)  # [batch, classes]
        
        # Softmax for classification
        x = F.softmax(x, dim=1)
        
        return x

def create_mcldnn_model(classes=11, input_shape1=[2, 1024], input_shape2=[1024, 1], 
                        weights=None, **kwargs):
    """
    Create MCLDNN model (compatible interface with original Keras version)
    
    Args:
        classes: Number of output classes
        input_shape1: Shape for I/Q channel input
        input_shape2: Shape for I and Q channel inputs
        weights: Path to weights file (optional)
        
    Returns:
        model: MCLDNN PyTorch model
    """
    
    if weights is not None and not os.path.exists(weights):
        raise ValueError('The `weights` argument should be either '
                        '`None` (random initialization), '
                        'or the path to the weights file to be loaded.')
    
    model = MCLDNN(input_shape1=input_shape1, 
                   input_shape2=input_shape2, 
                   classes=classes)
    
    # Load weights if provided
    if weights is not None:
        model.load_state_dict(torch.load(weights))
    
    return model

def prepare_mcldnn_inputs(data):
    """
    Convert RadioML dataset format to MCLDNN input format
    
    Args:
        data: Input data of shape [batch, 1024, 2] where last dim is [I, Q]
        
    Returns:
        tuple: (input1, input2, input3) for MCLDNN model
            - input1: I/Q channel [batch, 2, 1024]  
            - input2: I channel [batch, 1024]
            - input3: Q channel [batch, 1024]
    """
    # data shape: [batch, 1024, 2]
    # Extract I and Q channels
    I_channel = data[:, :, 0]  # [batch, 1024]
    Q_channel = data[:, :, 1]  # [batch, 1024]
    
    # Create I/Q combined input by transposing and stacking
    IQ_combined = data.transpose(1, 2)  # [batch, 2, 1024]
    
    return IQ_combined, I_channel, Q_channel

if __name__ == '__main__':
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # For RadioML2016.10a dataset (11 classes)
    model = create_mcldnn_model(classes=11)
    model = model.to(device)
    
    # For RadioML2016.10b dataset (10 classes)
    # model = create_mcldnn_model(classes=10)
    
    print("MCLDNN Model Summary:")
    print(f"Device: {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 32
    input1 = torch.randn(batch_size, 2, 1024).to(device)  # I/Q channel
    input2 = torch.randn(batch_size, 1024).to(device)     # I channel
    input3 = torch.randn(batch_size, 1024).to(device)     # Q channel
    
    model.eval()
    with torch.no_grad():
        output = model(input1, input2, input3)
    
    print(f"Input shapes: {input1.shape}, {input2.shape}, {input3.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")