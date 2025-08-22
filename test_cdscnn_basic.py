import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_compare.CDSCNN import CDSCNN as Net


def test_cdscnn_basic():
    """基本的CDSCNN模型测试"""
    print("Testing CDSCNN basic functionality...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 直接加载数据集
    train_dataset, _, valid_dataset, ___, test_dataset, __ = torch.load("dataset/RadioML_dataset.pt", map_location='cpu', weights_only=False)
    
    # 创建一个小的测试数据加载器
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 获取一个批次的数据
    data, target = next(iter(test_loader))
    print(f"Original data shape: {data.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target values: {target}")
    
    # 数据预处理：将实部和虚部concat 从 (batch, seq_len, features) 转换为 (batch, features, seq_len)
    if data.shape[-1] == 2:  # (batch, seq_len, 2)
        data = data.permute(0, 2, 1)  # (batch, 2, seq_len)
    print(f"Processed data shape: {data.shape}")
    
    # 创建模型
    model = Net(num_classes=15).to(device)
    model.eval()
    
    # 测试前向传播
    data = data.to(device)
    with torch.no_grad():
        try:
            output = model(data)
            print(f"Model output shape: {output.shape}")
            
            # 处理输出
            if output.dim() > 2:
                output = output.squeeze()
            print(f"Final output shape: {output.shape}")
            
            # 测试预测
            pred = output.argmax(dim=1)
            print(f"Predictions: {pred}")
            print(f"Predictions shape: {pred.shape}")
            
            print("✓ Model forward pass successful!")
            
        except Exception as e:
            print(f"✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
    
    # 打印模型参数信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Info:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试复数数据的chunk操作
    print(f"\nTesting complex data processing:")
    print(f"Input data shape: {data.shape}")
    real, imag = torch.chunk(data, 2, 1)  # complex_axis=1
    print(f"Real part shape: {real.shape}")
    print(f"Imaginary part shape: {imag.shape}")
    print(f"Real part sample: {real[0, 0, :5]}")
    print(f"Imaginary part sample: {imag[0, 0, :5]}")
    
    print("\n=== CDSCNN Basic Test Completed ===")


if __name__ == "__main__":
    test_cdscnn_basic()
