'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2024-09-08 17:50:19
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2024-11-04 17:12:37
Description: CDSCNN主训练文件 - 基于main_CSGNet.py风格

Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
'''
#
# @Date: 2024-09-08 17:50:19
# @LastEditors: jiangrd3 thomas-smith@live.cn
# @LastEditTime: 2025-08-22 Updated
# @FilePath: /complex_gru_single_radioml/main_CDSCNN.py
# @Author: thomas-smith123 thomas-smith@live.cn
# @Description: 
# @
# @Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
#
from model_compare.CDSCNN import CDSCNN as Net
import time
from torch.utils.tensorboard import SummaryWriter
import argparse
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler, Dataset
import torch,os
import torch.distributed as dist
from dataset import RandomShiftDataset
import random
import numpy as np
from utils import applyColorMap, NormMinandMax
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
PATH = './runs'

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: 权重因子，用于平衡正负样本（默认1.0）
        :param gamma: 调整焦点的强度（默认2.0）
        :param reduction: 损失的聚合方式 ('none', 'mean', 'sum')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        :param inputs: 模型输出的logits，形状为[N, C]，其中N为样本数，C为类别数
        :param targets: 真实标签，形状为[N]，每个样本的标签为0到C-1的整数
        """
        # 计算每个样本的类别概率
        BCE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)  # 计算每个样本的预测概率

        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        # 根据设定的聚合方式返回损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     # 关闭确定性以获得更好性能
     torch.backends.cudnn.deterministic = False
     # 启用cudnn基准测试以优化性能
     torch.backends.cudnn.benchmark = True
     # 启用TensorFloat-32以提升A100等新GPU性能
     torch.backends.cuda.matmul.allow_tf32 = True
     torch.backends.cudnn.allow_tf32 = True
     
     # 启用额外的CUDA优化
     if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
         torch.backends.cuda.enable_flash_sdp(True)
     if hasattr(torch.backends.cuda, 'enable_math_sdp'):
         torch.backends.cuda.enable_math_sdp(True)
     if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
         torch.backends.cuda.enable_mem_efficient_sdp(True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'  # 更改为主机的 IP 地址
    os.environ['MASTER_PORT'] = '12345'  # 确保此端口没有被占用
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()    

class OptimizedGPUDataset(Dataset):
    def __init__(self, data, labels, device):
        # 预先转换数据类型，避免在训练循环中重复转换
        # CDSCNN需要的输入格式: [batch_size, 2, seq_len]
        self.data = data.to(device, non_blocking=True, dtype=torch.float32, memory_format=torch.contiguous_format)
        self.labels = labels.to(device, non_blocking=True)
        # 预计算one-hot编码，避免训练时重复计算 - 数据集有15个调制类型
        self.labels_onehot = torch.nn.functional.one_hot(self.labels[:,0].to(torch.long), num_classes=15).to(torch.float)
        
        print(f"Dataset loaded to GPU {device}: {self.data.shape}, memory: {self.data.element_size() * self.data.nelement() / 1024**3:.2f}GB")
        
    def __len__(self):
        return self.data.shape[0]
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.labels_onehot[idx]

def split_for_rank(data, labels, world_size, rank):
    # 均匀分片，最后一份多余部分也分给最后一个rank
    total = data.shape[0]
    per = total // world_size
    start = rank * per
    end = (rank + 1) * per if rank < world_size - 1 else total
    return data[start:end], labels[start:end]

def train(rank, world_size, num_gpus, num_epochs, continue_train=False):
    print(rank, world_size, num_gpus, rank % num_gpus)
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
    # 初始化最佳准确度跟踪和起始epoch
    best_accuracy = 0.0
    start_epoch = 0
    
    if rank==0:
        # current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        tb_writer = SummaryWriter(os.path.join(PATH, current_time+'CDSCNN'))
        tb_dir = os.path.join(PATH, current_time+'CDSCNN')
        
        # 保存重要文件到tensorboard文件夹
        import shutil
        
        # 保存主要的Python文件
        try:
            shutil.copy2('main_CDSCNN.py', os.path.join(tb_dir, 'main_CDSCNN.py'))
            print(f"Saved main_CDSCNN.py to {tb_dir}")
        except Exception as e:
            print(f"Failed to save main_CDSCNN.py: {e}")
            
        # 保存网络模型文件
        try:
            shutil.copy2('model_compare/CDSCNN.py', os.path.join(tb_dir, 'CDSCNN.py'))
            shutil.copy2('model_compare/CDSC.py', os.path.join(tb_dir, 'CDSC.py'))
            shutil.copy2('model_compare/CDSC_utils.py', os.path.join(tb_dir, 'CDSC_utils.py'))
            print(f"Saved CDSCNN.py and related files to {tb_dir}")
        except Exception as e:
            print(f"Failed to save CDSCNN.py: {e}")
            
        # 保存其他重要文件
        try:
            shutil.copy2('utils.py', os.path.join(tb_dir, 'utils.py'))
            shutil.copy2('dataset.py', os.path.join(tb_dir, 'dataset.py'))
            print(f"Saved utils.py and dataset.py to {tb_dir}")
        except Exception as e:
            print(f"Failed to save additional files: {e}")
    else:
        tb_writer = None
        tb_dir = None
        
    gpu_id = rank % num_gpus
    setup(rank, world_size)
    torch.cuda.set_device(gpu_id)  # 将进程绑定到指定 GPU

    # CDSCNN模型实例化，输入格式: [batch_size, 2, seq_len]
    model = Net(num_classes=15)  # 15个调制类型
    
    # 继续训练功能
    if continue_train:
        checkpoint_files = [f for f in os.listdir(PATH) if f.startswith('checkpoint_') and f.endswith('.pth')]
        if checkpoint_files:
            # 找到最新的检查点
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(PATH, latest_checkpoint)
            
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}', weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_accuracy = checkpoint['best_accuracy']
            mm = checkpoint['mm']
            
            print(f"Resumed training from epoch {start_epoch}, best accuracy: {best_accuracy:.4f}")
        else:
            print("No checkpoint found, starting from scratch")
            mm = 0
    else:
        mm = 0
        
    model = model.to(gpu_id)


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据预处理：将实部和虚部concat
        # data shape: [batch_size, 2, 1024] -> [batch_size, 2*1, 1024]
        data = data.view(data.size(0), 2, -1)  # [batch_size, 2, 1024]
        # 处理target维度
        if target.dim() > 1:
            target = target.squeeze()  # 移除多余维度
        data, target = data.to(device), target.to(device).long()  # target需要是long类型
        
        optimizer.zero_grad()
        output = model(data)
        # output shape: [batch_size, num_classes, feature_size, 1]
        output = output.squeeze()  # 移除多余的维度
        if output.dim() == 1:
            output = output.unsqueeze(0)
        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}, '
                  f'Acc: {100.*correct/total:.2f}%')
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            # 数据预处理：将实部和虚部concat
            data = data.view(data.size(0), 2, -1)  # [batch_size, 2, 1024]
            # 处理target维度
            if target.dim() > 1:
                target = target.squeeze()  # 移除多余维度
            data, target = data.to(device), target.to(device).long()  # target需要是long类型
            
            output = model(data)
            output = output.squeeze()
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据集
    print("Loading dataset...")
    train_dataset, _, valid_dataset, ___, test_dataset, __ = torch.load("dataset/RadioML_dataset.pt", map_location='cpu', weights_only=False)
    
    # 合并训练集和测试集数据
    all_data = torch.concat((train_dataset.tensors[0], test_dataset.tensors[0]), 0)
    all_labels = torch.concat((train_dataset.tensors[1], test_dataset.tensors[1]), 0)
    
    # 创建完整数据集
    full_dataset = TensorDataset(all_data, all_labels)
    
    # 数据集分割
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    
    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 创建模型
    num_classes = 15  # RadioML2016.10b 有15个调制类型
    model = CDSCNN(num_classes=num_classes).to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 训练参数
    num_epochs = 50
    best_acc = 0.0
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # 测试
        test_loss, test_acc = test_model(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        print(f'Epoch [{epoch+1}/{num_epochs}] ({epoch_time:.2f}s)')
        print(f'Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.2f}%')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print('-' * 60)
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, 'runs/best_cdscnn_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, f'runs/cdscnn_checkpoint_epoch_{epoch+1}.pth')
    
    print(f'Training completed! Best test accuracy: {best_acc:.2f}%')


if __name__ == "__main__":
    main()
