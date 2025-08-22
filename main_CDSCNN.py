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
        # 将原始数据从 (batch, seq_len, features) 转换为 (batch, features, seq_len)
        if data.dim() == 3 and data.shape[-1] == 2:
            data = data.permute(0, 2, 1)  # (batch, seq_len, 2) -> (batch, 2, seq_len)
        
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

def train_one_epoch(epoch, model, train_loader, optimizer, scheduler, gpu_id, compute_loss, accumulation_steps, train_count, tb_writer, batch_size_multiplier, scaler, world_size, initial_lr):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0
    
    for i, (inputs, targets, _) in enumerate(train_loader):
        # 处理target维度
        if targets.dim() > 1:
            targets = targets.squeeze()  # 移除多余维度
        targets = targets.long()  # 确保target是long类型
        
        # 暂时不使用混合精度，因为复数操作可能不兼容
        # with torch.cuda.amp.autocast():
        outputs = model(inputs)
        # 如果输出有多余维度，压缩
        if outputs.dim() > 2:
            outputs = outputs.squeeze()
        loss = compute_loss(outputs, targets)
        loss = loss / accumulation_steps
        
        # scaler.scale(loss).backward()
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            # scaler.step(optimizer)
            # scaler.update()
            optimizer.step()
            optimizer.zero_grad()
        
        # 统计
        running_loss += loss.item() * accumulation_steps
        _, predicted = torch.max(outputs.data, 1)
        running_total += targets.size(0)
        running_corrects += (predicted == targets).sum().item()
        
        if i % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f'Epoch {epoch+1}, Batch {i}/{train_count}, Loss: {loss.item():.4f}, Acc: {100.*running_corrects/running_total:.2f}%, LR: {current_lr:.6f}')
            
            if tb_writer and torch.distributed.get_rank() == 0:
                global_step = epoch * train_count + i
                tb_writer.add_scalar('Loss/Train_Batch', loss.item(), global_step)
                tb_writer.add_scalar('Accuracy/Train_Batch', 100.*running_corrects/running_total, global_step)
                tb_writer.add_scalar('Learning_Rate', current_lr, global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * running_corrects / running_total
    
    if tb_writer and torch.distributed.get_rank() == 0:
        tb_writer.add_scalar('Loss/Train_Epoch', epoch_loss, epoch)
        tb_writer.add_scalar('Accuracy/Train_Epoch', epoch_acc, epoch)
    
    print(f'Train Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    return epoch_loss, epoch_acc

def validate(epoch, model, valid_loader, scheduler, gpu_id, compute_loss, valid_count, tb_writer, world_size):
    """验证函数"""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    running_total = 0
    
    with torch.no_grad():
        for inputs, targets, _ in valid_loader:
            # 处理target维度
            if targets.dim() > 1:
                targets = targets.squeeze()
            targets = targets.long()
            
            # 暂时不使用混合精度，因为复数操作可能不兼容
            # with torch.cuda.amp.autocast():
            outputs = model(inputs)
            if outputs.dim() > 2:
                outputs = outputs.squeeze()
            loss = compute_loss(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_total += targets.size(0)
            running_corrects += (predicted == targets).sum().item()
    
    epoch_loss = running_loss / len(valid_loader)
    epoch_acc = 100. * running_corrects / running_total
    
    if tb_writer and torch.distributed.get_rank() == 0:
        tb_writer.add_scalar('Loss/Valid_Epoch', epoch_loss, epoch)
        tb_writer.add_scalar('Accuracy/Valid_Epoch', epoch_acc, epoch)
    
    print(f'Valid Epoch {epoch+1}: Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
    
    # 调用调度器
    scheduler.step()
    
    return epoch_acc

def train(rank, world_size, num_gpus, num_epochs, continue_train=False):
    print(rank, world_size, num_gpus, rank % num_gpus)
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
    # 初始化最佳准确度跟踪和起始epoch
    best_accuracy = 0.0
    start_epoch = 0
    
    if rank==0:
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

    # CDSCNN模型实例化
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
    
    # 检查是否在调试模式下运行，如果是则跳过编译优化
    import sys
    is_debugging = sys.gettrace() is not None or hasattr(sys, '_getframe')
    
    if not is_debugging:
        try:
            # 使用更激进的编译模式
            model = torch.compile(model, mode='max-autotune', fullgraph=True)
            print(f"Rank {rank}: Model compiled with torch.compile (max-autotune)")
        except Exception as e:
            print(f"Rank {rank}: torch.compile failed: {e}, using regular model")
    else:
        print(f"Rank {rank}: Debugging mode detected, skipping torch.compile")
    
    model = nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[gpu_id],
        output_device=gpu_id,
        find_unused_parameters=False,  # 关闭以提升性能
        broadcast_buffers=False,       # 减少通信开销
        gradient_as_bucket_view=True   # 内存优化
    )
    
    # 动态调整batch_size和学习率的策略
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Rank {rank}: Model has {model_params:,} parameters")
    
    base_batch_size = 128  # 基础batch_size，保证训练稳定性
    batch_size_multiplier = 2  # 可以根据需要调整：1, 2, 3, 4
    base_lr = 0.002
    # 根据模型大小动态调整batch_size和学习率
    if model_params < 50000:  # 小网络
        base_batch_size = 32
        batch_size_multiplier = 1
        base_lr = 0.0005
        weight_decay = 1e-3
        print(f"Rank {rank}: Small model detected, using conservative settings")
    elif model_params < 200000:  # 中等网络
        base_batch_size = 64
        batch_size_multiplier = 1
        base_lr = 0.001
        weight_decay = 5e-4
        print(f"Rank {rank}: Medium model detected")
    else:  # 大网络
        base_batch_size = 128
        batch_size_multiplier = 1
        base_lr = 0.003
        weight_decay = 1e-4
        print(f"Rank {rank}: Large model detected")
        
    effective_batch_size = base_batch_size * batch_size_multiplier
    
    # 使用FocalLoss获得更好的训练效果
    compute_loss = FocalLoss()
    
    # 根据batch_size调整学习率 - 线性缩放规则
    lr_scale_factor = batch_size_multiplier  # 线性缩放
    adjusted_lr = base_lr * lr_scale_factor
    
    # 添加混合精度训练支持
    scaler = torch.cuda.amp.GradScaler()
    
    # 添加warmup来稳定大batch_size训练
    optimizer = torch.optim.Adam(model.parameters(), lr=adjusted_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    
    # 使用更激进的学习率衰减来配合大batch_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, 6//batch_size_multiplier), gamma=0.8)
    
    # 如果是继续训练，恢复优化器和调度器状态
    if continue_train:
        checkpoint_files = [f for f in os.listdir(PATH) if f.startswith('checkpoint_') and f.endswith('.pth')]
        if checkpoint_files:
            # 找到最新的检查点
            checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            latest_checkpoint = checkpoint_files[-1]
            checkpoint_path = os.path.join(PATH, latest_checkpoint)
            
            print(f"Loading optimizer and scheduler state from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{gpu_id}', weights_only=False)
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            print(f"Optimizer and scheduler state restored from epoch {checkpoint['epoch']}")
    
    
    train_dataset,_,valid_dataset,___,test_dataset,__ = torch.load("dataset/RadioML_dataset.pt", map_location='cpu', weights_only=False)
    _ = torch.concat((_,__),0)
    
    # CDSCNN需要的数据格式转换
    def prepare_data(dataset):
        data = dataset.tensors[0]  # (batch, seq_len, features)
        labels = dataset.tensors[1]
        # CDSCNN期望输入格式为 (batch, features, seq_len)
        # 数据格式: (batch_size, 1024, 2) -> (batch_size, 2, 1024)
        return data, labels
    
    train_data, train_labels = prepare_data(train_dataset)
    valid_data, valid_labels = prepare_data(valid_dataset)
    
    # 清理不需要的中间变量
    del test_dataset, _, train_dataset, valid_dataset, ___
    # 每个进程只加载自己负责的数据分片
    train_data_rank, train_labels_rank = split_for_rank(train_data, train_labels, world_size, rank)
    valid_data_rank, valid_labels_rank = split_for_rank(valid_data, valid_labels, world_size, rank)
    train_dataset = OptimizedGPUDataset(train_data_rank, train_labels_rank, device=gpu_id)
    valid_dataset = OptimizedGPUDataset(valid_data_rank, valid_labels_rank, device=gpu_id)
    
    del train_data, train_labels, valid_data, valid_labels  # 释放原始数据内存
    
    # 检查是否在调试模式下，调整数据加载配置
    import sys
    is_debugging = sys.gettrace() is not None
    
    # 根据调试模式和GPU数量调整参数
    if is_debugging:
        num_workers = 0
        multiprocessing_context = None
        pin_memory = False
        persistent_workers = False
        prefetch_factor = None
    else:
        # 生产模式：最大化性能，但考虑到数据已经在GPU上，减少workers
        num_workers = 0  # 数据已在GPU，不需要CPU workers
        multiprocessing_context = None
        pin_memory = False  # 数据已在GPU
        persistent_workers = False
        prefetch_factor = None
    
    print(f"Rank {rank}: DataLoader配置 - workers: {num_workers}, pin_memory: {pin_memory}")
    
    # 分布式采样器已无必要（每个进程只持有自己那份数据），直接用普通 DataLoader
    # 优化：使用适中的batch_size平衡性能和训练效果
    dataloader_kwargs = {
        'batch_size': effective_batch_size,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    if num_workers > 0:
        dataloader_kwargs.update({
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor,
            'multiprocessing_context': multiprocessing_context
        })
    
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **dataloader_kwargs)
    valid_loader = DataLoader(dataset=valid_dataset, shuffle=False, **dataloader_kwargs)
    train_count = len(train_loader)
    valid_count = len(valid_loader)
    
    for k,i in enumerate(range(start_epoch, num_epochs)):
        current_epoch = k + start_epoch
        print(f"Epoch {current_epoch+1}/{num_epochs}")
        
        train_one_epoch(current_epoch,model, train_loader, optimizer, scheduler, gpu_id, compute_loss,accumulation_steps=1,train_count=train_count,tb_writer=tb_writer, batch_size_multiplier=batch_size_multiplier, scaler=scaler, world_size=world_size, initial_lr=adjusted_lr)  # 训练一个 epoch
        
        val_accuracy = validate(current_epoch, model, valid_loader, scheduler, gpu_id, compute_loss, valid_count,tb_writer=tb_writer, world_size=world_size)  # 在每个 epoch 结束时进行验证
        
        # 保存常规模型
        torch.save(model.module, PATH+'/model{:s}.pth'.format(str(current_epoch+mm)))  # 保存模型
        
        # 只在主进程中检查和保存最佳模型
        if rank == 0:
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"New best accuracy: {best_accuracy:.4f} at epoch {current_epoch+1}")
                
                # 保存最佳模型到tensorboard文件夹
                best_model_path = os.path.join(tb_dir, 'best_model.pth')
                torch.save(model.module, best_model_path)
                print(f"Best model saved to {best_model_path}")
                
                # 记录最佳准确度到tensorboard
                tb_writer.add_scalar('Best Accuracy', best_accuracy, current_epoch)
            
            # 保存检查点（每5个epoch或最后一个epoch）
            if (current_epoch + 1) % 5 == 0 or current_epoch == num_epochs - 1:
                checkpoint = {
                    'epoch': current_epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'mm': mm
                }
                checkpoint_path = os.path.join(PATH, f'checkpoint_{current_epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
    
    if rank == 0:
        tb_writer.close()
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CDSCNN Training')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--gpus', default=8, type=int,
                        help='number of gpus per node')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--continue_train', action='store_true',
                        help='continue training from latest checkpoint')
    args = parser.parse_args()
    
    # 设置随机种子
    setup_seed(3407)
    
    # 记录训练参数
    print(f"Training configuration:")
    print(f"  - Nodes: {args.nodes}")
    print(f"  - GPUs per node: {args.gpus}")
    print(f"  - Total epochs: {args.epochs}")
    print(f"  - Continue training: {args.continue_train}")
    
    world_size = args.gpus * args.nodes
    
    if world_size > 1:
        # 多GPU训练
        mp.spawn(train, nprocs=args.gpus, args=(world_size, args.gpus, args.epochs, args.continue_train))
    else:
        # 单GPU训练
        train(0, 1, 1, args.epochs, args.continue_train)
