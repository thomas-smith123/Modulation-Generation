'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2024-09-08 17:50:19
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2025-08-21 Updated
Description: MCNet训练文件 - 基于main_ICAMC.py风格
- 支持分布式训练和混合精度训练
- 适配MCNet模型的数据格式
- 使用Focal Loss提升训练效果
- 支持断点续训和最佳模型保存
- 性能优化和调试模式支持

Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
'''
from model_compare.MCNet import MCNet as Net
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
    os.environ['MASTER_PORT'] = '12346'  # 确保此端口没有被占用
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()    

class OptimizedGPUDataset(Dataset):
    """已弃用：原来预先整体搬到 GPU。为避免多进程重复占满显存，改为使用原始 CPU TensorDataset + 按 batch 迁移。保留占位防止其它文件引用报错。"""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("OptimizedGPUDataset 已弃用，请直接使用原始 TensorDataset + DistributedSampler")

def split_for_rank(*args, **kwargs):  # 保留旧函数名以兼容潜在外部引用
    raise RuntimeError("split_for_rank 已移除，请使用 DistributedSampler")

def train(rank, world_size, num_gpus, num_epochs, continue_train=False):
    print(rank, world_size, num_gpus, rank % num_gpus)
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    
    # 初始化最佳准确度跟踪和起始epoch
    best_accuracy = 0.0
    start_epoch = 0
    
    if rank==0:
        # current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        tb_writer = SummaryWriter(os.path.join(PATH, current_time+'MCNet'))
        tb_dir = os.path.join(PATH, current_time+'MCNet')
        
        # 保存重要文件到tensorboard文件夹
        import shutil
        
        # 保存主要的Python文件
        try:
            shutil.copy2('main_MCNet.py', os.path.join(tb_dir, 'main_MCNet.py'))
            print(f"Saved main_MCNet.py to {tb_dir}")
        except Exception as e:
            print(f"Failed to save main_MCNet.py: {e}")
            
        # 保存网络模型文件
        try:
            shutil.copy2('model_compare/MCNet.py', os.path.join(tb_dir, 'MCNet.py'))
            print(f"Saved MCNet.py to {tb_dir}")
        except Exception as e:
            print(f"Failed to save MCNet.py: {e}")
            
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

    # MCNet模型实例化，输入格式: (batch_size, seq_len=1024, features=2)
    # input_shape参数格式: [batch_size, features, seq_len]
    model = Net(input_shape=[1, 2, 1024], modulation_num=15)  # 15个调制类型
    
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
    
    # 使用torch.compile优化模型推理速度（需要PyTorch 2.0+）
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
    # 根据GPU内存和训练稳定性选择合适的batch_size
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Rank {rank}: Model has {model_params:,} parameters")
    
    base_batch_size = 64  # MCNet比较复杂，使用更小的batch_size
    batch_size_multiplier = 1  # 可以根据需要调整
    base_lr = 0.001
    
    # 根据模型大小动态调整batch_size和学习率
    if model_params < 500000:  # 小网络
        base_batch_size = 128
        batch_size_multiplier = 2
        base_lr = 0.001
        weight_decay = 1e-3
        print(f"Rank {rank}: Small model detected, using conservative settings")
    elif model_params < 2000000:  # 中等网络
        base_batch_size = 128
        batch_size_multiplier = 1
        base_lr = 0.0001
        weight_decay = 5e-4
        print(f"Rank {rank}: Medium model detected")
    else:  # 大网络（MCNet属于这类）
        base_batch_size = 128
        batch_size_multiplier = 1
        base_lr = 0.0003
        weight_decay = 1e-4
        print(f"Rank {rank}: Large model detected")
        
    effective_batch_size = base_batch_size * batch_size_multiplier
    
    # 使用FocalLoss（标签用索引，不再手工 one-hot）
    compute_loss = FocalLoss()
    
    # 根据batch_size调整学习率 - 线性缩放规则
    lr_scale_factor = batch_size_multiplier  # 线性缩放
    adjusted_lr = base_lr * lr_scale_factor
    
    # 添加混合精度训练支持
    scaler = torch.cuda.amp.GradScaler()
    
    # 添加warmup来稳定大batch_size训练
    optimizer = torch.optim.Adam(model.parameters(), lr=adjusted_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    
    # 使用更激进的学习率衰减来配合大batch_size
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, 8//batch_size_multiplier), gamma=0.8)
    
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
    # train_dataset = TensorDataset(torch.concat((train_dataset.tensors[0],test_dataset.tensors[0]),0),torch.concat((train_dataset.tensors[1],test_dataset.tensors[1]),0))
    _ = torch.concat((_,__),0)
    
    # 优化数据预处理 - MCNet直接使用原始数据格式
    def prepare_data(dataset):
        data = dataset.tensors[0]  # 已经是 (batch, seq_len, features) 格式
        labels = dataset.tensors[1]
        # MCNet期望输入格式为 (batch, seq_len, features)，原始数据已经是这个格式
        # 数据格式: (batch_size, 1024, 2) - 1024是序列长度，2是I/Q两个特征
        return data, labels
    
    train_data, train_labels = prepare_data(train_dataset)
    valid_data, valid_labels = prepare_data(valid_dataset)
    # 直接使用原始 TensorDataset（CPU 上），依靠 DistributedSampler 切分
    # 兼容现有结构，重新构造 TensorDataset（保持 shape）
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    del test_dataset, _, ___  # 删除不必要变量
    
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
    
    # 使用 DistributedSampler
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    dataloader_kwargs = {
        'batch_size': effective_batch_size,
        'num_workers': num_workers,
        'pin_memory': True,  # 启用 page-locked 内存加速 H2D
    }
    if num_workers > 0:
        dataloader_kwargs.update({
            'persistent_workers': persistent_workers,
            'prefetch_factor': prefetch_factor,
            'multiprocessing_context': multiprocessing_context
        })
    train_loader = DataLoader(dataset=train_dataset, sampler=train_sampler, shuffle=False, **dataloader_kwargs)
    valid_loader = DataLoader(dataset=valid_dataset, sampler=valid_sampler, shuffle=False, **dataloader_kwargs)
    train_count = len(train_loader)
    valid_count = len(valid_loader)
    
    for current_epoch in range(start_epoch, num_epochs):
        print(f"Epoch {current_epoch+1}/{num_epochs}")

        # 设置 epoch 以确保各 rank shuffle 一致
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(current_epoch)
        
        # 训练一个 epoch
        train_one_epoch(current_epoch,model, train_loader, optimizer, scheduler, gpu_id, compute_loss,accumulation_steps=1,train_count=train_count,tb_writer=tb_writer, batch_size_multiplier=batch_size_multiplier, scaler=scaler, world_size=world_size, initial_lr=adjusted_lr)
        
        # 验证
        val_accuracy = validate(current_epoch, model, valid_loader, scheduler, gpu_id, compute_loss, valid_count,tb_writer=tb_writer, world_size=world_size)

        # 仅 rank0 保存常规模型
        if rank == 0:
            torch.save(model.module.state_dict(), PATH+f'/model{current_epoch+mm}.pth')
            
            # 检查与保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"New best accuracy: {best_accuracy:.4f} at epoch {current_epoch+1}")
                best_model_path = os.path.join(tb_dir, 'best_model.pth')
                torch.save(model.module.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")
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
                tb_writer.add_text('Checkpoint', f'Checkpoint saved at epoch {current_epoch+1}', current_epoch)
    
    # 训练结束时的总结
    if rank == 0:
        print(f"Training completed! Best accuracy achieved: {best_accuracy:.4f}")
        tb_writer.add_text('Training Summary', f'Training completed with best accuracy: {best_accuracy:.4f}', num_epochs-1)
    if rank==0:
        tb_writer.close()
    cleanup()

    
def train_one_epoch(epoch,model, train_loader, optimizer, lr_scheduler, rank, compute_loss, accumulation_steps, train_count,tb_writer, warmup_epochs=3, batch_size_multiplier=1, scaler=None, world_size=1, initial_lr=None):
    # print("model is created.")
    model.train()
    correct = 0
    total = 0
    total_loss = 0
    cnt = 0
    n = 100
    tmp = int(len(train_loader)/n)
    
    # 修正的Warmup学习率调整
    if epoch < warmup_epochs and initial_lr is not None:
        # 线性warmup：从很小的学习率逐渐增加到目标学习率
        warmup_lr = initial_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
        if rank == 0:
            print(f"Warmup epoch {epoch}: lr = {warmup_lr:.6f}")
    elif epoch == warmup_epochs and initial_lr is not None:
        # warmup结束后，恢复到初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = initial_lr
        if rank == 0:
            print(f"Warmup finished, restored lr to {initial_lr:.6f}")
    
    optimizer.zero_grad()
    for j,batch in enumerate(train_loader):
        data, label = batch  # label shape: (B,1)
        data = data.to(rank, non_blocking=True).to(torch.float32)
        labels_idx = label[:,0].to(torch.long).to(rank, non_blocking=True)
        
        # 使用混合精度训练
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(data)
                loss1 = compute_loss(pred, labels_idx)
                loss2 = 0 #torch.abs(attention).sum()*5e-6
                loss3 = 0# 1/torch.var(torch.abs(attention))*1e-2
                loss = loss1 + loss2 + loss3
            
            loss = loss / accumulation_steps  # 对loss进行平均以保持梯度尺度
            pred = torch.max(pred,1)[1]
            correct += torch.sum(pred==labels_idx)
            total += data.shape[0]
            
            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            if (j+1) % accumulation_steps == 0:  # 修正累积条件
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # 标准精度训练（向后兼容）
            pred = model(data)
            loss1 = compute_loss(pred, labels_idx)
            loss2 = 0 #torch.abs(attention).sum()*5e-6
            loss3 = 0# 1/torch.var(torch.abs(attention))*1e-2
            loss = loss1 + loss2 + loss3
            loss = loss / accumulation_steps  # 对loss进行平均以保持梯度尺度
            pred = torch.max(pred,1)[1]
            correct += torch.sum(pred==labels_idx)
            total += data.shape[0]
            loss.backward()
            if (j+1) % accumulation_steps == 0:  # 修正累积条件
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss * accumulation_steps  # 还原loss用于显示
            

        if (j+1)%n == 0:
            # 同步所有rank的训练统计信息（减少同步频率以提升性能）
            correct_tensor = torch.tensor([correct.item()], dtype=torch.float32, device=rank)
            total_tensor = torch.tensor([total], dtype=torch.float32, device=rank)
            loss_tensor = torch.tensor([total_loss.item()], dtype=torch.float32, device=rank)
            
            # 收集所有rank的数据
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            
            global_accuracy = correct_tensor.item() / total_tensor.item()
            global_loss = loss_tensor.item() / world_size / n
            
            if rank == 0:  # 只在主进程打印和记录
                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('Learning Rate', current_lr, epoch*tmp+cnt)########
                tb_writer.add_scalar('Training Loss', global_loss, epoch*tmp+cnt)
                tb_writer.add_scalar('Training Accuracy', global_accuracy, epoch*tmp+cnt)
                tb_writer.add_scalar('Training Accuracy (Rank 0)', correct.item()/total, epoch*tmp+cnt)  # 也记录rank 0的准确度作为对比
                
                # 添加性能监控
                samples_processed = total_tensor.item()
                tb_writer.add_scalar('Performance/SamplesProcessed', samples_processed, epoch*tmp+cnt)
                
                print(f"Tloss[{j}/{train_count}]:", global_loss,"global_acc:",global_accuracy, "rank0_acc:",correct.item()/total)
                total_loss = 0
                cnt += 1
    
    # 修正的scheduler调用：只在warmup结束后才调用
    if epoch >= warmup_epochs and (epoch+1)%1==0:
        lr_scheduler.step()
    

def validate(epoch,model, val_loader, scheduler, rank, compute_loss, valid_count,tb_writer, world_size=1):
    model.eval()
    correct = 0
    total_loss = 0
    total_loss_all = 0
    nn = 0
    total = 0
    cnt = 0
    n = 5
    tmp = int(len(val_loader)/n)
    with torch.no_grad():
        for j,batch in enumerate(val_loader):
            data, label = batch
            data = data.to(rank, non_blocking=True).to(torch.float32)
            labels_idx = label[:,0].to(torch.long).to(rank, non_blocking=True)
            
            # 数据已预处理为float32并在显存中，移除不必要的转换
            pred = model(data)
            loss1 = compute_loss(pred, labels_idx)
            loss2 = 0# torch.abs(attention).sum()*5e-6
            loss3 = 0 #1/torch.var(torch.abs(attention))*1e-2
            loss = loss1 + loss2 + loss3
            pred = torch.max(pred,1)[1]
            correct += torch.sum(pred==labels_idx)
            total += data.shape[0]
            total_loss += loss
            total_loss_all += loss
            nn = nn+n
            if (j+1)%n == 0:
                # 同步所有rank的验证统计信息
                correct_tensor = torch.tensor([correct.item()], dtype=torch.float32, device=rank)
                total_tensor = torch.tensor([total], dtype=torch.float32, device=rank)
                loss_tensor = torch.tensor([total_loss.item()], dtype=torch.float32, device=rank)
                
                # 收集所有rank的数据
                dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                
                global_accuracy = correct_tensor.item() / total_tensor.item()
                global_loss = loss_tensor.item() / world_size / n
                
                if rank == 0:  # 只在主进程打印和记录
                    tb_writer.add_scalar('Validation Loss', global_loss, epoch*tmp+cnt)
                    tb_writer.add_scalar('Validation Accuracy', global_accuracy, epoch*tmp+cnt)
                    tb_writer.add_scalar('Validation Accuracy (Rank 0)', correct.item()/total, epoch*tmp+cnt)  # 也记录rank 0的准确度作为对比
                    
                    print(f"Vloss:[{j}/{valid_count}]", global_loss,"global_acc:",global_accuracy, "rank0_acc:",correct.item()/total)
                    total_loss = 0
                    cnt += 1
    
    # 计算最终的全局准确度
    final_correct_tensor = torch.tensor([correct.item()], dtype=torch.float32, device=rank)
    final_total_tensor = torch.tensor([total], dtype=torch.float32, device=rank)
    
    # 收集所有rank的最终数据
    dist.all_reduce(final_correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(final_total_tensor, op=dist.ReduceOp.SUM)
    
    final_global_accuracy = final_correct_tensor.item() / final_total_tensor.item()
    
    return final_global_accuracy
    
def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='MCNet RadioML Classification Training')
    parser.add_argument('--continue_train', action='store_true', 
                       help='Continue training from the latest checkpoint')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs (default: 70)')
    args = parser.parse_args()
    
    setup_seed(183565)
    num_gpus = torch.cuda.device_count()  # 你拥有的 GPU 数量
    num_processes_per_gpu = 1  # 每个 GPU 上运行的进程数
    world_size = num_gpus * num_processes_per_gpu
    num_epochs = args.epochs  # 设置总训练 epoch 数
    
    print(f"Starting MCNet training with {num_epochs} epochs")
    if args.continue_train:
        print("Continue training mode enabled - will resume from latest checkpoint")
    else:
        print("Starting training from scratch")
    
    mp.spawn(train, args=(world_size, num_gpus, num_epochs, args.continue_train), nprocs=world_size, join=True)

if __name__ == "__main__":
    
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    
    main()
