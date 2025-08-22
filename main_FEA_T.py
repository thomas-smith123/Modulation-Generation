'''
FEA-T 训练脚本 (参考 main_MCNet.py 结构简化版)
- 分布式 DDP + 混合精度
- 使用 FocalLoss (索引标签)
- 自动保存 best_model 与 checkpoint
'''
import os, time, argparse, random, sys, shutil
import numpy as np
import torch

# 减少 Triton 编译信息输出
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
# 优化CPU性能设置
os.environ.setdefault("OMP_NUM_THREADS", "4")  # 限制OpenMP线程数
os.environ.setdefault("MKL_NUM_THREADS", "4")  # 限制MKL线程数
torch.set_num_threads(4)  # 限制PyTorch CPU线程数
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from model_compare.FEA_T import FEA_T

os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
PATH = './runs'

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__(); self.alpha=alpha; self.gamma=gamma; self.reduction=reduction
    def forward(self, inputs, targets):
        ce = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce)
        loss = self.alpha * (1-pt)**self.gamma * ce
        if self.reduction=='mean': return loss.mean()
        if self.reduction=='sum': return loss.sum()
        return loss

def setup_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark=True
    # 优化CUDA内存管理
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def setup(rank, world_size):
    os.environ['MASTER_ADDR']='localhost'; os.environ['MASTER_PORT']='12348'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup(): dist.destroy_process_group()

def build_model(num_classes):
    return FEA_T(num_class=num_classes)

def prepare_datasets():
    # 预期数据文件结构与 MCNet 相同: [train, train_snr, val, val_snr, test, test_snr]
    train_ds, train_snr, val_ds, val_snr, test_ds, test_snr = torch.load('dataset/RadioML_dataset.pt', map_location='cpu', weights_only=False)
    # 仅使用 train/val 参与训练与评估
    return train_ds, val_ds

def train_ddp(rank, world_size, epochs, continue_train=False):
    setup(rank, world_size)
    gpu_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(gpu_id)
    current_time = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    run_tag = current_time + 'FEAT'
    best_acc = 0.0; start_epoch=0; mm=0
    if rank==0:
        tb_dir=os.path.join(PATH, run_tag); os.makedirs(tb_dir, exist_ok=True)
        tb=SummaryWriter(tb_dir)
        for src,dst in [('main_FEA_T.py','main_FEA_T.py'), ('model_compare/FEA_T.py','FEA_T.py')]:
            try: shutil.copy2(src, os.path.join(tb_dir,dst))
            except Exception as e: print('Copy fail',src,e)
    else:
        tb=None; tb_dir=None

    num_classes=15
    model=build_model(num_classes=num_classes).to(gpu_id)
    if continue_train:
        ckpts=[f for f in os.listdir(PATH) if f.startswith('checkpoint_FEA_T_')]
        if ckpts:
            ckpts.sort(key=lambda x:int(x.split('_')[-1].split('.')[0]))
            latest=ckpts[-1]; cp=torch.load(os.path.join(PATH,latest), map_location=f'cuda:{gpu_id}', weights_only=False)
            model.load_state_dict(cp['model_state_dict']); start_epoch=cp['epoch']+1; best_acc=cp['best_accuracy']; mm=cp.get('mm',0)
            print(f'Rank {rank} resume {latest} from epoch {start_epoch} best {best_acc:.4f}')
    
    # 尝试编译模型以提升性能
    try:
        model=torch.compile(model, mode='max-autotune', fullgraph=True)
        if rank==0: print('Model compiled successfully')
    except Exception as e:
        if rank==0: print(f'Model compile failed: {e}, continuing without compilation')
    
    model=nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False, broadcast_buffers=False, gradient_as_bucket_view=True)

    # 动态 batch / lr 简化策略
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    base_bs=128; base_lr=1e-3; weight_decay=5e-4
    if params>2_000_000: base_bs=64; base_lr=6e-4; weight_decay=1e-4
    eff_bs=base_bs
    criterion=FocalLoss(); scaler=torch.cuda.amp.GradScaler()
    optimizer=torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.8)

    train_ds, val_ds = prepare_datasets()
    train_sampler=DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler=DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    # 优化数据加载器配置以减少CPU负载
    train_loader=DataLoader(train_ds, batch_size=eff_bs, sampler=train_sampler, 
                           pin_memory=True, num_workers=4, persistent_workers=True, 
                           prefetch_factor=2, drop_last=True)
    val_loader=DataLoader(val_ds, batch_size=eff_bs, sampler=val_sampler, 
                         pin_memory=True, num_workers=2, persistent_workers=True,
                         prefetch_factor=2, drop_last=False)

    for epoch in range(start_epoch, epochs):
        train_sampler.set_epoch(epoch)
        model.train(); running_loss=0; correct=0; total=0
        for step,(data,label) in enumerate(train_loader):
            # 优化数据传输 - 减少CPU开销
            data=data.to(gpu_id, non_blocking=True, dtype=torch.float32)  # 合并类型转换
            targets=label[:,0].to(gpu_id, non_blocking=True, dtype=torch.long)  # 合并类型转换
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logits=model(data)  # 移除不必要的unsqueeze，FEA_T内部会处理维度
                loss=criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            # 使用detach()减少CPU-GPU同步开销
            running_loss += loss.detach().item()*targets.size(0)
            pred=logits.argmax(1); correct += (pred==targets).sum().item(); total += targets.size(0)
            if (step+1)%200==0 and rank==0:  # 减少日志频率，从100改为200
                avg_loss = running_loss / total if total > 0 else 0
                avg_acc = correct / total if total > 0 else 0
                if tb:
                    tb.add_scalar('Train/Loss_iter', avg_loss, epoch*1000+step)
                    tb.add_scalar('Train/Acc_iter', avg_acc, epoch*1000+step)
        scheduler.step()
        # 验证阶段 - 添加混合精度和全局聚合
        model.eval(); val_correct=0; val_total=0
        with torch.no_grad():
            for data,label in val_loader:
                # 优化验证阶段的数据传输
                data=data.to(gpu_id, non_blocking=True, dtype=torch.float32)
                targets=label[:,0].to(gpu_id, non_blocking=True, dtype=torch.long)
                with torch.cuda.amp.autocast():
                    logits=model(data)  # 移除不必要的unsqueeze
                pred=logits.argmax(1); val_correct += (pred==targets).sum().item(); val_total += targets.size(0)
        # 汇总全局验证精度
        val_correct_t=torch.tensor([val_correct], dtype=torch.float32, device=gpu_id)
        val_total_t=torch.tensor([val_total], dtype=torch.float32, device=gpu_id)
        dist.all_reduce(val_correct_t); dist.all_reduce(val_total_t)
        val_acc=val_correct_t.item()/val_total_t.item()
        
        # 汇总全局训练统计
        train_loss_t=torch.tensor([running_loss], dtype=torch.float32, device=gpu_id)
        train_correct_t=torch.tensor([correct], dtype=torch.float32, device=gpu_id)
        train_total_t=torch.tensor([total], dtype=torch.float32, device=gpu_id)
        dist.all_reduce(train_loss_t); dist.all_reduce(train_correct_t); dist.all_reduce(train_total_t)
        train_acc = train_correct_t.item()/train_total_t.item() if train_total_t.item() > 0 else 0
        avg_train_loss = train_loss_t.item()/train_total_t.item() if train_total_t.item() > 0 else 0
        
        if rank==0 and tb:
            tb.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], epoch)
            tb.add_scalar('Train/EpochLoss', avg_train_loss, epoch)
            tb.add_scalar('Train/EpochAcc', train_acc, epoch)
            tb.add_scalar('Val/Acc', val_acc, epoch)
            print(f'Epoch {epoch+1}/{epochs} TrainAcc {train_acc:.4f} ValAcc {val_acc:.4f} Loss {avg_train_loss:.4f}')
            if val_acc>best_acc:
                best_acc=val_acc
                torch.save(model.module.state_dict(), os.path.join(tb_dir,'best_model.pth'))
                tb.add_scalar('Val/BestAcc', best_acc, epoch)
            if (epoch+1)%5==0 or epoch==epochs-1:
                ck={'epoch':epoch,'model_state_dict':model.module.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'best_accuracy':best_acc,'mm':mm}
                torch.save(ck, os.path.join(PATH,f'checkpoint_FEA_T_{epoch}.pth'))
    if rank==0 and tb:
        tb.add_text('Summary', f'Best Acc {best_acc:.4f}', epochs-1); tb.close()
    cleanup()

def main():
    parser=argparse.ArgumentParser(description='FEA-T Training')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--continue_train', action='store_true')
    args=parser.parse_args()
    setup_seed(1234)
    world_size=torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, args.epochs, args.continue_train), nprocs=world_size, join=True)

if __name__=='__main__':
    os.makedirs(PATH, exist_ok=True)
    main()
