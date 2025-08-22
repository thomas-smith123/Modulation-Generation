'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2025-08-22
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2025-08-22 Updated
Description: MCLDNN训练文件 - 按 main_PETCGDNN.py 模板改写
- 分布式 + 混合精度
- 在线构造三路输入 (不预先逐样本缓存三路张量)
- 保存/日志命名使用 MCLDNN 前缀
'''
import os, time, argparse, random, sys, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from model_compare.MCLDNN import create_mcldnn_model

os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
PATH = './runs'

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction
    def forward(self, inputs, targets_onehot):
        # 兼容 onehot / index
        if targets_onehot.ndim==2:
            targets = targets_onehot.argmax(dim=1)
        else:
            targets = targets_onehot
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
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    if hasattr(torch.backends.cuda,'enable_flash_sdp'): torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda,'enable_math_sdp'): torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda,'enable_mem_efficient_sdp'): torch.backends.cuda.enable_mem_efficient_sdp(True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR']='localhost'; os.environ['MASTER_PORT']='12347'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class OptimizedGPUDataset(Dataset):
    """与 PETCGDNN 结构一致：预先把本 rank 数据放到对应 GPU."""
    def __init__(self, data, labels, device, num_classes=15):
        self.data = data.to(device, non_blocking=True, dtype=torch.float32, memory_format=torch.contiguous_format)
        self.labels = labels.to(device, non_blocking=True)
        self.labels_onehot = torch.nn.functional.one_hot(self.labels[:,0].to(torch.long), num_classes=num_classes).to(torch.float)
        print(f"Dataset on GPU {device}: {self.data.shape}")
    def __len__(self): return self.data.shape[0]
    def __getitem__(self, idx): return self.data[idx], self.labels[idx], self.labels_onehot[idx]

def train(rank, world_size, num_gpus, num_epochs, continue_train=False):
    print(rank, world_size, num_gpus, rank % num_gpus)
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    best_accuracy = 0.0; start_epoch = 0
    if rank==0:
        tb_writer = SummaryWriter(os.path.join(PATH, current_time+'MCLDNN'))
        tb_dir = os.path.join(PATH, current_time+'MCLDNN')
        for src,dst in [ ('main_MCLDNN.py','main_MCLDNN.py'), ('model_compare/MCLDNN.py','MCLDNN.py'), ('utils.py','utils.py'), ('dataset.py','dataset.py') ]:
            try: shutil.copy2(src, os.path.join(tb_dir,dst))
            except Exception as e: print('Copy fail', src, e)
    else:
        tb_writer=None; tb_dir=None
    gpu_id = rank % num_gpus
    setup(rank, world_size); torch.cuda.set_device(gpu_id)
    model = create_mcldnn_model(classes=15, input_shape1=[2,1024], input_shape2=[1024,1])
    if continue_train:
        ckpts=[f for f in os.listdir(PATH) if f.startswith('checkpoint_') and f.endswith('.pth')]
        if ckpts:
            ckpts.sort(key=lambda x:int(x.split('_')[1].split('.')[0]))
            latest=ckpts[-1]; cp=torch.load(os.path.join(PATH,latest), map_location=f'cuda:{gpu_id}', weights_only=False)
            model.load_state_dict(cp['model_state_dict']); start_epoch=cp['epoch']+1; best_accuracy=cp['best_accuracy']; mm=cp.get('mm',0)
            print(f"Resume from {latest} epoch {start_epoch} best {best_accuracy:.4f}")
        else:
            mm=0; print('No checkpoint found')
    else:
        mm=0
    model=model.to(gpu_id)
    is_debugging = sys.gettrace() is not None or hasattr(sys,'_getframe')
    if not is_debugging:
        try:
            model=torch.compile(model, mode='max-autotune', fullgraph=True); print(f'Rank {rank}: compiled')
        except Exception as e: print('compile fail', e)
    else:
        print(f'Rank {rank}: debug mode skip compile')
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], output_device=gpu_id, find_unused_parameters=False, broadcast_buffers=False, gradient_as_bucket_view=True)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Rank {rank}: params {params:,}')
    base_batch_size=128; batch_size_multiplier=1; base_lr=0.001
    if params < 500000: base_batch_size=128; batch_size_multiplier=2; base_lr=0.001; weight_decay=1e-3; print(f'Rank {rank}: small')
    elif params < 2000000: base_batch_size=128; batch_size_multiplier=1; base_lr=0.0008; weight_decay=5e-4; print(f'Rank {rank}: medium')
    else: base_batch_size=64; batch_size_multiplier=1; base_lr=0.0003; weight_decay=1e-4; print(f'Rank {rank}: large')
    effective_batch_size=base_batch_size*batch_size_multiplier
    compute_loss=FocalLoss(); adjusted_lr=base_lr*batch_size_multiplier
    scaler=torch.cuda.amp.GradScaler()
    optimizer=torch.optim.Adam(model.parameters(), lr=adjusted_lr, betas=(0.9,0.999), eps=1e-8, weight_decay=weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1,10//batch_size_multiplier), gamma=0.8)
    # 加载数据 (与 PETCGDNN 相同结构 list)
    train_ds,_,val_ds,___,test_ds,__ = torch.load('dataset/RadioML_dataset.pt', map_location='cpu', weights_only=False)
    _ = torch.concat((_,__),0)  # 占位，保持变量使用结构一致
    def prep(ds): return ds.tensors[0], ds.tensors[1]
    Xtr,Ytr=prep(train_ds); Xval,Yval=prep(val_ds)
    del test_ds, train_ds, val_ds, ___
    def split(data,labels,ws,r):
        total=data.shape[0]; per= total//ws; s=r*per; e=(r+1)*per if r<ws-1 else total; return data[s:e], labels[s:e]
    Xtr_r,Ytr_r=split(Xtr,Ytr,world_size,rank); Xval_r,Yval_r=split(Xval,Yval,world_size,rank)
    train_dataset=OptimizedGPUDataset(Xtr_r,Ytr_r, device=gpu_id); valid_dataset=OptimizedGPUDataset(Xval_r,Yval_r, device=gpu_id)
    del Xtr,Ytr,Xval,Yval
    dataloader_kwargs={'batch_size':effective_batch_size,'num_workers':0,'pin_memory':False}
    train_loader=DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    valid_loader=DataLoader(valid_dataset, shuffle=False, **dataloader_kwargs)
    train_count=len(train_loader); valid_count=len(valid_loader)
    for k,i in enumerate(range(start_epoch, num_epochs)):
        cur_epoch=k+start_epoch
        print(f'Epoch {cur_epoch+1}/{num_epochs}')
        train_one_epoch(cur_epoch, model, train_loader, optimizer, scheduler, gpu_id, compute_loss, accumulation_steps=1, train_count=train_count, tb_writer=tb_writer, batch_size_multiplier=batch_size_multiplier, scaler=scaler, world_size=world_size, initial_lr=adjusted_lr)
        val_acc=validate(cur_epoch, model, valid_loader, scheduler, gpu_id, compute_loss, valid_count, tb_writer=tb_writer, world_size=world_size)
        torch.save(model.module, PATH+f'/model{cur_epoch+mm}.pth')
        if rank==0:
            if val_acc>best_accuracy:
                best_accuracy=val_acc; print(f'New best {best_accuracy:.4f} at epoch {cur_epoch+1}')
                torch.save(model.module, os.path.join(tb_dir,'best_model.pth'))
                tb_writer.add_scalar('Best Accuracy', best_accuracy, cur_epoch)
            if (cur_epoch+1)%5==0 or cur_epoch==num_epochs-1:
                ck={'epoch':cur_epoch,'model_state_dict':model.module.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),'best_accuracy':best_accuracy,'mm':mm}
                pth=os.path.join(PATH,f'checkpoint_{cur_epoch}.pth'); torch.save(ck,pth); print('Checkpoint saved',pth)
                tb_writer.add_text('Checkpoint', f'Checkpoint saved at epoch {cur_epoch+1}', cur_epoch)
    if rank==0:
        print(f'Training completed best {best_accuracy:.4f}')
        tb_writer.add_text('Training Summary', f'Best accuracy: {best_accuracy:.4f}', num_epochs-1)
        tb_writer.close()
    cleanup()

def train_one_epoch(epoch,model, train_loader, optimizer, lr_scheduler, rank, compute_loss, accumulation_steps, train_count,tb_writer, warmup_epochs=3, batch_size_multiplier=1, scaler=None, world_size=1, initial_lr=None):
    model.train(); correct=0; total=0; total_loss=0; cnt=0; n=100; tmp = max(1,int(len(train_loader)/n))
    if epoch < warmup_epochs and initial_lr is not None:
        warmup_lr = initial_lr * (epoch + 1) / warmup_epochs
        for pg in optimizer.param_groups: pg['lr']=warmup_lr
        if rank==0: print(f'Warmup epoch {epoch}: lr={warmup_lr:.6f}')
    elif epoch==warmup_epochs and initial_lr is not None:
        for pg in optimizer.param_groups: pg['lr']=initial_lr
        if rank==0: print(f'Warmup finished lr={initial_lr:.6f}')
    optimizer.zero_grad()
    for j,(data,label,label_onehot) in enumerate(train_loader):
        input1 = data.transpose(1,2)          # [B,2,1024]
        input2 = data[:,:,0]                  # [B,1024]
        input3 = data[:,:,1]
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(input1,input2,input3); loss1 = compute_loss(pred, label_onehot); loss=loss1
            loss = loss/accumulation_steps
            pred_idx = torch.max(pred,1)[1]
            correct += torch.sum(pred_idx==(label.transpose(0,1)[0])); total += data.shape[0]
            scaler.scale(loss).backward()
            if (j+1)%accumulation_steps==0:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        else:
            pred = model(input1,input2,input3); loss1=compute_loss(pred,label_onehot); loss=loss1/accumulation_steps
            pred_idx = torch.max(pred,1)[1]; correct += torch.sum(pred_idx==(label.transpose(0,1)[0])); total += data.shape[0]
            loss.backward()
            if (j+1)%accumulation_steps==0:
                optimizer.step(); optimizer.zero_grad()
        total_loss += loss*accumulation_steps
        if (j+1)%n==0:
            c_t=torch.tensor([correct.item()],dtype=torch.float32,device=rank); t_t=torch.tensor([total],dtype=torch.float32,device=rank); l_t=torch.tensor([total_loss.item()],dtype=torch.float32,device=rank)
            dist.all_reduce(c_t); dist.all_reduce(t_t); dist.all_reduce(l_t)
            g_acc=c_t.item()/t_t.item(); g_loss=l_t.item()/world_size/n
            if rank==0:
                cur_lr=optimizer.param_groups[0]['lr']
                tb_writer.add_scalar('Learning Rate', cur_lr, epoch*tmp+cnt)
                tb_writer.add_scalar('Training Loss', g_loss, epoch*tmp+cnt)
                tb_writer.add_scalar('Training Accuracy', g_acc, epoch*tmp+cnt)
                tb_writer.add_scalar('Training Accuracy (Rank 0)', correct.item()/total, epoch*tmp+cnt)
                print(f'Tloss[{j}/{train_count}]:', g_loss,'global_acc:',g_acc,'rank0_acc:',correct.item()/total)
                total_loss=0; cnt+=1
    if epoch>=warmup_epochs: lr_scheduler.step()

def validate(epoch,model, val_loader, scheduler, rank, compute_loss, valid_count,tb_writer, world_size=1):
    model.eval(); correct=0; total_loss=0; total=0; cnt=0; n=5; tmp=max(1,int(len(val_loader)/n))
    with torch.no_grad():
        for j,(data,label,label_onehot) in enumerate(val_loader):
            input1=data.transpose(1,2); input2=data[:,:,0]; input3=data[:,:,1]
            pred=model(input1,input2,input3); loss1=compute_loss(pred,label_onehot); loss=loss1
            pred_idx=torch.max(pred,1)[1]; correct+=torch.sum(pred_idx==(label.transpose(0,1)[0])); total+=data.shape[0]
            total_loss+=loss
            if (j+1)%n==0:
                c_t=torch.tensor([correct.item()],dtype=torch.float32,device=rank); t_t=torch.tensor([total],dtype=torch.float32,device=rank); l_t=torch.tensor([total_loss.item()],dtype=torch.float32,device=rank)
                dist.all_reduce(c_t); dist.all_reduce(t_t); dist.all_reduce(l_t)
                g_acc=c_t.item()/t_t.item(); g_loss=l_t.item()/world_size/n
                if rank==0:
                    tb_writer.add_scalar('Validation Loss', g_loss, epoch*tmp+cnt)
                    tb_writer.add_scalar('Validation Accuracy', g_acc, epoch*tmp+cnt)
                    tb_writer.add_scalar('Validation Accuracy (Rank 0)', correct.item()/total, epoch*tmp+cnt)
                    print(f'Vloss[{j}/{valid_count}]:', g_loss,'global_acc:',g_acc,'rank0_acc:',correct.item()/total)
                    total_loss=0; cnt+=1
    c_final=torch.tensor([correct.item()],dtype=torch.float32,device=rank); t_final=torch.tensor([total],dtype=torch.float32,device=rank)
    dist.all_reduce(c_final); dist.all_reduce(t_final)
    return c_final.item()/t_final.item()

def main():
    parser=argparse.ArgumentParser(description='MCLDNN RadioML Training')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--epochs', type=int, default=50)
    args=parser.parse_args(); setup_seed(183565)
    num_gpus=torch.cuda.device_count(); world_size=num_gpus; num_epochs=args.epochs
    print(f'Starting MCLDNN training with {num_epochs} epochs')
    if args.continue_train: print('Continue mode on')
    mp.spawn(train, args=(world_size, num_gpus, num_epochs, args.continue_train), nprocs=world_size, join=True)

if __name__ == '__main__':
    if not os.path.exists(PATH): os.mkdir(PATH)
    main()
