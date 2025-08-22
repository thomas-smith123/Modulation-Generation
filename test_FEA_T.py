'''
FEA-T 测试脚本 (参考 test_CDSCNN.py 风格)
- 加载 runs 下最新 FEA_T 相关目录 (包含 best_model.pth)
- 计算整体准确率 & 按 SNR (若数据包含 snr) 的混淆矩阵
- 保存测试结果：混淆矩阵图、SNR准确率图、详细结果文件
'''
import os, re, torch, numpy as np
import torch.nn.functional as F
from model_compare.FEA_T import FEA_T
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def plot_confusion_matrix(cm, title, class_names=None, normalize=True, figsize=(12, 10)):
    """
    绘制混淆矩阵
    """
    plt.figure(figsize=figsize)
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)  # 处理除零情况
        annot_data = cm_norm
        fmt = '.3f'
        cmap = 'Blues'
    else:
        annot_data = cm
        fmt = 'd'
        cmap = 'Blues'
    
    # 使用seaborn绘制热图
    sns.heatmap(annot_data, 
                annot=True, 
                fmt=fmt, 
                cmap=cmap,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8},
                xticklabels=class_names if class_names else range(cm.shape[1]),
                yticklabels=class_names if class_names else range(cm.shape[0]))
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    return plt.gcf()

def calculate_metrics(cm):
    """
    从混淆矩阵计算各种指标
    """
    n_classes = cm.shape[0]
    precision = []
    recall = []
    f1_score = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        precision.append(prec)
        recall.append(rec)
        f1_score.append(f1)
    
    # 总体准确率
    accuracy = np.diag(cm).sum() / cm.sum()
    
    # 宏平均
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)
    
    return {
        'accuracy': accuracy,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

@torch.no_grad()
def inference_model(model, loader, device):
    model.eval(); all_preds=[]; all_targets=[]
    for data,label in loader:
        data=data.to(device).to(torch.float32)
        targets=label[:,0].to(torch.long).to(device)
        logits=model(data)  # 修正：移除不必要的unsqueeze(1)
        preds=logits.argmax(1)
        all_preds.append(preds.cpu()); all_targets.append(targets.cpu())
    return torch.cat(all_preds), torch.cat(all_targets)

@torch.no_grad()
def confusion_matrix(preds, targets, num_classes):
    cm=torch.zeros((num_classes,num_classes), dtype=torch.int64)
    for p,t in zip(preds,targets): cm[t,p]+=1
    return cm

def find_latest_run(path='runs'):
    cand=[]
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path,d)) and 'FEAT' in d:
            full=os.path.join(path,d,'best_model.pth')
            if os.path.isfile(full): cand.append((d, os.path.getmtime(full)))
    if not cand: return None
    cand.sort(key=lambda x:x[1], reverse=True)
    return os.path.join(path, cand[0][0])

def load_dataset(split='test'):
    train_ds, train_snr, val_ds, val_snr, test_ds, test_snr = torch.load('dataset/RadioML_dataset.pt', map_location='cpu', weights_only=False)
    if split=='test': return test_ds, test_snr
    if split=='val': return val_ds, val_snr
    return train_ds, train_snr

def main():
    print("=== FEA-T Model Testing ===")
    
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 15个调制类型 - 与训练保持一致
    modulation_types = [
        '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 
        'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 
        'WBFM', 'MOD11', 'MOD12', 'MOD13', 'MOD14'
    ]
    class_names = [f"Class_{i}" for i in range(15)]
    
    run_dir=find_latest_run()
    if run_dir is None:
        print('未找到包含 FEA_T 模型的运行目录'); return
    print('使用模型目录:', run_dir)
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(run_dir, f"test_results_FEAT_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")
    
    # 加载模型
    checkpoint = torch.load(os.path.join(run_dir,'best_model.pth'), map_location=device, weights_only=False)
    
    # 检查checkpoint结构并提取state_dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 处理DDP和torch.compile的前缀问题
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # 移除 _orig_mod. 前缀 (torch.compile)
        if key.startswith('_orig_mod.'):
            new_key = key[10:]  # 移除 '_orig_mod.' 前缀
        # 移除 module. 前缀 (DistributedDataParallel)
        elif key.startswith('module.'):
            new_key = key[7:]   # 移除 'module.' 前缀
        else:
            new_key = key
        cleaned_state_dict[new_key] = value
    
    model=FEA_T(num_class=15).to(device)
    missing, unexpected=model.load_state_dict(cleaned_state_dict, strict=False)
    if missing: print('Missing keys:', missing[:5], '... (showing first 5)')
    if unexpected: print('Unexpected keys:', unexpected[:5], '... (showing first 5)')
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # 加载测试数据
    dataset, snr_list = load_dataset('test')
    loader=DataLoader(dataset, batch_size=512, shuffle=False, pin_memory=True)
    print(f"Test dataset size: {len(dataset)}")
    
    # 推理
    print("Running inference...")
    preds, targets = inference_model(model, loader, device)
    
    # 计算总体混淆矩阵和指标
    cm=confusion_matrix(preds, targets, num_classes=15).numpy()
    overall_metrics = calculate_metrics(cm)
    
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {overall_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {overall_metrics['macro_f1']:.4f}")
    
    # 生成总体混淆矩阵图
    fig_overall = plot_confusion_matrix(
        cm, 
        title="FEA-T Overall Confusion Matrix", 
        class_names=class_names,
        normalize=True
    )
    plt.savefig(os.path.join(results_dir, 'feat_overall_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 按 SNR 统计
    snr_accuracies = {}
    snr_confusion_matrices = {}
    key_snrs = [-10, 0, 10, 18]  # 关键SNR点用于绘制混淆矩阵
    
    if snr_list is not None and len(snr_list)==len(dataset):
        snr_to_idx=defaultdict(list)
        for idx,snr in enumerate(snr_list): snr_to_idx[int(snr)].append(idx)
        
        print("\nSNR-wise Results:")
        for snr, idxs in sorted(snr_to_idx.items()):
            idxs_t=torch.tensor(idxs, dtype=torch.long)
            snr_preds=preds[idxs_t]; snr_targets=targets[idxs_t]
            snr_cm=confusion_matrix(snr_preds, snr_targets, 15).numpy()
            snr_acc=(snr_preds==snr_targets).float().mean().item()
            snr_accuracies[snr] = snr_acc
            snr_confusion_matrices[snr] = snr_cm
            print(f'SNR {snr:>2d}dB: Acc {snr_acc*100:5.2f}% (samples: {len(idxs)})')
        
        # 绘制SNR vs 准确率图
        if snr_accuracies:
            plt.figure(figsize=(10, 6))
            snrs = sorted(snr_accuracies.keys())
            accuracies = [snr_accuracies[snr] for snr in snrs]
            
            plt.plot(snrs, accuracies, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('SNR (dB)')
            plt.ylabel('Accuracy')
            plt.title('FEA-T Classification Accuracy vs SNR')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'feat_snr_accuracy.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 为关键SNR生成混淆矩阵
        for snr in key_snrs:
            if snr in snr_confusion_matrices and snr_confusion_matrices[snr].sum() > 0:
                fig_snr = plot_confusion_matrix(
                    snr_confusion_matrices[snr],
                    title=f"FEA-T Confusion Matrix (SNR = {snr}dB)",
                    class_names=class_names,
                    normalize=True
                )
                plt.savefig(os.path.join(results_dir, f'feat_confusion_matrix_snr_{snr}db.png'), dpi=300, bbox_inches='tight')
                plt.close()
    else:
        print('SNR 列表不可用，跳过按 SNR 分析')
    
    # 生成详细分类报告
    report = classification_report(targets.numpy(), preds.numpy(), 
                                 target_names=class_names, digits=4)
    
    # 保存结果到文件
    with open(os.path.join(results_dir, 'feat_test_results.txt'), 'w') as f:
        f.write("=== FEA-T Model Test Results ===\n\n")
        f.write(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {overall_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {overall_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score: {overall_metrics['macro_f1']:.4f}\n\n")
        
        if snr_accuracies:
            f.write("SNR-wise Accuracy:\n")
            for snr in sorted(snr_accuracies.keys()):
                f.write(f"SNR {snr:2d}dB: {snr_accuracies[snr]:.4f}\n")
        
        f.write(f"\nDetailed Classification Report:\n")
        f.write(report)
    
    print(f"\nResults saved to: {results_dir}")
    print(f"  - feat_overall_confusion_matrix.png")
    if snr_accuracies:
        print(f"  - feat_snr_accuracy.png")
    print(f"  - feat_test_results.txt")
    for snr in key_snrs:
        if snr in snr_confusion_matrices and snr_confusion_matrices[snr].sum() > 0:
            print(f"  - feat_confusion_matrix_snr_{snr}db.png")
    
    print("\n=== Testing completed! ===")

if __name__=='__main__':
    main()
