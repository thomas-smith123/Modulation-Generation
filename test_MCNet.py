'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2024-09-08 17:50:19
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2025-08-21 Updated
Description: MCNet测试文件 - 基于test_ICAMC.py风格
- 支持生成不同SNR下的混淆矩阵和总体混淆矩阵
- 适配MCNet模型的数据格式 (batch, seq_len, features)
- 与main_MCNet.py保持一致的设置和优化
- 支持15个调制类型的RadioML数据集
- 增强的模型加载和错误处理
- 性能优化和调试模式支持

Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
'''
# 导入需要的模块
from model_compare.MCNet import MCNet as Net  # 根据main_MCNet.py使用的模型版本
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from dataset_snr import DatasetWithSNR
from torch.utils.data import DataLoader
import torch
import os
import seaborn as sns
import random
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

# 设置GPU - 与main_MCNet.py保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
# 设置训练结果路径 - 与main_MCNet.py保持一致
BASE_PATH = './runs'
PATH = None

# 自动寻找最新的MCNet训练结果
if os.path.exists(BASE_PATH):
    mcnet_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and 'MCNet' in d]
    if mcnet_dirs:
        # 按时间排序，选择最新的
        mcnet_dirs.sort(reverse=True)
        PATH = os.path.join(BASE_PATH, mcnet_dirs[0])
        print(f"Using latest MCNet training results from: {PATH}")
    else:
        print(f"No MCNet training results found in {BASE_PATH}, using default path")
        PATH = BASE_PATH
else:
    print(f"Base path {BASE_PATH} not found, using default")
    PATH = BASE_PATH


def plot_confusion_matrix(cm, title="Confusion Matrix", class_names=None, normalize=True):
    """
    绘制混淆矩阵
    Args:
        cm: 混淆矩阵
        title: 图标题
        class_names: 类别名称列表
        normalize: 是否归一化为百分比
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        # 按行归一化，得到百分比形式
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)  # 处理可能的NaN值
        
        # 绘制百分比混淆矩阵
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap="Blues", 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'}, annot_kws={'size': 7})
        plt.title(f'{title} (Normalized)')
    else:
        # 绘制原始计数混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 7})
        plt.title(f'{title} (Raw Counts)')
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    return plt.gcf()


def calculate_metrics(cm):
    """
    从混淆矩阵计算各种性能指标
    """
    # 计算每个类别的精确率、召回率、F1分数
    n_classes = cm.shape[0]
    precision = []
    recall = []
    f1_score = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        # 精确率
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision.append(prec)
        
        # 召回率
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        recall.append(rec)
        
        # F1分数
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        f1_score.append(f1)
    
    # 宏平均和微平均
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)
    
    # 微平均（等于总体准确率）
    micro_precision = micro_recall = micro_f1 = np.trace(cm) / np.sum(cm)
    
    return {
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        },
        'macro': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1_score': macro_f1
        },
        'micro': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1_score': micro_f1
        }
    }


def setup_seed(seed):
    """设置随机种子以确保结果可复现"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """主函数"""
    # 设置随机种子
    setup_seed(42)
    
    # 自动寻找最新的训练模型
    model_files = []
    if os.path.exists(PATH):
        model_files = [f for f in os.listdir(PATH) if f.startswith('model') and f.endswith('.pth')]
        model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    if not model_files:
        print(f"No model files found in {PATH}")
        print("Please ensure you have trained a model using main_MCNet.py first")
        return
    
    # 使用最新的模型
    latest_model = model_files[-1]
    model_path = os.path.join(PATH, latest_model)
    
    print(f"Using model: {latest_model}")
    print(f"Model path: {model_path}")
    
    # 创建时间戳用于结果文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"test_results_MCNet_{timestamp}"
    
    print(f"Results will be saved to: {save_dir}")
    
    # 生成混淆矩阵和性能分析
    generate_snr_confusion_matrices(model_path, save_dir)


def validate_model(model, val_loader, device, class_num=15, snr_list=None):
    """
    验证模型，生成混淆矩阵和性能指标
    """
    # 初始化混淆矩阵：[SNR数量, 类别数, 类别数]
    snr_confusion_matrices = {snr: np.zeros((class_num, class_num), dtype=int) for snr in snr_list}
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_snrs = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_loader):
            data, [snr, label] = batch_data
            
            if torch.cuda.is_available():
                data = data.to(device)
                label = label.to(device)
            
            # MCNet数据预处理：确保数据格式为 (batch, seq_len, features)
            # 输入格式: (batch_size, 1024, 2) - 1024是序列长度，2是I/Q两个特征
            if data.is_complex():
                # 如果是复数，转换为实数格式
                data_real = torch.stack([data.real, data.imag], dim=-1)  # (batch, seq_len, 2)
            else:
                # 如果已经是实数格式，确保是正确的形状
                data_real = data
            
            # 确保数据类型为float32
            data_real = data_real.to(torch.float32)
            
            # 前向传播 - MCNet期望输入格式为 (batch, seq_len, features)
            pred = model(data_real)
            # 检查模型输出是否为tuple (如果有额外输出)
            if isinstance(pred, tuple):
                pred = pred[0]  # 取第一个输出作为预测结果
            pred_labels = torch.argmax(pred, dim=1)
            
            # 收集预测结果
            for i, snr_val in enumerate(snr):
                snr_val = snr_val.item()
                # 处理标签格式 - 确保获取正确的类别标签
                if label.dim() > 1:
                    true_label = label[i, 0].item()  # 取第一列作为类别标签
                else:
                    true_label = label[i].item()
                pred_label = pred_labels[i].item()
                
                # 更新对应SNR的混淆矩阵
                snr_confusion_matrices[int(snr_val)][int(true_label), pred_label] += 1
                
                # 收集总体数据
                all_predictions.append(pred_label)
                all_labels.append(true_label)
                all_snrs.append(snr_val)
    
    # 计算总体混淆矩阵
    overall_confusion_matrix = np.zeros((class_num, class_num), dtype=int)
    for snr in snr_list:
        overall_confusion_matrix += snr_confusion_matrices[snr]
    
    return snr_confusion_matrices, overall_confusion_matrix, all_predictions, all_labels, all_snrs


def prepare_data_for_mcnet(dataset):
    """
    为MCNet准备数据格式
    MCNet期望输入格式为 (batch, seq_len, features)
    """
    # 检查原始数据格式
    print(f"Original data shape: {dataset.tensors[0].shape}")
    print(f"Original data type: {dataset.tensors[0].dtype}")
    
    # 获取原始数据
    data = dataset.tensors[0]
    labels = dataset.tensors[1]
    
    # 如果数据是复数格式，转换为实数格式
    if data.is_complex():
        # 复数数据: (batch, seq_len) -> (batch, seq_len, 2)
        data_real = torch.stack([data.real, data.imag], dim=-1)
        print(f"Converted complex data to real format: {data_real.shape}")
    else:
        # 如果已经是实数格式
        if data.dim() == 3 and data.shape[-1] == 2:
            # 已经是正确格式: (batch, seq_len, 2)
            data_real = data
        elif data.dim() == 2:
            # 需要添加特征维度: (batch, seq_len) -> (batch, seq_len, 1)
            data_real = data.unsqueeze(-1)
        else:
            # 其他格式，需要重新整形
            batch_size = data.shape[0]
            data_real = data.view(batch_size, -1, 2)  # 假设最后一维是2
    
    # 确保数据类型为float32
    data_real = data_real.to(torch.float32)
    
    print(f"Final data shape for MCNet: {data_real.shape}")
    print(f"Final data type: {data_real.dtype}")
    
    return data_real, labels


def generate_snr_confusion_matrices(model_path, save_dir="test_results_MCNet", test_batch_size=32):
    """
    生成不同SNR下的混淆矩阵和总体混淆矩阵
    """
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print("Loading test dataset...")
    
    # 加载测试数据集
    try:
        # 正确的数据加载方式：先加载数据，再创建DatasetWithSNR
        train_dataset, _, test_dataset, test_snr, valid_dataset, valid_snr = torch.load("./dataset/RadioML_dataset.pt", weights_only=False)
        
        print(f"Loaded dataset shapes:")
        print(f"  Test dataset: {test_dataset.tensors[0].shape}")
        print(f"  Test labels: {test_dataset.tensors[1].shape}")
        print(f"  Test SNR: {test_snr.shape}")
        
        # 为MCNet准备数据格式
        def prepare_data_for_mcnet(dataset):
            # 获取原始数据并转换格式
            data = dataset.tensors[0]  # 已经是正确格式或需要转换
            labels = dataset.tensors[1]
            return data, labels
        
        test_data, test_labels = prepare_data_for_mcnet(test_dataset)
        
        # 创建包含SNR信息的测试数据集
        dataset = DatasetWithSNR(test_data, [test_labels, test_snr])
        print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
        
        # 获取SNR值列表
        if test_snr.dim() > 1:
            snr_list = sorted(set(test_snr[:, 0].numpy().tolist()))
        else:
            snr_list = sorted(set(test_snr.numpy().tolist()))
        print(f"Available SNR values: {snr_list}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 调制类型名称（RadioML 2016.10a数据集）
    modulation_classes = [
        'BPSK', 'QPSK', '8PSK', '16QAM', '64QAM',
        'BFSK', 'CPFSK', 'PAM4', 'WBFM', 'AM-SSB',
        'AM-DSB', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC'
    ]
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    try:
        # MCNet模型实例化，与main_MCNet.py保持一致
        model = Net(input_shape=[1, 2, 1024], modulation_num=15)
        
        # 确定搜索路径（优先使用MCNet训练目录）
        search_path = PATH if PATH != BASE_PATH else BASE_PATH
        
        # 优先尝试加载MCNet训练目录中的最佳模型
        best_model_path = os.path.join(search_path, 'best_model.pth')
        if os.path.exists(best_model_path):
            model = torch.load(best_model_path, map_location=device, weights_only=False)
            print(f"Loaded best model from MCNet training directory: {best_model_path}")
        else:
            # 如果没有找到最佳模型，则加载指定的普通模型
            model = torch.load(model_path, map_location=device, weights_only=False)
            print(f"Best model not found, loaded model from: {model_path}")
            
        model = model.to(device)
        model.eval()
        print("Model loaded and set to evaluation mode")
        
        # 检查模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model has {total_params:,} parameters")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 根据调试模式调整数据加载器配置
    import sys
    is_debugging = sys.gettrace() is not None
    num_workers = 0 if is_debugging else 4
    
    test_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    
    # 进行验证
    print("Starting validation...")
    snr_cms, overall_cm, all_preds, all_labels, all_snrs = validate_model(
        model, test_loader, device, class_num=15, snr_list=snr_list
    )
    
    # 创建结果保存目录
    print(f"Saving results to: {save_dir}")
    
    # 保存每个SNR的混淆矩阵
    print("\nGenerating SNR-specific confusion matrices...")
    snr_accuracies = {}
    
    for snr in snr_list:
        cm = snr_cms[snr]
        
        # 计算准确率
        accuracy = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0
        snr_accuracies[snr] = accuracy
        print(f"SNR {snr} dB: Accuracy = {accuracy:.4f}, Samples = {np.sum(cm)}")
        
        # 绘制和保存混淆矩阵
        fig = plot_confusion_matrix(cm, 
                                  title=f"MCNet Confusion Matrix (SNR={snr}dB, Acc={accuracy:.3f})",
                                  class_names=modulation_classes, 
                                  normalize=True)
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_snr_{snr}dB.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始计数混淆矩阵
        fig_raw = plot_confusion_matrix(cm, 
                                      title=f"MCNet Confusion Matrix Raw Counts (SNR={snr}dB)",
                                      class_names=modulation_classes, 
                                      normalize=False)
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_raw_snr_{snr}dB.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算详细指标
        metrics = calculate_metrics(cm)
        
        # 保存详细指标到文件
        with open(os.path.join(save_dir, f'metrics_snr_{snr}dB.txt'), 'w') as f:
            f.write(f"MCNet Performance Metrics for SNR = {snr} dB\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write(f"Macro Average:\n")
            f.write(f"  Precision: {metrics['macro']['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['macro']['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['macro']['f1_score']:.4f}\n\n")
            f.write(f"Micro Average:\n")
            f.write(f"  Precision: {metrics['micro']['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['micro']['recall']:.4f}\n")
            f.write(f"  F1-Score: {metrics['micro']['f1_score']:.4f}\n\n")
            f.write("Per-Class Metrics:\n")
            for i, class_name in enumerate(modulation_classes):
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {metrics['per_class']['precision'][i]:.4f}\n")
                f.write(f"    Recall: {metrics['per_class']['recall'][i]:.4f}\n")
                f.write(f"    F1-Score: {metrics['per_class']['f1_score'][i]:.4f}\n")
    
    # 生成总体混淆矩阵
    print("\nGenerating overall confusion matrix...")
    try:
        overall_accuracy = np.trace(overall_cm) / np.sum(overall_cm)
        
        # 绘制总体混淆矩阵
        fig_total = plot_confusion_matrix(overall_cm, 
                                        title=f"MCNet Overall Confusion Matrix (Acc={overall_accuracy:.3f})",
                                        class_names=modulation_classes, 
                                        normalize=True)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_overall.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存原始计数总体混淆矩阵
        fig_total_raw = plot_confusion_matrix(overall_cm, 
                                            title="MCNet Overall Confusion Matrix (Raw Counts)",
                                            class_names=modulation_classes, 
                                            normalize=False)
        plt.savefig(os.path.join(save_dir, 'confusion_matrix_overall_raw.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 计算总体详细指标
        overall_metrics = calculate_metrics(overall_cm)
        
        # 保存总体指标
        with open(os.path.join(save_dir, 'overall_metrics.txt'), 'w') as f:
            f.write("MCNet Overall Performance Metrics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
            f.write(f"Macro Average:\n")
            f.write(f"  Precision: {overall_metrics['macro']['precision']:.4f}\n")
            f.write(f"  Recall: {overall_metrics['macro']['recall']:.4f}\n")
            f.write(f"  F1-Score: {overall_metrics['macro']['f1_score']:.4f}\n\n")
            f.write(f"Micro Average:\n")
            f.write(f"  Precision: {overall_metrics['micro']['precision']:.4f}\n")
            f.write(f"  Recall: {overall_metrics['micro']['recall']:.4f}\n")
            f.write(f"  F1-Score: {overall_metrics['micro']['f1_score']:.4f}\n\n")
            f.write("Per-Class Metrics:\n")
            for i, class_name in enumerate(modulation_classes):
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {overall_metrics['per_class']['precision'][i]:.4f}\n")
                f.write(f"    Recall: {overall_metrics['per_class']['recall'][i]:.4f}\n")
                f.write(f"    F1-Score: {overall_metrics['per_class']['f1_score'][i]:.4f}\n")
        
        print(f"Overall accuracy: {overall_accuracy:.4f}")
        
    except Exception as e:
        print(f"Error generating overall confusion matrix: {e}")
        import traceback
        traceback.print_exc()
    
    # 绘制SNR vs 准确率曲线
    if snr_accuracies:
        plt.figure(figsize=(10, 6))
        snrs = sorted(snr_accuracies.keys())
        accs = [snr_accuracies[snr] for snr in snrs]
        
        plt.plot(snrs, accs, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('SNR (dB)')
        plt.ylabel('Accuracy')
        plt.title('MCNet: Accuracy vs SNR')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 添加数值标签
        for snr, acc in zip(snrs, accs):
            plt.annotate(f'{acc:.3f}', (snr, acc), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'accuracy_vs_snr.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存SNR准确率数据
        with open(os.path.join(save_dir, 'snr_accuracies.txt'), 'w') as f:
            f.write("MCNet SNR vs Accuracy\n")
            f.write("=" * 25 + "\n")
            for snr in snrs:
                f.write(f"SNR {snr:3.0f} dB: {snr_accuracies[snr]:.4f}\n")
    
    print(f"\nAll results saved to: {save_dir}")
    print("Test completed successfully!")


def main():
    """主函数"""
    # 设置随机种子
    setup_seed(42)
    
    # 确定模型文件搜索路径
    search_path = PATH if PATH != BASE_PATH else BASE_PATH
    
    # 自动寻找最新的训练模型
    model_files = []
    if os.path.exists(search_path):
        model_files = [f for f in os.listdir(search_path) if f.startswith('model') and f.endswith('.pth')]
        model_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
    
    if not model_files:
        print(f"No model files found in {search_path}")
        print("Please ensure you have trained a model using main_MCNet.py first")
        print("Available files in search path:")
        if os.path.exists(search_path):
            all_files = os.listdir(search_path)
            print(f"  Total files: {len(all_files)}")
            pth_files = [f for f in all_files if f.endswith('.pth')]
            print(f"  .pth files: {pth_files[:10]}...")  # 只显示前10个
        return
    
    # 使用最新的模型
    latest_model = model_files[-1]
    model_path = os.path.join(search_path, latest_model)
    
    print(f"Using model: {latest_model}")
    print(f"Model path: {model_path}")
    print(f"Search path used: {search_path}")
    
    # 创建时间戳用于结果文件夹
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = f"test_results_MCNet_{timestamp}"
    
    print(f"Results will be saved to: {save_dir}")
    
    # 生成混淆矩阵和性能分析
    generate_snr_confusion_matrices(model_path, save_dir)


if __name__ == "__main__":
    print("MCNet Model Testing")
    print("=" * 50)
    main()
