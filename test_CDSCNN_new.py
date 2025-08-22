'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2024-09-08 17:50:19
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2025-08-22 Updated
Description: CDSCNN测试文件 - 基于test_CSGNet.py风格
- 支持生成不同SNR下的混淆矩阵和总体混淆矩阵
- 适配CDSCNN模型的数据格式 (batch, features, seq_len)
- 与main_CDSCNN.py保持一致的设置和优化
- 支持15个调制类型的RadioML数据集
- 增强的模型加载和错误处理
- 性能优化和调试模式支持

Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
'''
# 导入需要的模块
from model_compare.CDSCNN import CDSCNN as Net  # 根据main_CDSCNN.py使用的模型版本
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

# 设置GPU - 与main_CDSCNN.py保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
# 设置训练结果路径 - 与main_CDSCNN.py保持一致
BASE_PATH = './runs'
PATH = None

# 自动寻找最新的CDSCNN训练结果
if os.path.exists(BASE_PATH):
    cdscnn_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and 'CDSCNN' in d]
    if cdscnn_dirs:
        # 按时间排序，选择最新的
        cdscnn_dirs.sort(reverse=True)
        PATH = os.path.join(BASE_PATH, cdscnn_dirs[0])
        print(f"Using latest CDSCNN training results from: {PATH}")
    else:
        print(f"No CDSCNN training results found in {BASE_PATH}, using default path")
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


def plot_snr_accuracy(snr_accuracies):
    """
    绘制不同SNR下的准确率曲线
    """
    plt.figure(figsize=(10, 6))
    snrs = sorted(snr_accuracies.keys())
    accuracies = [snr_accuracies[snr] for snr in snrs]
    
    plt.plot(snrs, accuracies, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('CDSCNN Classification Accuracy vs SNR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


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
            
            # CDSCNN数据预处理：转换为 (batch, features, seq_len) 格式
            # 输入格式: (batch_size, 1024, 2) -> (batch_size, 2, 1024)
            if data.is_complex():
                # 如果是复数，转换为实数格式
                data_real = torch.stack([data.real, data.imag], dim=-1)  # (batch, seq_len, 2)
                data_real = data_real.permute(0, 2, 1)  # (batch, 2, seq_len)
            else:
                # 如果已经是实数格式，确保是正确的形状
                if data.shape[-1] == 2:  # (batch, seq_len, 2)
                    data_real = data.permute(0, 2, 1)  # (batch, 2, seq_len)
                else:
                    data_real = data
            
            # 确保数据类型为float32
            data_real = data_real.float()
            
            # 前向传播
            logits = model(data_real)
            if logits.dim() > 2:
                logits = logits.squeeze()
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=1)
            
            # 转换为numpy数组
            predictions_np = predictions.cpu().numpy()
            labels_np = label.cpu().numpy().flatten()
            snr_np = snr.cpu().numpy().flatten()
            
            # 收集所有预测和标签
            all_predictions.extend(predictions_np)
            all_labels.extend(labels_np)
            all_snrs.extend(snr_np)
            
            # 为每个SNR更新混淆矩阵
            for i in range(len(predictions_np)):
                current_snr = int(snr_np[i])
                if current_snr in snr_confusion_matrices:
                    true_label = int(labels_np[i])
                    pred_label = int(predictions_np[i])
                    snr_confusion_matrices[current_snr][true_label][pred_label] += 1
            
            if batch_idx % 50 == 0:
                print(f"Processed batch {batch_idx}/{len(val_loader)}")
    
    return (np.array(all_predictions), np.array(all_labels), 
            np.array(all_snrs), snr_confusion_matrices)


def test_all_snr():
    """
    对所有SNR进行测试
    """
    print("=== CDSCNN Model Testing ===")
    
    # 设置随机种子以确保结果的可重复性
    setup_seed(3407)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 15个调制类型 - 与训练保持一致
    modulation_types = [
        '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 
        'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 
        'WBFM', 'MOD11', 'MOD12', 'MOD13', 'MOD14'
    ]
    
    class_num = len(modulation_types)
    
    # SNR范围：-20dB到+18dB，步长2dB
    snr_list = list(range(-20, 20, 2))
    print(f"Testing SNR range: {snr_list}")
    
    # 加载数据集
    print("Loading dataset...")
    train_dataset, [train_snr, train_labels], valid_dataset, [valid_snr, valid_labels], test_dataset, [test_snr, test_labels] = torch.load(
        "dataset/RadioML_dataset.pt", map_location='cpu', weights_only=False
    )
    
    # 创建带SNR信息的数据集
    val_dataset = DatasetWithSNR(test_dataset.tensors[0], [test_snr, test_labels])
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=4)
    
    print(f"Test dataset size: {len(val_dataset)}")
    
    # 创建模型
    model = Net(num_classes=class_num).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 加载训练好的模型
    best_model_path = os.path.join(PATH, 'best_model.pth')
    model_files = []
    
    if os.path.exists(best_model_path):
        model_files.append(best_model_path)
        print(f"Found best model: {best_model_path}")
    
    # 也尝试寻找其他可能的模型文件
    if os.path.exists(PATH):
        for file in os.listdir(PATH):
            if file.startswith('model') and file.endswith('.pth'):
                model_path = os.path.join(PATH, file)
                if model_path not in model_files:
                    model_files.append(model_path)
    
    if not model_files:
        print(f"No trained model found in {PATH}")
        print("Available files:")
        if os.path.exists(PATH):
            for file in os.listdir(PATH):
                print(f"  {file}")
        print("Please train the model first using main_CDSCNN.py")
        return
    
    # 选择最新的模型文件
    latest_model = max(model_files, key=os.path.getmtime)
    print(f"Loading model from: {latest_model}")
    
    try:
        # 尝试加载模型
        checkpoint = torch.load(latest_model, map_location=device, weights_only=False)
        if hasattr(checkpoint, 'state_dict'):
            model.load_state_dict(checkpoint.state_dict())
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # 假设checkpoint就是模型的state_dict或整个模型
            if isinstance(checkpoint, nn.Module):
                model = checkpoint.to(device)
            else:
                model.load_state_dict(checkpoint)
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading methods...")
        
        try:
            # 尝试直接加载为整个模型
            model = torch.load(latest_model, map_location=device, weights_only=False)
            print("Model loaded as complete model!")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return
    
    # 模型设置为评估模式
    model.eval()
    
    # 开始测试
    print("Starting model validation...")
    start_time = time.time()
    
    all_predictions, all_labels, all_snrs, snr_confusion_matrices = validate_model(
        model, val_loader, device, class_num, snr_list
    )
    
    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds")
    
    # 计算总体混淆矩阵
    overall_cm = confusion_matrix(all_labels, all_predictions)
    
    # 计算总体性能指标
    overall_metrics = calculate_metrics(overall_cm)
    
    print(f"\n=== Overall Performance ===")
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {overall_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {overall_metrics['macro_f1']:.4f}")
    
    # 生成总体混淆矩阵图
    fig_overall = plot_confusion_matrix(
        overall_cm, 
        title="CDSCNN Overall Confusion Matrix", 
        class_names=modulation_types,
        normalize=True
    )
    plt.savefig(f'cdscnn_overall_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算各SNR下的准确率
    snr_accuracies = {}
    for snr in snr_list:
        if snr in snr_confusion_matrices:
            cm = snr_confusion_matrices[snr]
            if cm.sum() > 0:  # 确保该SNR有数据
                accuracy = np.diag(cm).sum() / cm.sum()
                snr_accuracies[snr] = accuracy
                print(f"SNR {snr:2d}dB: Accuracy = {accuracy:.4f}")
    
    # 绘制SNR-准确率曲线
    if snr_accuracies:
        fig_snr = plot_snr_accuracy(snr_accuracies)
        plt.savefig(f'cdscnn_snr_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 为每个SNR生成混淆矩阵（可选，只生成几个关键SNR的）
    key_snrs = [-10, 0, 10, 18]  # 选择几个关键SNR值
    for snr in key_snrs:
        if snr in snr_confusion_matrices and snr_confusion_matrices[snr].sum() > 0:
            fig_snr_cm = plot_confusion_matrix(
                snr_confusion_matrices[snr],
                title=f"CDSCNN Confusion Matrix at SNR = {snr}dB",
                class_names=modulation_types,
                normalize=True
            )
            plt.savefig(f'cdscnn_confusion_matrix_snr_{snr}db.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # 生成详细的分类报告
    print(f"\n=== Detailed Classification Report ===")
    report = classification_report(all_labels, all_predictions, target_names=modulation_types, digits=4)
    print(report)
    
    # 保存结果到文件
    with open('cdscnn_test_results.txt', 'w') as f:
        f.write("=== CDSCNN Model Test Results ===\n\n")
        f.write(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {overall_metrics['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {overall_metrics['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score: {overall_metrics['macro_f1']:.4f}\n\n")
        
        f.write("SNR-wise Accuracy:\n")
        for snr in sorted(snr_accuracies.keys()):
            f.write(f"SNR {snr:2d}dB: {snr_accuracies[snr]:.4f}\n")
        
        f.write(f"\nDetailed Classification Report:\n")
        f.write(report)
    
    print(f"\nResults saved to:")
    print(f"  - cdscnn_overall_confusion_matrix.png")
    print(f"  - cdscnn_snr_accuracy.png")
    print(f"  - cdscnn_test_results.txt")
    for snr in key_snrs:
        if snr in snr_confusion_matrices and snr_confusion_matrices[snr].sum() > 0:
            print(f"  - cdscnn_confusion_matrix_snr_{snr}db.png")
    
    print("\n=== Testing completed! ===")


if __name__ == "__main__":
    test_all_snr()
