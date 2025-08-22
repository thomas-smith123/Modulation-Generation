'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2025-08-22
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2025-08-22 Updated
Description: MCLDNN测试文件 - 基于test_PETCGDNN.py风格
- 支持生成不同SNR下的混淆矩阵和总体混淆矩阵
- 适配MCLDNN模型的多输入数据格式
- 与main_MCLDNN.py保持一致的设置和优化
- 支持15个调制类型的RadioML数据集
- 增强的模型加载和错误处理
- 性能优化和调试模式支持

Copyright (c) 2025 by thomas-smith123 &&, All Rights Reserved. 
'''
# 导入需要的模块
from model_compare.MCLDNN import create_mcldnn_model, prepare_mcldnn_inputs
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# 设置GPU - 与main_MCLDNN.py保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
# 设置训练结果路径 - 与main_MCLDNN.py保持一致
BASE_PATH = './runs'
PATH = None

# 自动寻找最新的MCLDNN训练结果
if os.path.exists(BASE_PATH):
    mcldnn_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and 'MCLDNN' in d]
    if mcldnn_dirs:
        # 按时间排序，选择最新的
        mcldnn_dirs.sort(reverse=True)
        PATH = os.path.join(BASE_PATH, mcldnn_dirs[0])
        print(f"Using latest MCLDNN training results from: {PATH}")
    else:
        print(f"No MCLDNN training results found in {BASE_PATH}, using default path")
        PATH = BASE_PATH
else:
    print(f"Base path {BASE_PATH} not found, using default")
    PATH = BASE_PATH

class MCLDNNTestDataset(torch.utils.data.Dataset):
    """
    MCLDNN测试专用数据集类，处理多输入格式
    """
    def __init__(self, data, labels, snr_values):
        """
        Args:
            data: RadioML数据，形状为[N, 1024, 2]
            labels: 标签，形状为[N]
            snr_values: SNR值，形状为[N]
        """
        self.data = data
        self.labels = labels
        self.snr_values = snr_values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 获取样本、标签和SNR
        sample = self.data[idx]  # [1024, 2]
        label = self.labels[idx]
        snr = self.snr_values[idx]
        
        # 转换为MCLDNN需要的多输入格式
        input1, input2, input3 = prepare_mcldnn_inputs(sample.unsqueeze(0))
        
        # 确保标签和SNR是正确的类型
        if isinstance(label, torch.Tensor):
            label = label.long().squeeze()
        else:
            label = torch.tensor(label, dtype=torch.long)
            
        if isinstance(snr, torch.Tensor):
            snr = snr.float().squeeze()
        else:
            snr = torch.tensor(snr, dtype=torch.float)
        
        return input1.squeeze(0), input2.squeeze(0), input3.squeeze(0), label, snr

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
        
        # 计算精确率
        if tp + fp > 0:
            prec = tp / (tp + fp)
        else:
            prec = 0.0
        precision.append(prec)
        
        # 计算召回率
        if tp + fn > 0:
            rec = tp / (tp + fn)
        else:
            rec = 0.0
        recall.append(rec)
        
        # 计算F1分数
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_score.append(f1)
    
    # 计算总体准确率
    accuracy = np.trace(cm) / np.sum(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_precision': np.mean(precision),
        'avg_recall': np.mean(recall),
        'avg_f1': np.mean(f1_score)
    }

def find_best_model(search_path):
    """
    在指定路径中寻找最佳模型文件
    
    Args:
        search_path: 搜索路径
        
    Returns:
        tuple: (模型文件路径, 模型文件名) 或 (None, None)
    """
    if not os.path.exists(search_path):
        return None, None
    
    # 首先查找best_model.pth
    best_model_path = os.path.join(search_path, 'best_model.pth')
    if os.path.exists(best_model_path):
        return best_model_path, 'best_model.pth'
    
    # 如果没有best_model.pth，查找最新的checkpoint
    checkpoint_files = [f for f in os.listdir(search_path) 
                       if f.startswith('checkpoint_') and f.endswith('.pth')]
    
    if checkpoint_files:
        # 按epoch数排序，选择最新的
        checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
        latest_checkpoint = checkpoint_files[0]
        return os.path.join(search_path, latest_checkpoint), latest_checkpoint
    
    # 查找其他.pth文件
    pth_files = [f for f in os.listdir(search_path) if f.endswith('.pth')]
    if pth_files:
        # 按文件修改时间排序
        pth_files.sort(key=lambda x: os.path.getmtime(os.path.join(search_path, x)), reverse=True)
        latest_pth = pth_files[0]
        return os.path.join(search_path, latest_pth), latest_pth
    
    return None, None

def main():
    print("MCLDNN Model Testing")
    print("=" * 50)
    
    # 搜索模型文件
    model_path, model_filename = find_best_model(PATH)
    
    if model_path is None:
        print(f"No model files found in {PATH}")
        print("Please ensure you have trained a model using main_MCLDNN.py first")
        
        # 显示搜索路径中的文件
        if os.path.exists(PATH):
            files = os.listdir(PATH)
            pth_files = [f for f in files if f.endswith('.pth')]
            print(f"Available files in search path:")
            print(f"  Total files: {len(files)}")
            if pth_files:
                print(f"  .pth files: {pth_files}...")
        return
    
    print(f"Found best model: {model_path}")
    print(f"Using best model: {model_filename}")
    print(f"Using model: {model_filename}")
    print(f"Model path: {model_path}")
    print(f"Search path used: {PATH}")
    
    # 创建结果保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PATH, f"test_results_MCLDNN_{timestamp}")
    print(f"Results will be saved to: {results_dir}")
    
    # 加载测试数据集
    print("Loading test dataset...")
    
    try:
        # 加载RadioML数据集
        data = torch.load('./dataset/RadioML_dataset.pt', weights_only=False)
        
        # 根据数据集格式解析数据
        if isinstance(data, list) and len(data) >= 6:
            # 格式: [train_dataset, train_snr, val_dataset, val_snr, test_dataset, test_snr]
            test_dataset_obj = data[4]  # 测试数据集对象
            test_snr = data[5]  # 测试SNR数据
            
            # 从TensorDataset中提取数据和标签
            test_data = test_dataset_obj.tensors[0]  # 数据
            test_labels = test_dataset_obj.tensors[1]  # 标签
            
        elif isinstance(data, dict):
            # 字典格式
            test_data = data['X_test']
            test_labels = data['Y_test']
            test_snr = data.get('test_snr', torch.zeros(len(test_labels)))
        else:
            raise ValueError(f"Unsupported dataset format: {type(data)}")
        
        print(f"Loaded dataset shapes:")
        print(f"  Test dataset: {test_data.shape}")
        print(f"  Test labels: {test_labels.shape}")
        print(f"  Test SNR: {test_snr.shape}")
        
        print(f"Dataset loaded successfully. Total samples: {len(test_data)}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 显示可用的SNR值
    unique_snrs = sorted(torch.unique(test_snr).tolist())
    print(f"Available SNR values: {unique_snrs}")
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"Loading model from: {model_path}")
    
    try:
        # 创建MCLDNN模型
        model = create_mcldnn_model(classes=15, input_shape1=[2, 1024], input_shape2=[1024, 1])
        
        # 加载训练好的权重
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from MCLDNN training directory: {model_path}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Loaded model weights from: {model_path}")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    print("Model loaded and set to evaluation mode")
    
    # 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params:,} parameters")
    
    # 创建测试数据集
    test_dataset = MCLDNNTestDataset(test_data, test_labels, test_snr)
    
    # 数据加载器
    def collate_fn(batch):
        """自定义collate函数处理MCLDNN的多输入格式"""
        input1_list, input2_list, input3_list, labels_list, snr_list = zip(*batch)
        
        input1_batch = torch.stack(input1_list, dim=0)
        input2_batch = torch.stack(input2_list, dim=0) 
        input3_batch = torch.stack(input3_list, dim=0)
        labels_batch = torch.stack(labels_list, dim=0)
        snr_batch = torch.stack(snr_list, dim=0)
        
        return input1_batch, input2_batch, input3_batch, labels_batch, snr_batch
    
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, 
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)
    
    print("Starting validation...")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 初始化结果存储
    all_predictions = []
    all_labels = []
    all_snrs = []
    snr_accuracies = {}
    snr_metrics = {}
    
    # 模型评估
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (input1, input2, input3, labels, snrs) in enumerate(test_loader):
            input1 = input1.to(device, non_blocking=True)
            input2 = input2.to(device, non_blocking=True)
            input3 = input3.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            snrs = snrs.to(device, non_blocking=True)
            
            # 前向传播
            with torch.cuda.amp.autocast():
                outputs = model(input1, input2, input3)
            
            _, predicted = torch.max(outputs, 1)
            
            # 收集结果
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_snrs.extend(snrs.cpu().numpy())
            
            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / len(test_loader) * 100
                print(f"Progress: {progress:.1f}% ({batch_idx + 1}/{len(test_loader)})")
    
    evaluation_time = time.time() - start_time
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # 转换为numpy数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_snrs = np.array(all_snrs)
    
    # 定义调制类型名称（15个类别）
    modulation_classes = [
        '8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 
        'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 
        'WBFM', 'B-FM', 'D-BPSK', 'D-QPSK', 'O-QPSK'
    ]
    
    print("Generating SNR-specific confusion matrices...")
    
    # 按SNR分析性能
    snr_results = []
    
    for snr in unique_snrs:
        snr_mask = all_snrs == snr
        snr_labels = all_labels[snr_mask]
        snr_predictions = all_predictions[snr_mask]
        
        if len(snr_labels) == 0:
            continue
            
        # 计算准确率
        accuracy = accuracy_score(snr_labels, snr_predictions)
        snr_accuracies[snr] = accuracy
        
        # 计算混淆矩阵
        cm = confusion_matrix(snr_labels, snr_predictions, labels=range(15))
        
        # 计算详细指标
        metrics = calculate_metrics(cm)
        snr_metrics[snr] = metrics
        
        print(f"SNR {snr} dB: Accuracy = {accuracy:.4f}, Samples = {len(snr_labels)}")
        
        # 保存SNR特定的混淆矩阵
        # 原始计数矩阵
        fig = plot_confusion_matrix(cm, f"Confusion Matrix - SNR {snr} dB", 
                                   modulation_classes, normalize=False)
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_snr_{snr}dB_counts.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 归一化百分比矩阵
        fig = plot_confusion_matrix(cm, f"Confusion Matrix - SNR {snr} dB", 
                                   modulation_classes, normalize=True)
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_snr_{snr}dB_normalized.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存混淆矩阵数据
        cm_df_counts = pd.DataFrame(cm, index=modulation_classes, columns=modulation_classes)
        cm_df_counts.to_csv(os.path.join(results_dir, f'confusion_matrix_snr_{snr}dB_counts.csv'))
        
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        cm_df_normalized = pd.DataFrame(cm_normalized, index=modulation_classes, columns=modulation_classes)
        cm_df_normalized.to_csv(os.path.join(results_dir, f'confusion_matrix_snr_{snr}dB_normalized.csv'))
        
        # 收集结果用于汇总
        snr_results.append({
            'SNR': snr,
            'Accuracy': accuracy,
            'Precision': metrics['avg_precision'],
            'Recall': metrics['avg_recall'],
            'F1': metrics['avg_f1'],
            'Samples': len(snr_labels)
        })
    
    print("Generating overall confusion matrix...")
    
    # 生成总体混淆矩阵
    overall_cm = confusion_matrix(all_labels, all_predictions, labels=range(15))
    overall_metrics = calculate_metrics(overall_cm)
    overall_accuracy = overall_metrics['accuracy']
    
    print(f"Overall accuracy: {overall_accuracy:.4f}")
    
    # 保存总体混淆矩阵
    fig = plot_confusion_matrix(overall_cm, "Overall Confusion Matrix", 
                               modulation_classes, normalize=False)
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_overall_counts.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    fig = plot_confusion_matrix(overall_cm, "Overall Confusion Matrix", 
                               modulation_classes, normalize=True)
    plt.savefig(os.path.join(results_dir, 'confusion_matrix_overall_normalized.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存总体混淆矩阵数据
    overall_cm_df_counts = pd.DataFrame(overall_cm, index=modulation_classes, columns=modulation_classes)
    overall_cm_df_counts.to_csv(os.path.join(results_dir, 'confusion_matrix_overall_counts.csv'))
    
    overall_cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis]
    overall_cm_normalized = np.nan_to_num(overall_cm_normalized)
    overall_cm_df_normalized = pd.DataFrame(overall_cm_normalized, index=modulation_classes, columns=modulation_classes)
    overall_cm_df_normalized.to_csv(os.path.join(results_dir, 'confusion_matrix_overall_normalized.csv'))
    
    # 生成SNR vs Accuracy图
    print("Generating SNR vs Accuracy plot...")
    
    snrs_list = list(snr_accuracies.keys())
    accuracies_list = list(snr_accuracies.values())
    
    plt.figure(figsize=(12, 8))
    plt.plot(snrs_list, accuracies_list, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Accuracy')
    plt.title('MCLDNN: Accuracy vs SNR')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # 添加数值标注
    for snr, acc in zip(snrs_list, accuracies_list):
        plt.annotate(f'{acc:.3f}', (snr, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'snr_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存结果汇总
    results_summary = pd.DataFrame(snr_results)
    results_summary.to_csv(os.path.join(results_dir, 'performance_summary.csv'), index=False)
    
    # 保存详细的分类报告
    detailed_report = classification_report(all_labels, all_predictions, 
                                          target_names=modulation_classes, 
                                          output_dict=True)
    
    detailed_report_df = pd.DataFrame(detailed_report).transpose()
    detailed_report_df.to_csv(os.path.join(results_dir, 'detailed_classification_report.csv'))
    
    # 生成TensorBoard日志
    tb_writer = SummaryWriter(log_dir=results_dir)
    
    # 记录总体指标
    tb_writer.add_scalar('Test/Overall_Accuracy', overall_accuracy, 0)
    tb_writer.add_scalar('Test/Overall_Precision', overall_metrics['avg_precision'], 0)
    tb_writer.add_scalar('Test/Overall_Recall', overall_metrics['avg_recall'], 0)
    tb_writer.add_scalar('Test/Overall_F1', overall_metrics['avg_f1'], 0)
    
    # 记录SNR特定的指标
    for snr in unique_snrs:
        if snr in snr_metrics:
            metrics = snr_metrics[snr]
            tb_writer.add_scalar(f'Test_SNR/Accuracy_SNR_{snr}', metrics['accuracy'], 0)
            tb_writer.add_scalar(f'Test_SNR/Precision_SNR_{snr}', metrics['avg_precision'], 0)
            tb_writer.add_scalar(f'Test_SNR/Recall_SNR_{snr}', metrics['avg_recall'], 0)
            tb_writer.add_scalar(f'Test_SNR/F1_SNR_{snr}', metrics['avg_f1'], 0)
    
    tb_writer.close()
    
    print(f"All results saved to: {results_dir}")
    print("保存的文件包括:")
    print("- CSV格式混淆矩阵(原始计数和百分比)")
    print("- PNG格式可视化图片")
    print("- 性能指标汇总表")
    print("- TensorBoard日志文件")
    print("Test completed successfully!")

if __name__ == '__main__':
    main()
