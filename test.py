'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2024-09-08 17:50:19
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2024-09-24 10:31:59
Description: 改进的测试文件 - 支持生成不同SNR下的混淆矩阵和总体混淆矩阵

Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
'''
# 导入需要的模块
from model.modelv0_7 import Net  # 根据main.py使用的模型版本
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

# 设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
if os.path.exists('./complex_gru_single_radioml/runs'):
    PATH = './complex_gru_single_radioml/runs/20250816-093022'
else:
    PATH = './runs/20250816-093022'

# 不预定义标签，从数据中获取


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
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap="Blues", 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage'})
        plt.title(f'{title} (Normalized)')
    else:
        # 绘制原始计数混淆矩阵
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
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
    plt.title('Classification Accuracy vs SNR')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt.gcf()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate_model(model, val_loader, device, class_num=8, snr_list=None):
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
            
            # 数据预处理（与训练时保持一致）
            data = torch.unsqueeze(data, 1)
            data = torch.unsqueeze(data, 1)
            data = torch.concat((data.real, data.imag), 1)
            
            # 前向传播
            pred, _, _ = model(data.to(torch.float32))
            pred_labels = torch.max(pred, 1)[1]
            
            # 收集预测结果
            for i, snr_val in enumerate(snr):
                snr_val = snr_val.item()
                true_label = label[i].item()
                pred_label = pred_labels[i].item()
                
                # 更新对应SNR的混淆矩阵
                snr_confusion_matrices[int(snr_val)][int(true_label), pred_label] += 1
                
                # 收集总体数据
                all_predictions.append(pred_label)
                all_labels.append(true_label)
                all_snrs.append(snr_val)
    
    # 生成总体混淆矩阵
    overall_cm = confusion_matrix(all_labels, all_predictions)
    
    return snr_confusion_matrices, overall_cm, all_predictions, all_labels, all_snrs


def test_model():
    """
    主测试函数
    """
    # 设置随机种子
    setup_seed(93592)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载最新的模型（你可以修改这里选择要测试的模型）
    model_path = os.path.join(PATH, 'best_model.pth')  # 使用最新的模型
    print(f"Loading model from: {model_path}")
    
    try:
        model = torch.load(model_path, map_location=device)
        model = model.to(device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # 加载数据集
    print("Loading dataset...")
    train_dataset, _, valid_dataset, valid_snr, test_dataset, test_snr = torch.load("./dataset/RadioML_dataset.pt")
    
    # 获取SNR列表和类别数
    snr_list = sorted(list(set(np.array(valid_snr[:, 0]))))
    class_num = len(list(set(np.array(valid_dataset.tensors[1][:, 0]))))
    
    # 创建简单的类别标签（label0, label1, etc.）
    class_names = [f"label{i}" for i in range(class_num)]
    
    print(f"SNR range: {min(snr_list)} to {max(snr_list)} dB")
    print(f"Number of classes: {class_num}")
    print(f"Number of validation samples: {len(valid_dataset)}")
    
    # 准备验证数据加载器
    valid_dataset = DatasetWithSNR(
        torch.complex(valid_dataset.tensors[0][..., 0], valid_dataset.tensors[0][..., 1]), 
        [valid_dataset.tensors[1], valid_snr]
    )
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # 进行验证
    print("Starting validation...")
    snr_cms, overall_cm, all_preds, all_labels, all_snrs = validate_model(
        model, valid_loader, device, class_num, snr_list
    )
    
    # 创建结果保存目录
    result_dir = os.path.join(PATH, 'test_results')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存结果到TensorBoard
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    tb_writer = SummaryWriter(os.path.join(result_dir, f'test_{current_time}'))
    
    # 计算和显示总体性能
    print("\n" + "="*50)
    print("OVERALL PERFORMANCE")
    print("="*50)
    
    overall_metrics = calculate_metrics(overall_cm)
    print(f"Overall Accuracy: {overall_metrics['accuracy']:.4f}")
    print(f"Macro Precision: {overall_metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {overall_metrics['macro_recall']:.4f}")
    print(f"Macro F1-Score: {overall_metrics['macro_f1']:.4f}")
    
    # 绘制总体混淆矩阵（百分比形式）
    print("Generating overall normalized confusion matrix...")
    overall_cm_fig = plot_confusion_matrix(
        overall_cm, 
        title="Overall Confusion Matrix", 
        class_names=class_names,
        normalize=True
    )
    tb_writer.add_figure('Overall_Confusion_Matrix_Normalized', overall_cm_fig)
    overall_cm_fig.savefig(os.path.join(result_dir, 'overall_confusion_matrix_normalized.png'), dpi=300, bbox_inches='tight')
    plt.close(overall_cm_fig)
    
    # 绘制总体混淆矩阵（原始计数）
    print("Generating overall raw confusion matrix...")
    overall_cm_raw_fig = plot_confusion_matrix(
        overall_cm, 
        title="Overall Confusion Matrix (Raw Counts)", 
        class_names=class_names,
        normalize=False
    )
    tb_writer.add_figure('Overall_Confusion_Matrix_Raw', overall_cm_raw_fig)
    overall_cm_raw_fig.savefig(os.path.join(result_dir, 'overall_confusion_matrix_raw.png'), dpi=300, bbox_inches='tight')
    plt.close(overall_cm_raw_fig)
    
    # 保存总体混淆矩阵为CSV格式
    # 保存原始计数混淆矩阵
    overall_cm_df = pd.DataFrame(overall_cm, index=class_names, columns=class_names)
    overall_cm_df.to_csv(os.path.join(result_dir, 'overall_confusion_matrix_raw.csv'))
    
    # 保存百分比形式混淆矩阵
    overall_cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1)[:, np.newaxis]
    overall_cm_normalized = np.nan_to_num(overall_cm_normalized)
    overall_cm_norm_df = pd.DataFrame(overall_cm_normalized, index=class_names, columns=class_names)
    overall_cm_norm_df.to_csv(os.path.join(result_dir, 'overall_confusion_matrix_normalized.csv'))
    
    # 计算每个SNR下的性能
    print("\n" + "="*50)
    print("PER-SNR PERFORMANCE")
    print("="*50)
    
    snr_accuracies = {}
    snr_metrics_summary = []
    
    for snr in snr_list:
        cm = snr_cms[snr]
        metrics = calculate_metrics(cm)
        snr_accuracies[snr] = metrics['accuracy']
        
        print(f"SNR {snr:2.0f} dB: Accuracy = {metrics['accuracy']:.4f}, "
              f"Precision = {metrics['macro_precision']:.4f}, "
              f"Recall = {metrics['macro_recall']:.4f}, "
              f"F1 = {metrics['macro_f1']:.4f}")
        
        # 保存每个SNR的详细指标
        snr_metrics_summary.append({
            'SNR': snr,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['macro_precision'],
            'Recall': metrics['macro_recall'],
            'F1_Score': metrics['macro_f1']
        })
        
        # 绘制每个SNR的混淆矩阵（百分比形式）
        if cm.sum() > 0:  # 确保该SNR有数据
            print(f"Generating confusion matrix for SNR {snr} dB...")
            snr_cm_fig = plot_confusion_matrix(
                cm, 
                title=f"Confusion Matrix (SNR = {snr} dB)", 
                class_names=class_names,
                normalize=True
            )
            tb_writer.add_figure(f'SNR_{snr}_Confusion_Matrix', snr_cm_fig)
            snr_cm_fig.savefig(os.path.join(result_dir, f'confusion_matrix_snr_{snr}db.png'), dpi=300, bbox_inches='tight')
            plt.close(snr_cm_fig)
            
            # 保存每个SNR的混淆矩阵为CSV格式
            # 原始计数
            snr_cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            snr_cm_df.to_csv(os.path.join(result_dir, f'confusion_matrix_snr_{snr}db_raw.csv'))
            
            # 百分比形式
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            snr_cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
            snr_cm_norm_df.to_csv(os.path.join(result_dir, f'confusion_matrix_snr_{snr}db_normalized.csv'))
    
    # 绘制SNR-准确率曲线
    print("Generating SNR vs Accuracy plot...")
    snr_acc_fig = plot_snr_accuracy(snr_accuracies)
    tb_writer.add_figure('SNR_vs_Accuracy', snr_acc_fig)
    snr_acc_fig.savefig(os.path.join(result_dir, 'snr_vs_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close(snr_acc_fig)
    
    # 保存详细的性能报告
    metrics_df = pd.DataFrame(snr_metrics_summary)
    metrics_df.to_csv(os.path.join(result_dir, 'snr_performance_summary.csv'), index=False)
    
    # 保存分类报告
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=class_names,
        output_dict=True
    )
    
    with open(os.path.join(result_dir, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report\n")
        f.write("="*50 + "\n")
        f.write(classification_report(all_labels, all_preds, target_names=class_names))
        f.write("\n\nPer-SNR Performance Summary\n")
        f.write("="*50 + "\n")
        f.write(metrics_df.to_string(index=False))
        f.write("\n\nSaved Files Summary:\n")
        f.write("="*50 + "\n")
        f.write("- overall_confusion_matrix_raw.csv: 总体混淆矩阵(原始计数)\n")
        f.write("- overall_confusion_matrix_normalized.csv: 总体混淆矩阵(百分比)\n")
        f.write("- confusion_matrix_snr_XXdb_raw.csv: 各SNR混淆矩阵(原始计数)\n")
        f.write("- confusion_matrix_snr_XXdb_normalized.csv: 各SNR混淆矩阵(百分比)\n")
        f.write("- snr_performance_summary.csv: SNR性能汇总\n")
        f.write("- PNG图片文件: 可视化混淆矩阵和SNR准确率曲线\n")
    
    print(f"\nDetailed results saved to: {result_dir}")
    print("保存的文件包括:")
    print("- CSV格式混淆矩阵(原始计数和百分比)")
    print("- PNG格式可视化图片")
    print("- 性能指标汇总表")
    print("- TensorBoard日志文件")
    print("Test completed successfully!")
    
    tb_writer.close()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    test_model()


if __name__ == "__main__":
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    main()
