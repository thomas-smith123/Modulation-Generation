'''
Author: thomas-smith123 thomas-smith@live.cn
Date: 2024-09-08 17:50:19
LastEditors: thomas-smith123 thomas-smith@live.cn
LastEditTime: 2025-08-21 Updated
Description: ICAMC测试文件 - 基于test_daelstm.py风格
- 支持生成不同SNR下的混淆矩阵和总体混淆矩阵
- 适配ICAMC模型的数据格式 (batch, seq_len, features)
- 与main_ICAMC.py保持一致的设置和优化
- 支持15个调制类型的RadioML数据集
- 增强的模型加载和错误处理
- 性能优化和调试模式支持

Copyright (c) 2024 by thomas-smith123 &&, All Rights Reserved. 
'''
# 导入需要的模块
from model_compare.ICAMC import ICAMC as Net  # 根据main_ICAMC.py使用的模型版本
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

# 设置GPU - 与main_ICAMC.py保持一致
os.environ["CUDA_VISIBLE_DEVICES"] = "0,7,1,2,3,4,5,6"
# 设置训练结果路径 - 与main_ICAMC.py保持一致
BASE_PATH = './runs'
PATH = None

# 自动寻找最新的ICAMC训练结果
if os.path.exists(BASE_PATH):
    icamc_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)) and 'ICAMC' in d]
    if icamc_dirs:
        # 按时间排序，选择最新的
        icamc_dirs.sort(reverse=True)
        PATH = os.path.join(BASE_PATH, icamc_dirs[0])
        print(f"Using latest ICAMC training results from: {PATH}")
    else:
        print(f"No ICAMC training results found in {BASE_PATH}, using default path")
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
    plt.title('Classification Accuracy vs SNR')
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
            
            # ICAMC数据预处理：确保数据格式为 (batch, seq_len, features)
            # 输入格式: (batch_size, 1024, 2) - 1024是序列长度，2是I/Q两个特征
            if data.is_complex():
                # 如果是复数，转换为实数格式
                data_real = torch.stack([data.real, data.imag], dim=-1)  # (batch, seq_len, 2)
            else:
                # 如果已经是实数格式，确保是正确的形状
                data_real = data
            
            # 确保数据类型为float32
            data_real = data_real.to(torch.float32)
            
            # 前向传播 - ICAMC期望输入格式为 (batch, seq_len, features)
            pred = model(data_real)
            pred_labels = torch.max(pred, 1)[1]
            
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
    
    # 生成总体混淆矩阵
    overall_cm = confusion_matrix(all_labels, all_predictions)
    
    return snr_confusion_matrices, overall_cm, all_predictions, all_labels, all_snrs


def test_model():
    """
    主测试函数
    """
    # 设置随机种子 - 与main_ICAMC.py保持一致
    setup_seed(183565)
    
    # 设置设备和性能优化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 启用性能优化
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimizations enabled")
    
    # 检查是否在调试模式下运行
    import sys
    is_debugging = sys.gettrace() is not None or hasattr(sys, '_getframe')
    print(f"Debug mode: {is_debugging}")
    
    # 加载最新的模型（你可以修改这里选择要测试的模型）
    model_path = os.path.join(PATH, 'best_model.pth')  # 使用最新的模型
    print(f"Loading model from: {model_path}")
    
    try:
        # 先尝试直接加载模型
        model = torch.load(model_path, map_location=device, weights_only=False)
        model = model.to(device)
        model.eval()  # 设置为评估模式
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        # 尝试加载检查点文件
        try:
            checkpoint_files = [f for f in os.listdir(PATH) if f.startswith('checkpoint_') and f.endswith('.pth')]
            if checkpoint_files:
                checkpoint_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                latest_checkpoint = checkpoint_files[-1]
                checkpoint_path = os.path.join(PATH, latest_checkpoint)
                print(f"Trying to load from checkpoint: {checkpoint_path}")
                
                # 创建模型实例 - ICAMC使用不同的输入格式
                model = Net(input_shape=[2, 1024], modulation_num=15)
                
                # 加载检查点
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)
                model.eval()
                print(f"Model loaded from checkpoint: {latest_checkpoint}")
            else:
                print("No checkpoint files found!")
                print(f"Available files in {PATH}: {os.listdir(PATH) if os.path.exists(PATH) else 'Path does not exist'}")
                return
        except Exception as e2:
            print(f"Error loading checkpoint: {e2}")
            return

    # 加载数据集
    print("Loading dataset...")
    train_dataset, _, test_dataset, test_snr, valid_dataset, valid_snr = torch.load("./dataset/RadioML_dataset.pt", weights_only=False)
    
    # 获取SNR列表和类别数 - 修正为15个类别
    snr_list = sorted(list(set(np.array(valid_snr[:, 0]))))
    class_num = 15  # RadioML数据集有15个调制类型
    
    # 创建简单的类别标签（label0, label1, etc.）
    class_names = [
        '32PSK',
        '16APSK',
        '32QAM',
        '32APSK',
        'OQPSK',
        'BPSK',
        '8PSK',
        '16PSK',
        '64APSK',
        '128QAM',
        '128APSK',
        '64QAM',
        'QPSK',
        '256QAM',
        '16QAM'
    ]
    
    print(f"SNR range: {min(snr_list)} to {max(snr_list)} dB")
    print(f"Number of classes: {class_num}")
    print(f"Number of validation samples: {len(valid_dataset)}")
    
    # 准备验证数据加载器 - ICAMC直接使用原始数据格式
    # 数据格式: (batch_size, 1024, 2) - 1024是序列长度，2是I/Q两个特征
    def prepare_data_for_icamc(dataset):
        data = dataset.tensors[0]  # 已经是 (batch, seq_len, features) 格式
        labels = dataset.tensors[1]
        return data, labels
    
    valid_data, valid_labels = prepare_data_for_icamc(valid_dataset)
    
    # 创建适配ICAMC的数据集
    valid_dataset = DatasetWithSNR(
        valid_data,  # 直接使用原始数据，不转换为复数
        [valid_labels, valid_snr]
    )
    
    # 根据调试模式调整数据加载器配置
    import sys
    is_debugging = sys.gettrace() is not None
    num_workers = 0 if is_debugging else 4
    
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    
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
    print("ICAMC OVERALL PERFORMANCE")
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
        title="ICAMC Overall Confusion Matrix", 
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
        title="ICAMC Overall Confusion Matrix (Raw Counts)", 
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
    print("ICAMC PER-SNR PERFORMANCE")
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
                title=f"ICAMC Confusion Matrix (SNR = {snr} dB)", 
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
        f.write("ICAMC Classification Report\n")
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
    print("ICAMC Test completed successfully!")
    
    tb_writer.close()


def main():
    test_model()


if __name__ == "__main__":
    if not os.path.exists('./runs'):
        os.makedirs('./runs')
    main()
