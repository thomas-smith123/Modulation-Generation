import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os

from model_compare.CDSCNN import CDSCNN


def test_model_detailed(model, test_loader, device, class_names):
    """详细测试模型性能"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_outputs = []
    
    total_time = 0
    num_batches = 0
    
    print("Testing model...")
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # 数据预处理：将实部和虚部concat
            data = data.view(data.size(0), 2, -1)  # [batch_size, 2, 1024]
            data, target = data.to(device), target.to(device)
            
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            total_time += inference_time
            num_batches += 1
            
            output = output.squeeze()
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            pred = output.argmax(dim=1)
            
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_outputs.extend(output.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f'Processing batch {batch_idx}/{len(test_loader)}')
    
    # 计算平均推理时间
    avg_inference_time = total_time / num_batches
    print(f"Average inference time per batch: {avg_inference_time*1000:.2f} ms")
    
    return np.array(all_predictions), np.array(all_targets), np.array(all_outputs)


def plot_confusion_matrix(predictions, targets, class_names, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(targets, predictions)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CDSCNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    plt.show()


def analyze_per_class_performance(predictions, targets, class_names):
    """分析每个类别的性能"""
    cm = confusion_matrix(targets, predictions)
    
    # 计算每个类别的准确率、召回率等
    per_class_acc = []
    per_class_precision = []
    per_class_recall = []
    
    for i in range(len(class_names)):
        # 准确率（该类预测正确的数量/该类总数量）
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        per_class_acc.append(class_acc)
        
        # 精确率（该类预测正确的数量/预测为该类的总数量）
        class_precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        per_class_precision.append(class_precision)
        
        # 召回率（等同于准确率）
        per_class_recall.append(class_acc)
    
    # 打印结果
    print("\nPer-class Performance:")
    print("-" * 80)
    print(f"{'Class':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'Support':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        support = cm[i, :].sum()
        print(f"{class_name:<15} {per_class_acc[i]:<10.4f} {per_class_precision[i]:<10.4f} "
              f"{per_class_recall[i]:<10.4f} {support:<10}")
    
    print("-" * 80)
    print(f"{'Average':<15} {np.mean(per_class_acc):<10.4f} {np.mean(per_class_precision):<10.4f} "
          f"{np.mean(per_class_recall):<10.4f} {cm.sum():<10}")


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 调制类型名称（15个类别）
    class_names = ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM', 'MOD11', 'MOD12', 'MOD13', 'MOD14']
    
    # 加载数据集
    print("Loading dataset...")
    train_dataset, _, valid_dataset, ___, test_dataset, __ = torch.load("dataset/RadioML_dataset.pt", map_location='cpu', weights_only=False)
    
    # 合并所有数据进行测试
    all_data = torch.concat((train_dataset.tensors[0], test_dataset.tensors[0]), 0)
    all_labels = torch.concat((train_dataset.tensors[1], test_dataset.tensors[1]), 0)
    
    # 创建完整数据集用于测试
    full_dataset = TensorDataset(all_data, all_labels)
    
    # 创建测试数据加载器
    batch_size = 64
    test_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"Total test samples: {len(full_dataset)}")
    
    # 创建模型
    num_classes = 11
    model = CDSCNN(num_classes=num_classes).to(device)
    
    # 加载训练好的模型
    model_path = 'runs/best_cdscnn_model.pth'
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
        print(f"Model was trained for {checkpoint['epoch']+1} epochs")
        print(f"Best test accuracy during training: {checkpoint['test_acc']:.2f}%")
    else:
        print(f"Model file {model_path} not found!")
        print("Please train the model first using main_CDSCNN.py")
        return
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # 测试模型
    predictions, targets, outputs = test_model_detailed(model, test_loader, device, class_names)
    
    # 计算总体准确率
    accuracy = (predictions == targets).mean()
    print(f"\nOverall Test Accuracy: {accuracy*100:.2f}%")
    
    # 生成分类报告
    print("\nClassification Report:")
    print(classification_report(targets, predictions, target_names=class_names, digits=4))
    
    # 分析每个类别的性能
    analyze_per_class_performance(predictions, targets, class_names)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(predictions, targets, class_names, 'cdscnn_confusion_matrix.png')
    
    # 计算按SNR的性能（如果数据集支持）
    print("\nTesting completed!")
    print(f"Results saved to confusion matrix plot.")


if __name__ == "__main__":
    main()
