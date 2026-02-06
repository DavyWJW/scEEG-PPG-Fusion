#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨数据集验证脚本
用MESA训练的AttnSleep模型在CFS数据集上测试

这种方式可以验证模型的泛化能力，是评估模型鲁棒性的重要指标。

使用方法:
    python cross_dataset_evaluation.py \
        --model_path ./mesa_results/best_mesa_4class_corrected_model.pth \
        --cfs_data_dir ./cfs_eeg_c3m2_data
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, cohen_kappa_score, f1_score
)
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

# 导入模型组件
from model import MRCNN, TCE, EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward
from copy import deepcopy


# ============================================================================
# 1. AttnSleep 4类别模型 (与MESA训练时完全一致)
# ============================================================================

class AttnSleep4Class(nn.Module):
    """4类别版本的AttnSleep模型"""

    def __init__(self):
        super(AttnSleep4Class, self).__init__()

        N = 2
        d_model = 80
        d_ff = 120
        h = 5
        dropout = 0.1
        num_classes = 4
        afr_reduced_cnn_size = 30

        self.mrcnn = MRCNN(afr_reduced_cnn_size)

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)

        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x):
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return final_output


# ============================================================================
# 2. CFS EEG数据集 (用于测试)
# ============================================================================

class CFSEEGTestDataset(Dataset):
    """
    CFS EEG测试数据集
    5类标签转换为4类以匹配MESA模型
    """

    def __init__(self, npz_files, verbose=True):
        if verbose:
            print(f"🔧 加载 {len(npz_files)} 个CFS EEG文件用于测试...")

        # CFS标签映射: 5类 -> 4类
        self.cfs_label_map = {
            0: 0,  # Wake -> Wake
            1: 1,  # NREM1 -> Light Sleep
            2: 1,  # NREM2 -> Light Sleep
            3: 2,  # NREM3 -> Deep Sleep
            4: 3,  # REM -> REM
        }

        self.file_sample_map = []
        failed_files = []

        total_samples = 0
        class_counts = np.zeros(4, dtype=np.int64)

        iterator = tqdm(npz_files, desc="加载CFS数据") if verbose else npz_files
        for npz_file in iterator:
            try:
                data = np.load(npz_file)
                y = data['y']

                for idx in range(len(y)):
                    original_label = int(y[idx])
                    if original_label in self.cfs_label_map:
                        new_label = self.cfs_label_map[original_label]
                        self.file_sample_map.append((npz_file, idx))
                        class_counts[new_label] += 1
                        total_samples += 1

                data.close()

            except Exception as e:
                failed_files.append((npz_file, str(e)))
                continue

        if verbose:
            if failed_files:
                print(f"\n⚠️  {len(failed_files)} 个文件加载失败")

            print(f"\n✅ CFS测试数据加载完成:")
            print(f"   总样本数: {total_samples:,}")
            print(f"   被试数: {len(npz_files) - len(failed_files)}")

            class_names = ['Wake', 'Light Sleep', 'Deep Sleep', 'REM']
            print(f"\n📊 类别分布:")
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                pct = count / total_samples * 100 if total_samples > 0 else 0
                print(f"   {i}: {name}: {count:,} ({pct:.1f}%)")

        self.class_counts = class_counts
        self.total_samples = total_samples

    def __getitem__(self, index):
        file_path, sample_idx = self.file_sample_map[index]

        data = np.load(file_path)
        x = data['x'][sample_idx]
        y = data['y'][sample_idx]
        data.close()

        original_label = int(y)
        y_label = self.cfs_label_map[original_label]

        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        y_tensor = torch.tensor(y_label, dtype=torch.long)

        # Z-score标准化
        x_tensor = (x_tensor - x_tensor.mean()) / (x_tensor.std() + 1e-8)

        return x_tensor, y_tensor

    def __len__(self):
        return len(self.file_sample_map)


# ============================================================================
# 3. 跨数据集评估函数
# ============================================================================

def cross_dataset_evaluation(model_path, cfs_data_dir, output_dir, config):
    """
    使用MESA训练的模型在CFS数据集上进行评估
    """
    
    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")
    
    # ========== 1. 加载MESA训练的模型 ==========
    print(f"\n📦 加载MESA训练的模型: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = AttnSleep4Class().to(device)
    
    # 加载权重
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"✅ 模型加载成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 2. 加载CFS测试数据 ==========
    print(f"\n📂 加载CFS测试数据: {cfs_data_dir}")
    
    npz_files = sorted(glob.glob(os.path.join(cfs_data_dir, '*.npz')))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"未在 {cfs_data_dir} 找到NPZ文件")
    
    print(f"   找到 {len(npz_files)} 个被试文件")
    
    # 创建数据集和加载器
    test_dataset = CFSEEGTestDataset(npz_files, verbose=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # ========== 3. 在CFS上进行推理 ==========
    print(f"\n🧪 在CFS数据集上进行推理...")
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="推理进度"):
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # ========== 4. 计算评估指标 ==========
    print(f"\n📊 计算评估指标...")
    
    # 基本指标
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 每个类别的指标
    class_names = ['Wake', 'Light', 'Deep', 'REM']
    report = classification_report(
        all_labels, all_preds, 
        target_names=class_names, 
        output_dict=True,
        zero_division=0
    )
    
    # ========== 5. 打印结果 ==========
    print(f"\n{'=' * 70}")
    print(f"跨数据集评估结果: MESA → CFS")
    print(f"{'=' * 70}")
    
    print(f"\n📈 整体指标:")
    print(f"   准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"   Cohen's Kappa: {kappa:.4f}")
    print(f"   F1 Score (weighted): {f1_weighted:.4f}")
    print(f"   F1 Score (macro): {f1_macro:.4f}")
    
    print(f"\n📊 每个类别的性能:")
    print(f"{'类别':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for name in class_names:
        p = report[name]['precision']
        r = report[name]['recall']
        f = report[name]['f1-score']
        s = report[name]['support']
        print(f"{name:<15} {p:>10.3f} {r:>10.3f} {f:>10.3f} {int(s):>10}")
    
    print(f"\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # ========== 6. 可视化 ==========
    # 混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绝对数值
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)\nMESA → CFS Cross-Dataset')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # 归一化 (按行)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)\nMESA → CFS Cross-Dataset')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_dataset_confusion_matrix.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # 每个类别的性能条形图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(class_names))
    width = 0.25
    
    precision = [report[name]['precision'] for name in class_names]
    recall = [report[name]['recall'] for name in class_names]
    f1_scores = [report[name]['f1-score'] for name in class_names]
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='seagreen')
    
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance: MESA → CFS Cross-Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_dataset_per_class_performance.png'), 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    # ========== 7. 保存结果 ==========
    results = {
        'experiment': 'Cross-Dataset Evaluation: MESA → CFS',
        'model_path': model_path,
        'cfs_data_dir': cfs_data_dir,
        'n_subjects': len(npz_files),
        'n_samples': len(test_dataset),
        'metrics': {
            'accuracy': float(accuracy),
            'kappa': float(kappa),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro)
        },
        'per_class_metrics': {
            name: {
                'precision': float(report[name]['precision']),
                'recall': float(report[name]['recall']),
                'f1': float(report[name]['f1-score']),
                'support': int(report[name]['support'])
            } for name in class_names
        },
        'confusion_matrix': cm.tolist(),
        'class_distribution': {
            name: int(test_dataset.class_counts[i]) 
            for i, name in enumerate(class_names)
        },
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(output_dir, 'cross_dataset_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 结果已保存到: {output_dir}")
    print(f"   - cross_dataset_results.json")
    print(f"   - cross_dataset_confusion_matrix.png")
    print(f"   - cross_dataset_per_class_performance.png")
    
    return results


# ============================================================================
# 4. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Dataset Evaluation: MESA → CFS')
    parser.add_argument('--model_path', type=str, required=True,
                        help='MESA训练的模型路径 (.pth文件)')
    parser.add_argument('--cfs_data_dir', type=str, required=True,
                        help='CFS EEG NPZ文件目录')
    parser.add_argument('--output_dir', type=str, default='./cross_dataset_results',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载workers')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔄 跨数据集验证: MESA → CFS")
    print("=" * 70)
    print("\n这个实验验证MESA训练的模型在CFS数据集上的泛化能力")
    
    # 检查GPU
    if torch.cuda.is_available():
        print(f"\n✅ CUDA可用")
        print(f"   使用GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("\n⚠️  CUDA不可用，使用CPU")
    
    # 配置
    config = {
        'batch_size': args.batch_size,
        'gpu_id': args.gpu_id,
        'num_workers': args.num_workers
    }
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'mesa_to_cfs_{timestamp}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        results = cross_dataset_evaluation(
            args.model_path,
            args.cfs_data_dir,
            output_dir,
            config
        )
        
        print(f"\n{'=' * 70}")
        print(f"🎉 跨数据集评估完成!")
        print(f"{'=' * 70}")
        print(f"\n最终结果 (MESA模型 → CFS测试):")
        print(f"   准确率: {results['metrics']['accuracy'] * 100:.2f}%")
        print(f"   Kappa: {results['metrics']['kappa']:.4f}")
        print(f"   F1 (weighted): {results['metrics']['f1_weighted']:.4f}")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 70)
    print("✅ 全部完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
