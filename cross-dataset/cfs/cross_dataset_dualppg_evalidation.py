#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨数据集验证脚本 - 双流PPG模型
用MESA训练的PPG+Unfiltered PPG Cross-Attention模型在CFS数据集上测试

使用方法:
    python cross_dataset_ppg_evaluation.py \
        --model_path ./outputs/ppg_unfiltered_xxx/checkpoints/best_model.pth \
        --cfs_data_dir ../../data
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, cohen_kappa_score, f1_score
)
import json
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import argparse
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# 导入双流PPG模型
from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention


# ============================================================================
# 1. CFS PPG数据集 (用于跨数据集测试)
# ============================================================================

class CFSPPGTestDataset(Dataset):
    """
    CFS PPG测试数据集
    加载预处理好的CFS PPG数据用于跨数据集测试

    数据格式与MESA对齐:
    - PPG: [1, 1228800] (10小时 @ 34.13Hz)
    - Labels: [1200] (1200个30秒epoch)
    - 4类标签: Wake=0, Light=1, Deep=2, REM=3
    """

    def __init__(self, data_path, verbose=True):
        """
        Args:
            data_path: 数据目录，包含cfs_ppg_with_labels.h5和cfs_subject_index.h5
        """
        self.data_path = data_path
        self.verbose = verbose

        # 文件路径
        self.ppg_file = os.path.join(data_path, 'cfs_ppg_with_labels.h5')
        self.index_file = os.path.join(data_path, 'cfs_subject_index.h5')

        # 检查文件
        if not os.path.exists(self.ppg_file):
            raise FileNotFoundError(f"未找到PPG文件: {self.ppg_file}")
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"未找到索引文件: {self.index_file}")

        # 参数
        self.windows_per_subject = 1200
        self.samples_per_window = 1024

        # 加载所有被试
        self._load_subjects()

    def _load_subjects(self):
        """加载所有被试信息"""
        if self.verbose:
            print(f"🔧 加载CFS PPG数据...")
            print(f"   PPG文件: {self.ppg_file}")
            print(f"   索引文件: {self.index_file}")

        # 获取所有有效被试
        with h5py.File(self.index_file, 'r') as f:
            all_subjects = list(f['subjects'].keys())

            self.subjects = []
            self.subject_indices = {}

            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows == self.windows_per_subject:
                    indices = f[f'subjects/{subj}/window_indices'][:]
                    self.subjects.append(subj)
                    self.subject_indices[subj] = indices[0]  # 起始索引

        if self.verbose:
            print(f"\n✅ 加载完成:")
            print(f"   有效被试数: {len(self.subjects)}")
            print(f"   总epoch数: {len(self.subjects) * self.windows_per_subject:,}")

        # 统计标签分布
        self._compute_label_distribution()

    def _compute_label_distribution(self):
        """计算标签分布"""
        class_counts = np.zeros(4, dtype=np.int64)
        total_valid = 0

        with h5py.File(self.ppg_file, 'r') as f:
            for subj in self.subjects:
                start_idx = self.subject_indices[subj]
                labels = f['labels'][start_idx:start_idx + self.windows_per_subject]

                for label in labels:
                    if 0 <= label < 4:
                        class_counts[label] += 1
                        total_valid += 1

        self.class_counts = class_counts
        self.total_valid_epochs = total_valid

        if self.verbose:
            class_names = ['Wake', 'Light', 'Deep', 'REM']
            print(f"\n📊 标签分布:")
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                pct = count / total_valid * 100 if total_valid > 0 else 0
                print(f"   {i}: {name}: {count:,} ({pct:.1f}%)")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        """获取一个被试的完整10小时数据"""
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]

        with h5py.File(self.ppg_file, 'r') as f:
            # 读取1200个窗口
            ppg_windows = f['ppg'][start_idx:start_idx + self.windows_per_subject]
            labels = f['labels'][start_idx:start_idx + self.windows_per_subject]

        # 拼接成连续信号
        ppg_continuous = ppg_windows.reshape(-1)  # [1228800]

        # 转换为tensor
        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)  # [1, 1228800]
        labels_tensor = torch.LongTensor(labels)  # [1200]

        return ppg_tensor, labels_tensor


# ============================================================================
# 2. 跨数据集评估函数
# ============================================================================

def cross_dataset_ppg_evaluation(model_path, cfs_data_dir, output_dir, config):
    """
    使用MESA训练的双流PPG模型在CFS数据集上进行评估
    """

    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")

    # ========== 1. 加载MESA训练的模型 ==========
    print(f"\n📦 加载MESA训练的双流PPG模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型
    model = PPGUnfilteredCrossAttention(
        n_classes=4,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_fusion_blocks=config.get('n_fusion_blocks', 3),
        noise_config=config.get('noise_config', None)
    ).to(device)

    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_kappa' in checkpoint:
            print(f"   模型最佳验证Kappa: {checkpoint['best_kappa']:.4f}")
        if 'epoch' in checkpoint:
            print(f"   训练epoch: {checkpoint['epoch']}")
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"✅ 模型加载成功")
    print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 2. 加载CFS测试数据 ==========
    print(f"\n📂 加载CFS PPG数据: {cfs_data_dir}")

    test_dataset = CFSPPGTestDataset(cfs_data_dir, verbose=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # ========== 3. 在CFS上进行推理 ==========
    print(f"\n🧪 在CFS数据集上进行推理...")
    print(f"   被试数: {len(test_dataset)}")
    print(f"   Batch size: {config['batch_size']}")

    all_preds = []
    all_labels = []

    # 用于per-patient评估
    patient_predictions = defaultdict(list)
    patient_labels = defaultdict(list)

    # 记录模态权重
    clean_weights_all = []
    noisy_weights_all = []

    use_amp = config.get('use_amp', True) and torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (ppg, labels) in enumerate(tqdm(test_loader, desc="推理进度")):
            ppg = ppg.to(device)

            # 推理
            if use_amp:
                with autocast():
                    outputs = model(ppg)
            else:
                outputs = model(ppg)

            # 获取模态权重
            clean_weight, noisy_weight = model.get_modality_weights()
            if clean_weight is not None:
                clean_weights_all.append(clean_weight.mean().item() if hasattr(clean_weight, 'mean') else clean_weight)
                noisy_weights_all.append(noisy_weight.mean().item() if hasattr(noisy_weight, 'mean') else noisy_weight)

            # 处理输出
            outputs = outputs.permute(0, 2, 1)  # [B, 1200, 4]

            batch_size = outputs.shape[0]
            for i in range(batch_size):
                patient_idx = batch_idx * config['batch_size'] + i

                # 获取有效预测和标签
                mask = labels[i] != -1
                if mask.any():
                    valid_outputs = outputs[i][mask]
                    valid_labels = labels[i][mask]

                    _, predicted = valid_outputs.max(1)

                    # 保存
                    pred_np = predicted.cpu().numpy()
                    label_np = valid_labels.numpy()

                    patient_predictions[patient_idx].extend(pred_np)
                    patient_labels[patient_idx].extend(label_np)

                    all_preds.extend(pred_np)
                    all_labels.extend(label_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ========== 4. 计算评估指标 ==========
    print(f"\n📊 计算评估指标...")

    # Overall指标
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # Per-patient指标
    patient_kappas = []
    patient_accuracies = []
    patient_f1s = []

    for patient_idx in patient_predictions:
        if len(patient_predictions[patient_idx]) > 0:
            p_preds = np.array(patient_predictions[patient_idx])
            p_labels = np.array(patient_labels[patient_idx])

            patient_acc = accuracy_score(p_labels, p_preds)
            patient_accuracies.append(patient_acc)

            # 只有多个类别时才计算kappa
            if len(np.unique(p_labels)) > 1:
                patient_kappa = cohen_kappa_score(p_labels, p_preds)
                patient_kappas.append(patient_kappa)

            patient_f1 = f1_score(p_labels, p_preds, average='weighted', zero_division=0)
            patient_f1s.append(patient_f1)

    median_accuracy = np.median(patient_accuracies) if patient_accuracies else 0
    median_kappa = np.median(patient_kappas) if patient_kappas else 0
    median_f1 = np.median(patient_f1s) if patient_f1s else 0

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])

    # 每个类别的指标
    class_names = ['Wake', 'Light', 'Deep', 'REM']
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
        labels=[0, 1, 2, 3]
    )

    # ========== 5. 打印结果 ==========
    print(f"\n{'=' * 70}")
    print(f"跨数据集评估结果: MESA → CFS (双流PPG模型)")
    print(f"{'=' * 70}")

    print(f"\n📈 Overall指标:")
    print(f"   准确率: {accuracy * 100:.2f}%")
    print(f"   Kappa: {kappa:.4f}")
    print(f"   F1 (weighted): {f1_weighted:.4f}")
    print(f"   F1 (macro): {f1_macro:.4f}")

    print(f"\n📈 Per-Patient Median指标:")
    print(f"   准确率: {median_accuracy * 100:.2f}%")
    print(f"   Kappa: {median_kappa:.4f}")
    print(f"   F1: {median_f1:.4f}")

    if patient_kappas:
        print(f"\n   Kappa分布:")
        print(f"     Min: {np.min(patient_kappas):.4f}")
        print(f"     25%: {np.percentile(patient_kappas, 25):.4f}")
        print(f"     Median: {median_kappa:.4f}")
        print(f"     75%: {np.percentile(patient_kappas, 75):.4f}")
        print(f"     Max: {np.max(patient_kappas):.4f}")

    # 模态权重
    if clean_weights_all:
        avg_clean = np.mean(clean_weights_all)
        avg_noisy = np.mean(noisy_weights_all)
        print(f"\n🔀 模态权重 (平均):")
        print(f"   Clean PPG: {avg_clean:.3f}")
        print(f"   Noisy PPG: {avg_noisy:.3f}")

    print(f"\n📊 每个类别的性能:")
    print(f"{'类别':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for name in class_names:
        if name in report:
            p = report[name]['precision']
            r = report[name]['recall']
            f = report[name]['f1-score']
            s = report[name]['support']
            print(f"{name:<15} {p:>10.3f} {r:>10.3f} {f:>10.3f} {int(s):>10}")

    print(f"\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names,
                                zero_division=0, labels=[0, 1, 2, 3]))

    # ========== 6. 可视化 ==========
    # 混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)\nMESA → CFS (Dual-Stream PPG)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)\nMESA → CFS (Dual-Stream PPG)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_dataset_ppg_confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Per-patient Kappa分布
    if patient_kappas:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(patient_kappas, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(median_kappa, color='red', linestyle='--', linewidth=2,
                        label=f'Median: {median_kappa:.3f}')
        axes[0].axvline(np.mean(patient_kappas), color='orange', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(patient_kappas):.3f}')
        axes[0].set_xlabel('Kappa')
        axes[0].set_ylabel('Number of Patients')
        axes[0].set_title('Per-Patient Kappa Distribution\nMESA → CFS')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(patient_kappas, vert=True)
        axes[1].set_ylabel('Kappa')
        axes[1].set_title('Per-Patient Kappa Box Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cross_dataset_ppg_kappa_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # 每个类别的性能条形图
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(class_names))
    width = 0.25

    precision = [report[name]['precision'] if name in report else 0 for name in class_names]
    recall = [report[name]['recall'] if name in report else 0 for name in class_names]
    f1_scores = [report[name]['f1-score'] if name in report else 0 for name in class_names]

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='coral')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', color='seagreen')

    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance: MESA → CFS (Dual-Stream PPG)')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_dataset_ppg_per_class_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 7. 保存结果 ==========
    results = {
        'experiment': 'Cross-Dataset Evaluation: MESA → CFS (Dual-Stream PPG)',
        'model_path': model_path,
        'cfs_data_dir': cfs_data_dir,
        'n_subjects': len(test_dataset),
        'n_samples': len(all_labels),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'kappa': float(kappa),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro)
        },
        'per_patient_median_metrics': {
            'accuracy': float(median_accuracy),
            'kappa': float(median_kappa),
            'f1': float(median_f1)
        },
        'per_patient_kappa_stats': {
            'min': float(np.min(patient_kappas)) if patient_kappas else 0,
            'max': float(np.max(patient_kappas)) if patient_kappas else 0,
            'mean': float(np.mean(patient_kappas)) if patient_kappas else 0,
            'std': float(np.std(patient_kappas)) if patient_kappas else 0,
            'median': float(median_kappa),
            '25_percentile': float(np.percentile(patient_kappas, 25)) if patient_kappas else 0,
            '75_percentile': float(np.percentile(patient_kappas, 75)) if patient_kappas else 0
        },
        'modality_weights': {
            'clean_ppg': float(np.mean(clean_weights_all)) if clean_weights_all else None,
            'noisy_ppg': float(np.mean(noisy_weights_all)) if noisy_weights_all else None
        },
        'per_class_metrics': {
            name: {
                'precision': float(report[name]['precision']) if name in report else 0,
                'recall': float(report[name]['recall']) if name in report else 0,
                'f1': float(report[name]['f1-score']) if name in report else 0,
                'support': int(report[name]['support']) if name in report else 0
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

    results_path = os.path.join(output_dir, 'cross_dataset_ppg_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 结果已保存到: {output_dir}")

    return results


# ============================================================================
# 3. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Dataset Evaluation: MESA → CFS (Dual-Stream PPG)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='MESA训练的双流PPG模型路径 (.pth文件)')
    parser.add_argument('--cfs_data_dir', type=str, required=True,
                        help='CFS数据目录 (包含cfs_ppg_with_labels.h5)')
    parser.add_argument('--output_dir', type=str, default='./cross_dataset_ppg_results',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='禁用混合精度')

    args = parser.parse_args()

    print("=" * 70)
    print("🔄 跨数据集验证: MESA → CFS (双流PPG模型)")
    print("=" * 70)
    print("\n验证MESA训练的PPG+Unfiltered PPG模型在CFS上的泛化能力")

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
        'num_workers': args.num_workers,
        'use_amp': not args.no_amp,
        'd_model': 256,
        'n_heads': 8,
        'n_fusion_blocks': 3,
        'noise_config': {
            'noise_level': 0.1,
            'drift_amplitude': 0.1,
            'drift_frequency': 0.1,
            'spike_probability': 0.01,
            'spike_amplitude': 0.5
        }
    }

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'mesa_to_cfs_ppg_{timestamp}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    try:
        results = cross_dataset_ppg_evaluation(
            args.model_path,
            args.cfs_data_dir,
            output_dir,
            config
        )

        print(f"\n{'=' * 70}")
        print(f"🎉 跨数据集评估完成!")
        print(f"{'=' * 70}")
        print(f"\n最终结果 (MESA双流PPG → CFS):")
        print(f"   Overall准确率: {results['overall_metrics']['accuracy'] * 100:.2f}%")
        print(f"   Overall Kappa: {results['overall_metrics']['kappa']:.4f}")
        print(f"   Median Kappa: {results['per_patient_median_metrics']['kappa']:.4f}")
        print(f"   F1 (weighted): {results['overall_metrics']['f1_weighted']:.4f}")

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