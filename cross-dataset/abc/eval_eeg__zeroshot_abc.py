#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
跨数据集验证脚本
用MESA训练的AttnSleep模型在ABC数据集上测试

使用方法:
    python cross_dataset_eeg_abc.py \
        --model_path ./mesa_eeg_model/best_model.pth \
        --abc_data_dir ./abc_eeg_data
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
from collections import defaultdict

warnings.filterwarnings('ignore')

# ============================================================================
# 1. 模型组件 (从model.py复制，避免导入依赖问题)
# ============================================================================

import math
import copy
from copy import deepcopy
import torch.nn.functional as F


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return torch.nn.functional.gelu(x)


class MRCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(MRCNN, self).__init__()
        drate = 0.5
        self.GELU = GELU()
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        x_concat = self.AFR(x_concat)
        return x_concat


def attention(query, key, value, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        self.__padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=self.__padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attn = attention(query, key, value, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linear(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))
        return self.sublayer_output[1](x, self.feed_forward)


class TCE(nn.Module):
    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


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
# 2. ABC EEG数据集 (用于测试)
# ============================================================================

class ABCEEGTestDataset(Dataset):
    """
    ABC EEG测试数据集
    5类标签转换为4类以匹配MESA模型
    """

    def __init__(self, npz_files, verbose=True):
        if verbose:
            print(f"🔧 加载 {len(npz_files)} 个ABC EEG文件用于测试...")

        # ABC标签已经是4类，无需映射
        # 0: Wake, 1: Light, 2: Deep, 3: REM
        self.label_map = {
            0: 0,  # Wake -> Wake
            1: 1,  # Light -> Light
            2: 2,  # Deep -> Deep
            3: 3,  # REM -> REM
        }

        self.file_sample_map = []
        self.subject_sample_ranges = {}  # 用于per-patient评估
        failed_files = []

        total_samples = 0
        class_counts = np.zeros(4, dtype=np.int64)

        iterator = tqdm(npz_files, desc="加载ABC数据") if verbose else npz_files
        for npz_file in iterator:
            try:
                data = np.load(npz_file)
                y = data['y']

                subject_id = Path(npz_file).stem
                start_idx = len(self.file_sample_map)

                for idx in range(len(y)):
                    original_label = int(y[idx])
                    if original_label in self.label_map:
                        new_label = self.label_map[original_label]
                        self.file_sample_map.append((npz_file, idx, subject_id))
                        class_counts[new_label] += 1
                        total_samples += 1

                end_idx = len(self.file_sample_map)
                self.subject_sample_ranges[subject_id] = (start_idx, end_idx)

                data.close()

            except Exception as e:
                failed_files.append((npz_file, str(e)))
                continue

        if verbose:
            if failed_files:
                print(f"\n⚠️  {len(failed_files)} 个文件加载失败:")
                for f, e in failed_files[:5]:
                    print(f"   - {Path(f).name}: {e}")

            print(f"\n✅ ABC测试数据加载完成:")
            print(f"   总样本数: {total_samples:,}")
            print(f"   被试数: {len(npz_files) - len(failed_files)}")

            class_names = ['Wake', 'Light Sleep', 'Deep Sleep', 'REM']
            print(f"\n📊 类别分布:")
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                pct = count / total_samples * 100 if total_samples > 0 else 0
                print(f"   {i}: {name}: {count:,} ({pct:.1f}%)")

        self.class_counts = class_counts
        self.total_samples = total_samples
        self.n_subjects = len(npz_files) - len(failed_files)

    def __getitem__(self, index):
        file_path, sample_idx, subject_id = self.file_sample_map[index]

        data = np.load(file_path)
        x = data['x'][sample_idx]
        y = data['y'][sample_idx]
        data.close()

        original_label = int(y)
        y_label = self.label_map[original_label]

        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        y_tensor = torch.tensor(y_label, dtype=torch.long)

        # Z-score标准化
        x_tensor = (x_tensor - x_tensor.mean()) / (x_tensor.std() + 1e-8)

        return x_tensor, y_tensor, subject_id

    def __len__(self):
        return len(self.file_sample_map)


# ============================================================================
# 3. 跨数据集评估函数
# ============================================================================

def cross_dataset_evaluation(model_path, abc_data_dir, output_dir, config):
    """
    使用MESA训练的模型在ABC数据集上进行评估
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

    # ========== 2. 加载ABC测试数据 ==========
    print(f"\n📂 加载ABC测试数据: {abc_data_dir}")

    npz_files = sorted(glob.glob(os.path.join(abc_data_dir, '*.npz')))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"未在 {abc_data_dir} 找到NPZ文件")

    print(f"   找到 {len(npz_files)} 个被试文件")

    # 创建数据集和加载器
    test_dataset = ABCEEGTestDataset(npz_files, verbose=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # ========== 3. 在ABC上进行推理 ==========
    print(f"\n🧪 在ABC数据集上进行推理...")

    all_preds = []
    all_labels = []
    all_probs = []
    all_subjects = []

    with torch.no_grad():
        for data, target, subject_ids in tqdm(test_loader, desc="推理进度"):
            data = data.to(device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_subjects.extend(subject_ids)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # ========== 4. 计算评估指标 ==========
    print(f"\n📊 计算评估指标...")

    # Overall指标
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

    # ========== 5. Per-patient评估 ==========
    print(f"\n📊 计算Per-patient指标...")

    patient_results = defaultdict(lambda: {'preds': [], 'labels': []})
    for pred, label, subj in zip(all_preds, all_labels, all_subjects):
        patient_results[subj]['preds'].append(pred)
        patient_results[subj]['labels'].append(label)

    patient_kappas = []
    patient_accuracies = []
    patient_f1s = []
    patient_details = []

    for subj, data in patient_results.items():
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])

        acc = accuracy_score(labels, preds)
        patient_accuracies.append(acc)

        # 只有多类别时才计算kappa
        if len(np.unique(labels)) > 1:
            kap = cohen_kappa_score(labels, preds)
            patient_kappas.append(kap)
        else:
            kap = float('nan')

        f1 = f1_score(labels, preds, average='weighted', zero_division=0)
        patient_f1s.append(f1)

        patient_details.append({
            'subject_id': subj,
            'accuracy': float(acc),
            'kappa': float(kap) if not np.isnan(kap) else None,
            'f1': float(f1),
            'n_epochs': len(labels)
        })

    # 计算中位数
    median_accuracy = np.median(patient_accuracies)
    median_kappa = np.median([k for k in patient_kappas if not np.isnan(k)])
    median_f1 = np.median(patient_f1s)

    # ========== 6. 打印结果 ==========
    print(f"\n{'=' * 70}")
    print(f"跨数据集评估结果: MESA → ABC")
    print(f"{'=' * 70}")

    print(f"\n📈 Overall指标:")
    print(f"   准确率 (Accuracy): {accuracy * 100:.2f}%")
    print(f"   Cohen's Kappa: {kappa:.4f}")
    print(f"   F1 Score (weighted): {f1_weighted:.4f}")
    print(f"   F1 Score (macro): {f1_macro:.4f}")

    print(f"\n📈 Per-patient Median指标:")
    print(f"   Median准确率: {median_accuracy * 100:.2f}%")
    print(f"   Median Kappa: {median_kappa:.4f}")
    print(f"   Median F1: {median_f1:.4f}")

    print(f"\n📊 Per-patient Kappa分布:")
    valid_kappas = [k for k in patient_kappas if not np.isnan(k)]
    print(f"   Min: {np.min(valid_kappas):.4f}")
    print(f"   25%: {np.percentile(valid_kappas, 25):.4f}")
    print(f"   Median: {median_kappa:.4f}")
    print(f"   75%: {np.percentile(valid_kappas, 75):.4f}")
    print(f"   Max: {np.max(valid_kappas):.4f}")

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

    # ========== 7. 可视化 ==========
    # 混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)\nMESA → ABC Cross-Dataset')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)\nMESA → ABC Cross-Dataset')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # Kappa分布图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(valid_kappas, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(median_kappa, color='red', linestyle='--', label=f'Median: {median_kappa:.3f}')
    axes[0].set_xlabel('Cohen\'s Kappa')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Per-Patient Kappa Distribution')
    axes[0].legend()

    axes[1].boxplot(valid_kappas)
    axes[1].set_ylabel('Cohen\'s Kappa')
    axes[1].set_title('Per-Patient Kappa Boxplot')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kappa_distribution.png'),
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
    ax.set_title('Per-Class Performance: MESA → ABC Cross-Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ========== 8. 保存结果 ==========
    results = {
        'experiment': 'Cross-Dataset Evaluation: MESA → ABC',
        'model_path': model_path,
        'abc_data_dir': abc_data_dir,
        'n_subjects': test_dataset.n_subjects,
        'n_samples': len(test_dataset),
        'overall_metrics': {
            'accuracy': float(accuracy),
            'kappa': float(kappa),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro)
        },
        'per_patient_metrics': {
            'median_accuracy': float(median_accuracy),
            'median_kappa': float(median_kappa),
            'median_f1': float(median_f1),
            'mean_kappa': float(np.mean(valid_kappas)),
            'std_kappa': float(np.std(valid_kappas)),
            'min_kappa': float(np.min(valid_kappas)),
            'max_kappa': float(np.max(valid_kappas))
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

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 保存per-patient详细结果
    import pandas as pd
    patient_df = pd.DataFrame(patient_details)
    patient_df.to_csv(os.path.join(output_dir, 'patient_results.csv'), index=False)

    print(f"\n💾 结果已保存到: {output_dir}")
    print(f"   - results.json")
    print(f"   - patient_results.csv")
    print(f"   - confusion_matrix.png")
    print(f"   - kappa_distribution.png")
    print(f"   - per_class_performance.png")

    return results


# ============================================================================
# 4. 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Dataset Evaluation: MESA → ABC')
    parser.add_argument('--model_path', type=str, required=True,
                        help='MESA训练的模型路径 (.pth文件)')
    parser.add_argument('--abc_data_dir', type=str, required=True,
                        help='ABC EEG NPZ文件目录')
    parser.add_argument('--output_dir', type=str, default='./cross_dataset_eeg_abc_results',
                        help='输出目录')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载workers (Windows建议设为0)')

    args = parser.parse_args()

    print("=" * 70)
    print("🔄 跨数据集验证: MESA → ABC (EEG)")
    print("=" * 70)
    print("\n这个实验验证MESA训练的EEG模型在ABC数据集上的泛化能力")

    if torch.cuda.is_available():
        print(f"\n✅ CUDA可用")
        print(f"   使用GPU {args.gpu_id}: {torch.cuda.get_device_name(args.gpu_id)}")
    else:
        print("\n⚠️  CUDA不可用，使用CPU")

    config = {
        'batch_size': args.batch_size,
        'gpu_id': args.gpu_id,
        'num_workers': args.num_workers
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'mesa_to_abc_{timestamp}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        results = cross_dataset_evaluation(
            args.model_path,
            args.abc_data_dir,
            output_dir,
            config
        )

        print(f"\n{'=' * 70}")
        print(f"🎉 跨数据集评估完成!")
        print(f"{'=' * 70}")
        print(f"\n最终结果 (MESA EEG模型 → ABC测试):")
        print(f"   Overall准确率: {results['overall_metrics']['accuracy'] * 100:.2f}%")
        print(f"   Overall Kappa: {results['overall_metrics']['kappa']:.4f}")
        print(f"   Median Kappa: {results['per_patient_metrics']['median_kappa']:.4f}")
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