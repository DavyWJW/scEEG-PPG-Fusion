#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MESA预训练双流PPG模型在ABC上微调

基于finetune_dual_ppg_on_cfs.py修改，支持ABC数据集特点：
- 支持不完整记录（<1200 windows），自动padding
- 混合精度训练和梯度累积以节省内存
- 多种微调策略

用法:
    python finetune_dual_ppg_on_abc.py --strategy full --lr 1e-5
    python finetune_dual_ppg_on_abc.py --strategy head_only --lr 1e-4
    python finetune_dual_ppg_on_abc.py --strategy discriminative --lr 1e-4
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from datetime import datetime
from collections import Counter, defaultdict
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import gc

# 导入双流模型
sys.path.append('..')
try:
    from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
except ImportError:
    print("尝试从当前目录导入...")
    from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention


# ============================================================================
# 数据集
# ============================================================================

class ABCDualPPGDataset(Dataset):
    """ABC PPG数据集 - 用于PPGUnfilteredCrossAttention模型
    
    支持不完整记录（<1200 windows），自动padding到1200
    
    注意：PPGUnfilteredCrossAttention模型内部会自动生成噪声版本，
    所以数据集只需要返回干净的PPG信号
    """

    def __init__(self, ppg_file, index_file, split='train', seed=42,
                 train_ratio=0.6, val_ratio=0.2, min_windows=600):
        """
        Args:
            ppg_file: PPG数据H5文件
            index_file: 被试索引H5文件
            split: 'train', 'val', 'test'
            seed: 随机种子
            train_ratio: 训练集比例
            val_ratio: 验证集比例（测试集 = 1 - train - val）
            min_windows: 最小windows数，少于此数的被试将被排除
        """
        self.ppg_file = ppg_file
        self.index_file = index_file
        self.split = split
        self.target_windows = 1200
        self.samples_per_window = 1024

        # 加载索引
        with h5py.File(index_file, 'r') as f:
            all_subjects = list(f['subjects'].keys())
            
            # 过滤掉windows数太少的被试
            valid_subjects = []
            self.subject_n_windows = {}
            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows >= min_windows:
                    valid_subjects.append(subj)
                    self.subject_n_windows[subj] = n_windows
        
        print(f"有效被试: {len(valid_subjects)}/{len(all_subjects)} (min_windows={min_windows})")

        # 划分数据集
        test_ratio = 1 - train_ratio - val_ratio
        train_subjects, temp_subjects = train_test_split(
            valid_subjects, test_size=(val_ratio + test_ratio), random_state=seed
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=test_ratio/(val_ratio + test_ratio), random_state=seed
        )

        if split == 'train':
            self.subjects = train_subjects
        elif split == 'val':
            self.subjects = val_subjects
        else:
            self.subjects = test_subjects

        # 获取每个被试的起始索引
        self.subject_indices = {}
        with h5py.File(index_file, 'r') as f:
            for subj in self.subjects:
                indices = f[f'subjects/{subj}/window_indices'][:]
                self.subject_indices[subj] = indices[0]

        print(f"{split} set: {len(self.subjects)} subjects")

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]
        n_windows = self.subject_n_windows[subject_id]

        with h5py.File(self.ppg_file, 'r') as f:
            ppg_windows = f['ppg'][start_idx:start_idx + n_windows]
            labels = f['labels'][start_idx:start_idx + n_windows]

        # Padding到1200 windows
        if n_windows < self.target_windows:
            ppg_padded = np.zeros((self.target_windows, self.samples_per_window), dtype=np.float32)
            labels_padded = np.full(self.target_windows, -1, dtype=np.int64)
            
            ppg_padded[:n_windows] = ppg_windows
            labels_padded[:n_windows] = labels
        else:
            ppg_padded = ppg_windows[:self.target_windows].astype(np.float32)
            labels_padded = labels[:self.target_windows].astype(np.int64)

        # 拼接成连续信号
        ppg = ppg_padded.reshape(-1)  # [1228800]

        # 转换为tensor
        ppg_tensor = torch.FloatTensor(ppg).unsqueeze(0)  # [1, 1228800]
        labels_tensor = torch.LongTensor(labels_padded)  # [1200]

        return ppg_tensor, labels_tensor


# ============================================================================
# 微调训练器
# ============================================================================

class DualPPGFineTuner:
    """双流PPG微调训练器"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 内存优化设置
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.set_per_process_memory_fraction(0.95)

        # 混合精度训练
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

        # 梯度累积
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)

        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            config['output_dir'],
            f"abc_dual_finetune_{config['strategy']}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # 保存配置
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_pretrained_model(self):
        """加载预训练双流模型"""
        model = PPGUnfilteredCrossAttention()

        if self.config['pretrained_path'] and os.path.exists(self.config['pretrained_path']):
            print(f"\n加载预训练模型: {self.config['pretrained_path']}")
            checkpoint = torch.load(self.config['pretrained_path'], map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print("✅ 预训练双流模型加载成功")
        else:
            print("⚠️  未指定预训练模型或文件不存在，从头训练")

        return model.to(self.device)

    def setup_finetune_strategy(self, model):
        """设置微调策略"""
        strategy = self.config['strategy']

        if strategy == 'full':
            # 全模型微调
            print("\n策略: 全模型微调")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        elif strategy == 'head_only':
            # 只微调分类头和融合层
            print("\n策略: 只微调分类头和融合层")

            # 冻结两个编码器
            for param in model.clean_ppg_encoder.parameters():
                param.requires_grad = False
            for param in model.noisy_ppg_encoder.parameters():
                param.requires_grad = False

            # 可训练的部分：融合层、时序建模、分类头
            trainable_params = []

            # 融合层
            for param in model.fusion_blocks.parameters():
                param.requires_grad = True
            trainable_params.extend(model.fusion_blocks.parameters())

            # 自适应权重
            for param in model.modality_weighting.parameters():
                param.requires_grad = True
            trainable_params.extend(model.modality_weighting.parameters())

            # 特征聚合
            for param in model.feature_aggregation.parameters():
                param.requires_grad = True
            trainable_params.extend(model.feature_aggregation.parameters())

            # 时序建模
            for param in model.temporal_blocks.parameters():
                param.requires_grad = True
            trainable_params.extend(model.temporal_blocks.parameters())

            # 特征细化
            for param in model.feature_refinement.parameters():
                param.requires_grad = True
            trainable_params.extend(model.feature_refinement.parameters())

            # 分类器
            for param in model.classifier.parameters():
                param.requires_grad = True
            trainable_params.extend(model.classifier.parameters())

            optimizer = optim.Adam(
                trainable_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        elif strategy == 'progressive':
            # 渐进式解冻
            print("\n策略: 渐进式解冻")

            # 初始只训练分类头
            for param in model.parameters():
                param.requires_grad = False

            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.feature_refinement.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        elif strategy == 'discriminative':
            # 差异化学习率
            print("\n策略: 差异化学习率")

            for param in model.parameters():
                param.requires_grad = True

            base_lr = self.config['learning_rate']
            param_groups = [
                {'params': model.clean_ppg_encoder.parameters(), 'lr': base_lr * 0.01},
                {'params': model.noisy_ppg_encoder.parameters(), 'lr': base_lr * 0.01},
                {'params': model.fusion_blocks.parameters(), 'lr': base_lr * 0.1},
                {'params': model.modality_weighting.parameters(), 'lr': base_lr * 0.1},
                {'params': model.feature_aggregation.parameters(), 'lr': base_lr * 0.1},
                {'params': model.temporal_blocks.parameters(), 'lr': base_lr * 0.1},
                {'params': model.feature_refinement.parameters(), 'lr': base_lr},
                {'params': model.classifier.parameters(), 'lr': base_lr},
            ]

            optimizer = optim.Adam(param_groups, weight_decay=self.config['weight_decay'])

        else:
            raise ValueError(f"未知策略: {strategy}")

        # 统计可训练参数
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"可训练参数: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

        return optimizer

    def unfreeze_layer(self, model, layer_name):
        """解冻指定层"""
        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
            print(f"  解冻: {layer_name}")

    def calculate_class_weights(self, dataloader):
        """计算类别权重"""
        print("\n计算类别权重...")
        all_labels = []

        for ppg, labels in tqdm(dataloader, desc="扫描标签", leave=False):
            valid_labels = labels[labels >= 0].numpy().flatten()
            all_labels.extend(valid_labels.tolist())

        label_counts = Counter(all_labels)
        
        print("标签分布:")
        label_names = ['Wake', 'Light', 'Deep', 'REM']
        class_counts = []
        for i in range(4):
            count = label_counts.get(i, 1)
            class_counts.append(count)
            print(f"  {label_names[i]}: {count} ({100 * count / len(all_labels):.1f}%)")

        # 逆频率权重
        weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
        weights = weights / weights.sum() * 4

        print(f"类别权重: {[f'{w:.3f}' for w in weights.tolist()]}")

        return weights.to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion):
        """训练一个epoch - 带内存优化"""
        model.train()
        running_loss = 0.0
        total = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            ppg, labels = batch
            ppg = ppg.to(self.device)
            labels = labels.to(self.device)

            # 混合精度训练
            if self.use_amp:
                with autocast():
                    outputs = model(ppg)
                    outputs = outputs.permute(0, 2, 1)

                    loss = criterion(
                        outputs.reshape(-1, 4),
                        labels.reshape(-1)
                    )

                    # 梯度累积
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(ppg)
                outputs = outputs.permute(0, 2, 1)

                loss = criterion(
                    outputs.reshape(-1, 4),
                    labels.reshape(-1)
                )

                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

            mask = labels != -1
            valid_count = mask.sum().item()
            total += valid_count
            running_loss += loss.item() * self.gradient_accumulation_steps * valid_count

            # 定期清理缓存
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

        # 处理最后不完整的累积步骤
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            if self.use_amp:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        gc.collect()
        torch.cuda.empty_cache()

        return running_loss / total if total > 0 else 0

    def evaluate(self, model, dataloader, criterion):
        """评估 - 带内存优化"""
        model.eval()
        running_loss = 0.0

        all_preds = []
        all_labels = []
        patient_preds = defaultdict(list)
        patient_labels = defaultdict(list)

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
                ppg, labels = batch
                ppg = ppg.to(self.device)
                labels = labels.to(self.device)

                # 混合精度推理
                if self.use_amp:
                    with autocast():
                        outputs = model(ppg)
                        outputs = outputs.permute(0, 2, 1)

                        loss = criterion(
                            outputs.reshape(-1, 4),
                            labels.reshape(-1)
                        )
                else:
                    outputs = model(ppg)
                    outputs = outputs.permute(0, 2, 1)

                    loss = criterion(
                        outputs.reshape(-1, 4),
                        labels.reshape(-1)
                    )

                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    mask = labels[i] != -1
                    if mask.any():
                        patient_outputs = outputs[i][mask]
                        patient_labels_i = labels[i][mask]

                        _, predicted = patient_outputs.max(1)

                        patient_idx = batch_idx * dataloader.batch_size + i
                        patient_preds[patient_idx].extend(predicted.cpu().numpy())
                        patient_labels[patient_idx].extend(patient_labels_i.cpu().numpy())

                        all_preds.extend(predicted.cpu().numpy())
                        all_labels.extend(patient_labels_i.cpu().numpy())

                        running_loss += loss.item() * patient_labels_i.numel()

                # 定期清理缓存
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()

        # 计算指标
        overall_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        overall_kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        overall_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0

        # Per-patient指标
        patient_kappas = []
        for patient_idx in patient_preds:
            if len(set(patient_labels[patient_idx])) > 1:
                kappa = cohen_kappa_score(patient_labels[patient_idx], patient_preds[patient_idx])
                patient_kappas.append(kappa)

        median_kappa = np.median(patient_kappas) if patient_kappas else 0

        return {
            'loss': running_loss / len(all_labels) if all_labels else 0,
            'accuracy': overall_acc,
            'kappa': overall_kappa,
            'f1': overall_f1,
            'median_kappa': median_kappa,
            'all_preds': all_preds,
            'all_labels': all_labels
        }

    def train(self):
        """主训练流程"""
        print("\n" + "=" * 70)
        print("开始ABC双流PPG微调")
        print("=" * 70)

        # 创建数据集
        train_dataset = ABCDualPPGDataset(
            self.config['abc_ppg_file'],
            self.config['abc_index_file'],
            split='train',
            min_windows=self.config.get('min_windows', 600)
        )
        val_dataset = ABCDualPPGDataset(
            self.config['abc_ppg_file'],
            self.config['abc_index_file'],
            split='val',
            min_windows=self.config.get('min_windows', 600)
        )
        test_dataset = ABCDualPPGDataset(
            self.config['abc_ppg_file'],
            self.config['abc_index_file'],
            split='test',
            min_windows=self.config.get('min_windows', 600)
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'],
                                  shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'],
                                shuffle=False, num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'],
                                 shuffle=False, num_workers=0, pin_memory=True)

        # 加载模型
        model = self.load_pretrained_model()

        # 设置微调策略
        optimizer = self.setup_finetune_strategy(model)

        # 计算类别权重
        class_weights = self.calculate_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # 训练
        best_kappa = -1
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        patience_counter = 0

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_kappa': [],
            'val_accuracy': []
        }

        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print("-" * 50)

            # 渐进式解冻
            if self.config['strategy'] == 'progressive':
                if epoch == self.config.get('unfreeze_tcn_epoch', 5):
                    print("\n🔓 解冻TCN层和融合层")
                    self.unfreeze_layer(model, 'temporal_blocks')
                    self.unfreeze_layer(model, 'fusion_blocks')
                    self.unfreeze_layer(model, 'modality_weighting')
                    self.unfreeze_layer(model, 'feature_aggregation')

                    optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=self.config['learning_rate'] * 0.1,
                        weight_decay=self.config['weight_decay']
                    )

                elif epoch == self.config.get('unfreeze_all_epoch', 10):
                    print("\n🔓 解冻全部层")
                    for param in model.parameters():
                        param.requires_grad = True

                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config['learning_rate'] * 0.01,
                        weight_decay=self.config['weight_decay']
                    )

            # 训练
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            # 验证
            val_results = self.evaluate(model, val_loader, criterion)

            # 记录
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_results['loss'])
            history['val_kappa'].append(val_results['kappa'])
            history['val_accuracy'].append(val_results['accuracy'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")
            print(f"Val Acc: {val_results['accuracy']:.4f}, Kappa: {val_results['kappa']:.4f}, "
                  f"Median Kappa: {val_results['median_kappa']:.4f}")

            # 学习率调度
            scheduler.step(val_results['kappa'])

            # 保存最佳模型
            if val_results['kappa'] > best_kappa:
                best_kappa = val_results['kappa']
                patience_counter = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_kappa': best_kappa,
                    'config': self.config
                }, best_model_path)

                print(f"✅ 保存最佳模型 (Kappa: {best_kappa:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break

        # 测试最佳模型
        print("\n" + "=" * 70)
        print("测试最佳模型")
        print("=" * 70)

        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_results = self.evaluate(model, test_loader, criterion)

        print(f"\n测试结果:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Kappa: {test_results['kappa']:.4f}")
        print(f"  Median Kappa: {test_results['median_kappa']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(
            test_results['all_labels'],
            test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM']
        ))

        # 保存结果
        results = {
            'strategy': self.config['strategy'],
            'pretrained_path': self.config['pretrained_path'],
            'test_accuracy': float(test_results['accuracy']),
            'test_kappa': float(test_results['kappa']),
            'test_median_kappa': float(test_results['median_kappa']),
            'test_f1': float(test_results['f1']),
            'best_val_kappa': float(best_kappa),
            'history': history
        }

        with open(os.path.join(self.output_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)

        # 绘制混淆矩阵
        cm = confusion_matrix(test_results['all_labels'], test_results['all_preds'])
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                    yticklabels=['Wake', 'Light', 'Deep', 'REM'])
        plt.title(f'Confusion Matrix (Kappa: {test_results["kappa"]:.3f})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 绘制训练曲线
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(history['train_loss'], label='Train')
        axes[0].plot(history['val_loss'], label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].set_title('Loss Curve')

        axes[1].plot(history['val_kappa'], label='Kappa')
        axes[1].plot(history['val_accuracy'], label='Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].legend()
        axes[1].set_title('Validation Metrics')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ 结果已保存到: {self.output_dir}")

        return results


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='微调MESA预训练双流PPG模型到ABC')

    # 数据路径
    parser.add_argument('--abc_ppg_file', type=str,
                        default='../../data/abc_ppg_with_labels.h5')
    parser.add_argument('--abc_index_file', type=str,
                        default='../../data/abc_subject_index.h5')
    parser.add_argument('--pretrained_path', type=str,
                        default='./dual_best_model.pth',
                        help='MESA预训练双流模型路径')

    # 微调策略
    parser.add_argument('--strategy', type=str, default='discriminative',
                        choices=['full', 'head_only', 'progressive', 'discriminative'],
                        help='微调策略')

    # 训练参数
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='批次大小（建议1以节省内存）')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--min_windows', type=int, default=600,
                        help='最小windows数，少于此数的被试将被排除')

    # 内存优化参数
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数（有效batch_size = batch_size * steps）')
    parser.add_argument('--no_amp', action='store_true',
                        help='禁用混合精度训练')

    # 渐进式解冻参数
    parser.add_argument('--unfreeze_tcn_epoch', type=int, default=5)
    parser.add_argument('--unfreeze_all_epoch', type=int, default=10)

    # 输出
    parser.add_argument('--output_dir', type=str, default='./abc_dual_finetune_outputs')

    args = parser.parse_args()

    # 构建配置
    config = {
        'abc_ppg_file': args.abc_ppg_file,
        'abc_index_file': args.abc_index_file,
        'pretrained_path': args.pretrained_path,
        'strategy': args.strategy,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'min_windows': args.min_windows,
        'unfreeze_tcn_epoch': args.unfreeze_tcn_epoch,
        'unfreeze_all_epoch': args.unfreeze_all_epoch,
        'output_dir': args.output_dir,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'use_amp': not args.no_amp,
    }

    print("\n" + "=" * 70)
    print("MESA → ABC 双流PPG微调 (PPGUnfilteredCrossAttention)")
    print("=" * 70)
    print(f"\n配置:")
    print(f"  策略: {config['strategy']}")
    print(f"  学习率: {config['learning_rate']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  梯度累积: {config['gradient_accumulation_steps']} (有效batch={config['batch_size'] * config['gradient_accumulation_steps']})")
    print(f"  混合精度: {'启用' if config['use_amp'] else '禁用'}")
    print(f"  预训练模型: {config['pretrained_path']}")
    print(f"  ABC数据: {config['abc_ppg_file']}")
    print(f"  最小windows: {config['min_windows']}")

    # 开始微调
    finetuner = DualPPGFineTuner(config)
    results = finetuner.train()

    print("\n" + "=" * 70)
    print("微调完成!")
    print("=" * 70)


if __name__ == '__main__':
    main()
