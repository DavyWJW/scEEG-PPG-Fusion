"""
Train EEG-PPG Cross-Attention Fusion Model

用法:
python train_cross_attention_fusion.py \
    --eeg_model ../../work/mesa-short_window/outputs/eeg_window_3min/best_model.pth \
    --ppg_model ../../work/mesa-short_window/outputs/short_window/3min_2/best_model.pth \
    --output_dir ./outputs/cross_attention_fusion
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import h5py

# 添加模型路径
sys.path.insert(0, '../../work/mesa-short_window')

from cross_attention_fusion import create_fusion_model, EEGPPGCrossAttentionFusion


class FusionDataset(Dataset):
    """EEG-PPG融合数据集"""

    def __init__(self,
                 eeg_folders: list,
                 ppg_h5_path: str,
                 ppg_index_path: str,
                 subject_ids: list,
                 window_minutes: int = 3,
                 samples_per_epoch: int = 1024):
        """
        Args:
            eeg_folders: EEG npz文件目录列表
            ppg_h5_path: PPG数据H5文件路径
            ppg_index_path: PPG索引H5文件路径
            subject_ids: 使用的subject ID列表
            window_minutes: 窗口长度（分钟）
            samples_per_epoch: 每个epoch的PPG采样数
        """
        self.eeg_folders = eeg_folders
        self.ppg_h5_path = ppg_h5_path
        self.ppg_index_path = ppg_index_path
        self.window_minutes = window_minutes
        self.epochs_per_window = window_minutes * 2  # 每30秒一个epoch
        self.samples_per_epoch = samples_per_epoch

        # 构建EEG文件路径映射
        self.eeg_file_map = {}
        for folder in eeg_folders:
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    if fname.endswith('.npz'):
                        # 从文件名提取subject ID
                        # 格式: mesa-sleep-0679.npz -> 0679
                        sid = fname.replace('.npz', '')
                        if sid.startswith('mesa-sleep-'):
                            sid = sid.replace('mesa-sleep-', '')
                        self.eeg_file_map[sid] = os.path.join(folder, fname)

        # 加载PPG数据
        self.ppg_h5 = h5py.File(ppg_h5_path, 'r')
        self.ppg_data = self.ppg_h5['ppg']
        self.ppg_labels = self.ppg_h5['labels']

        # 加载PPG索引
        self.ppg_index_h5 = h5py.File(ppg_index_path, 'r')

        # 构建样本索引
        self.samples = []
        self._build_samples(subject_ids)

        print(f"Loaded {len(self.samples)} samples from {len(subject_ids)} subjects")

    def _build_samples(self, subject_ids):
        """构建训练样本索引"""
        matched_subjects = 0
        no_eeg = 0
        no_ppg = 0

        for sid in subject_ids:
            sid_str = str(sid).zfill(4) if isinstance(sid, int) else sid

            # 检查EEG数据
            if sid_str not in self.eeg_file_map:
                no_eeg += 1
                continue
            eeg_path = self.eeg_file_map[sid_str]

            # 检查PPG索引数据
            ppg_key = f'subjects/{sid_str}/window_indices'
            if ppg_key not in self.ppg_index_h5:
                no_ppg += 1
                continue

            matched_subjects += 1

            # 加载EEG数据
            eeg_data = np.load(eeg_path)
            eeg_signals = eeg_data['x']  # (n_epochs, signal_length)
            eeg_labels = eeg_data['y']  # (n_epochs,)

            # 加载PPG索引
            ppg_indices = self.ppg_index_h5[ppg_key][:]

            # 确保数据长度匹配
            n_epochs = min(len(eeg_signals), len(ppg_indices))

            # 标签映射 (EEG: '0','1','2','3','4' -> 0,1,1,2,3)
            mapped_labels = []
            for label in eeg_labels[:n_epochs]:
                if isinstance(label, bytes):
                    label = label.decode()
                label = str(label)
                if label == '0':
                    mapped_labels.append(0)  # Wake
                elif label in ['1', '2']:
                    mapped_labels.append(1)  # Light
                elif label == '3':
                    mapped_labels.append(2)  # Deep
                elif label in ['4', '5']:
                    mapped_labels.append(3)  # REM
                else:
                    mapped_labels.append(-1)  # Invalid
            mapped_labels = np.array(mapped_labels)

            # 创建窗口样本
            n_windows = n_epochs // self.epochs_per_window

            for win_idx in range(n_windows):
                start_epoch = win_idx * self.epochs_per_window
                end_epoch = start_epoch + self.epochs_per_window

                # 检查标签有效性
                window_labels = mapped_labels[start_epoch:end_epoch]
                if np.any(window_labels == -1):
                    continue

                self.samples.append({
                    'subject_id': sid_str,
                    'eeg_path': eeg_path,
                    'eeg_start': start_epoch,
                    'ppg_indices': ppg_indices[start_epoch:end_epoch],
                    'labels': window_labels
                })

        print(f"  Subject matching: {matched_subjects} matched, {no_eeg} missing EEG, {no_ppg} missing PPG")
        print(f"  EEG subjects available: {len(self.eeg_file_map)}")
        print(f"  PPG subjects in index: {len([k for k in self.ppg_index_h5.keys() if k.startswith('subjects/')])}")

        # 打印一些样例ID来调试
        if len(self.samples) == 0:
            print(f"  Sample EEG IDs: {list(self.eeg_file_map.keys())[:5]}")
            print(
                f"  Sample PPG keys: {list(self.ppg_index_h5['subjects'].keys())[:5] if 'subjects' in self.ppg_index_h5 else 'No subjects key'}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载EEG数据
        eeg_data = np.load(sample['eeg_path'])
        eeg_signals = eeg_data['x']
        start = sample['eeg_start']
        end = start + self.epochs_per_window
        eeg_window = eeg_signals[start:end]  # (n_epochs, signal_length)

        # Z-score标准化
        eeg_window = (eeg_window - eeg_window.mean()) / (eeg_window.std() + 1e-8)

        # 加载PPG数据
        ppg_indices = sample['ppg_indices']
        ppg_epochs = self.ppg_data[ppg_indices]  # (n_epochs, samples_per_epoch)
        ppg_window = ppg_epochs.flatten()  # (n_epochs * samples_per_epoch,)

        # Z-score标准化
        ppg_window = (ppg_window - ppg_window.mean()) / (ppg_window.std() + 1e-8)

        # 标签
        labels = sample['labels']

        return {
            'eeg': torch.FloatTensor(eeg_window),
            'ppg': torch.FloatTensor(ppg_window),
            'labels': torch.LongTensor(labels)
        }

    def __del__(self):
        if hasattr(self, 'ppg_h5') and self.ppg_h5:
            self.ppg_h5.close()
        if hasattr(self, 'ppg_index_h5') and self.ppg_index_h5:
            self.ppg_index_h5.close()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        eeg = batch['eeg'].to(device)
        ppg = batch['ppg'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(eeg, ppg)  # (batch, n_epochs, n_classes)

        # 计算损失
        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 收集预测
        preds = outputs.argmax(dim=-1).cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels_np)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # 计算指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    return avg_loss, accuracy, kappa


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            eeg = batch['eeg'].to(device)
            ppg = batch['ppg'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(eeg, ppg)

            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            loss = criterion(outputs_flat, labels_flat)

            total_loss += loss.item()

            preds = outputs.argmax(dim=-1).cpu().numpy().flatten()
            labels_np = labels.cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(labels_np)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    # 详细报告
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds,
                                   target_names=['Wake', 'Light', 'Deep', 'REM'],
                                   output_dict=True)

    return avg_loss, accuracy, kappa, cm, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_model', type=str, required=True,
                        help='Path to pretrained EEG model')
    parser.add_argument('--ppg_model', type=str, required=True,
                        help='Path to pretrained PPG model')
    parser.add_argument('--eeg_folders', type=str, nargs='+',
                        default=['../../data/eeg-1', '../../data/eeg-2', '../../data/eeg-3'],
                        help='EEG data directories')
    parser.add_argument('--ppg_h5_path', type=str,
                        default='../../data/mesa_ppg_with_labels.h5',
                        help='PPG H5 file path')
    parser.add_argument('--ppg_index_path', type=str,
                        default='../../data/mesa_subject_index.h5',
                        help='PPG subject index H5 file path')
    parser.add_argument('--testset_json', type=str,
                        default='./testset.json',
                        help='Test set subject IDs JSON file')
    parser.add_argument('--output_dir', type=str,
                        default='./outputs/cross_attention_fusion',
                        help='Output directory')
    parser.add_argument('--n_fusion_blocks', type=int, default=2,
                        help='Number of cross-attention fusion blocks')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 60)
    print("EEG-PPG Cross-Attention Fusion Training")
    print("=" * 60)

    # 创建模型
    print("\nLoading pretrained models and creating fusion model...")
    model = create_fusion_model(
        eeg_model_path=args.eeg_model,
        ppg_model_path=args.ppg_model,
        device=args.device,
        n_fusion_blocks=args.n_fusion_blocks,
        freeze_encoders=True
    )

    # 加载数据
    print("\nLoading datasets...")

    # 加载test subjects
    with open(args.testset_json, 'r') as f:
        test_subjects = json.load(f)
    test_subjects_set = set(test_subjects)

    # 获取所有可用的subjects（从多个EEG数据目录）
    all_subjects = []
    for eeg_folder in args.eeg_folders:
        if os.path.exists(eeg_folder):
            for fname in os.listdir(eeg_folder):
                if fname.endswith('.npz'):
                    # 从文件名提取subject ID
                    # 格式: mesa-sleep-0679.npz -> 0679
                    sid = fname.replace('.npz', '')
                    if sid.startswith('mesa-sleep-'):
                        sid = sid.replace('mesa-sleep-', '')
                    if sid not in test_subjects_set and sid not in all_subjects:
                        all_subjects.append(sid)

    # 划分train/val (90%/10%)
    np.random.seed(42)
    np.random.shuffle(all_subjects)
    n_val = max(1, int(len(all_subjects) * 0.1))
    val_subjects = all_subjects[:n_val]
    train_subjects = all_subjects[n_val:]

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")

    train_dataset = FusionDataset(
        eeg_folders=args.eeg_folders,
        ppg_h5_path=args.ppg_h5_path,
        ppg_index_path=args.ppg_index_path,
        subject_ids=train_subjects,
        window_minutes=3
    )

    val_dataset = FusionDataset(
        eeg_folders=args.eeg_folders,
        ppg_h5_path=args.ppg_h5_path,
        ppg_index_path=args.ppg_index_path,
        subject_ids=val_subjects,
        window_minutes=3
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()

    # 只优化可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练循环
    best_kappa = -1
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'train_kappa': [],
               'val_loss': [], 'val_acc': [], 'val_kappa': []}

    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        # 训练
        train_loss, train_acc, train_kappa = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        # 验证
        val_loss, val_acc, val_kappa, cm, report = validate(
            model, val_loader, criterion, args.device
        )

        scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_kappa'].append(val_kappa)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")

        # 保存最佳模型
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_acc': val_acc,
                'confusion_matrix': cm.tolist(),
                'classification_report': report
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"  -> New best model saved! (Kappa: {val_kappa:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # 保存训练历史
    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training completed! Best validation Kappa: {best_kappa:.4f}")
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()