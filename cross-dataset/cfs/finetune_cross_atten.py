"""
Fine-tuning EEG-PPG Cross-Attention Fusion Model on CFS/ABC
在CFS或ABC数据集上fine-tune MESA预训练的模型

用法:
python finetune_cross_dataset.py \
    --fusion_model ./outputs/cross_attention_fusion_weighted/best_model.pth \
    --eeg_model ../../work/mesa-short_window/outputs/eeg_window_3min/best_model.pth \
    --ppg_model ../../work/mesa-short_window/outputs/short_window/3min_2/best_model.pth \
    --dataset cfs \
    --output_dir ./outputs/cross_attention_finetune_cfs
"""

import os
import sys
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, confusion_matrix,
    classification_report, precision_recall_fscore_support
)
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '../../work/mesa-short_window')

from cross_attention_fusion import create_fusion_model


class CrossDatasetFusionDataset(Dataset):
    """CFS/ABC数据集"""

    def __init__(self,
                 eeg_folder: str,
                 ppg_h5_path: str,
                 ppg_index_path: str,
                 subject_ids: list = None,
                 window_minutes: int = 3,
                 samples_per_epoch: int = 1024,
                 dataset_name: str = 'cfs'):

        self.eeg_folder = eeg_folder
        self.ppg_h5_path = ppg_h5_path
        self.ppg_index_path = ppg_index_path
        self.window_minutes = window_minutes
        self.epochs_per_window = window_minutes * 2
        self.samples_per_epoch = samples_per_epoch
        self.dataset_name = dataset_name

        # 构建EEG文件路径映射
        self.eeg_file_map = {}
        if os.path.exists(eeg_folder):
            for fname in os.listdir(eeg_folder):
                if fname.endswith('.npz'):
                    sid = fname.replace('.npz', '')
                    # 处理不同格式的文件名
                    if dataset_name == 'abc':
                        # abc-baseline-900001 -> 900001_baseline
                        if 'baseline' in sid:
                            parts = sid.split('-')
                            num_id = parts[-1]  # 900001
                            sid = f"{num_id}_baseline"
                    elif '-' in sid:
                        # 其他格式如 cfs-visit5-0001 -> 0001
                        parts = sid.split('-')
                        sid = parts[-1]
                    self.eeg_file_map[sid] = os.path.join(eeg_folder, fname)

        # 加载PPG数据
        self.ppg_h5 = h5py.File(ppg_h5_path, 'r')
        self.ppg_data = self.ppg_h5['ppg']
        self.ppg_labels = self.ppg_h5['labels']

        # 加载PPG索引
        self.ppg_index_h5 = h5py.File(ppg_index_path, 'r')

        # 构建样本索引
        self.samples = []
        self._build_samples(subject_ids)

        print(f"Loaded {len(self.samples)} samples from {dataset_name.upper()}")

    def _build_samples(self, subject_ids=None):
        """构建样本索引"""
        if 'subjects' in self.ppg_index_h5:
            ppg_subjects = list(self.ppg_index_h5['subjects'].keys())
        else:
            ppg_subjects = []

        print(f"  PPG subjects in index: {len(ppg_subjects)}")
        print(f"  Sample PPG IDs: {ppg_subjects[:5]}")
        print(f"  Sample EEG IDs: {list(self.eeg_file_map.keys())[:5]}")

        # 如果指定了subject_ids，只使用这些
        if subject_ids is not None:
            # 对于ABC，subject_ids是 ['900001_baseline', ...] 格式
            ppg_subjects = [s for s in ppg_subjects if s in subject_ids]
            print(f"  Filtered to {len(ppg_subjects)} subjects based on subject_ids")

        matched_subjects = 0

        for sid in ppg_subjects:
            # 检查EEG数据 - sid已经是正确格式如 '900001_baseline'
            if sid not in self.eeg_file_map:
                continue

            eeg_path = self.eeg_file_map[sid]

            ppg_key = f'subjects/{sid}/window_indices'
            if ppg_key not in self.ppg_index_h5:
                continue

            matched_subjects += 1

            eeg_data = np.load(eeg_path)
            eeg_signals = eeg_data['x']
            eeg_labels = eeg_data['y']

            ppg_indices = self.ppg_index_h5[ppg_key][:]
            n_epochs = min(len(eeg_signals), len(ppg_indices))

            mapped_labels = []
            for label in eeg_labels[:n_epochs]:
                if isinstance(label, bytes):
                    label = label.decode()
                label_int = int(label)

                if self.dataset_name == 'abc':
                    # ABC已经是4类格式: 0=Wake, 1=Light, 2=Deep, 3=REM
                    if label_int in [0, 1, 2, 3]:
                        mapped_labels.append(label_int)
                    else:
                        mapped_labels.append(-1)
                else:
                    # MESA/CFS是5类格式: 0=Wake, 1=N1, 2=N2, 3=N3, 4/5=REM
                    if label_int == 0:
                        mapped_labels.append(0)  # Wake
                    elif label_int in [1, 2]:
                        mapped_labels.append(1)  # Light
                    elif label_int == 3:
                        mapped_labels.append(2)  # Deep
                    elif label_int in [4, 5]:
                        mapped_labels.append(3)  # REM
                    else:
                        mapped_labels.append(-1)
            mapped_labels = np.array(mapped_labels)

            n_windows = n_epochs // self.epochs_per_window

            for win_idx in range(n_windows):
                start_epoch = win_idx * self.epochs_per_window
                end_epoch = start_epoch + self.epochs_per_window

                window_labels = mapped_labels[start_epoch:end_epoch]
                if np.any(window_labels == -1):
                    continue

                self.samples.append({
                    'subject_id': sid,
                    'eeg_path': eeg_path,
                    'eeg_start': start_epoch,
                    'ppg_indices': ppg_indices[start_epoch:end_epoch],
                    'labels': window_labels
                })

        print(f"  Matched subjects: {matched_subjects}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        eeg_data = np.load(sample['eeg_path'])
        eeg_signals = eeg_data['x']
        start = sample['eeg_start']
        end = start + self.epochs_per_window
        eeg_window = eeg_signals[start:end]
        eeg_window = (eeg_window - eeg_window.mean()) / (eeg_window.std() + 1e-8)

        ppg_indices = sample['ppg_indices']
        ppg_epochs = self.ppg_data[ppg_indices]
        ppg_window = ppg_epochs.flatten()
        ppg_window = (ppg_window - ppg_window.mean()) / (ppg_window.std() + 1e-8)

        labels = sample['labels']

        return {
            'eeg': torch.FloatTensor(eeg_window),
            'ppg': torch.FloatTensor(ppg_window),
            'labels': torch.LongTensor(labels),
            'subject_id': sample['subject_id']
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

        optimizer.zero_grad()
        outputs = model(eeg, ppg)

        outputs_flat = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)
        loss = criterion(outputs_flat, labels_flat)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = outputs.argmax(dim=-1).cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()
        all_preds.extend(preds)
        all_labels.extend(labels_np)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

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

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    report = classification_report(all_labels, all_preds,
                                   labels=[0, 1, 2, 3],
                                   target_names=['Wake', 'Light', 'Deep', 'REM'],
                                   output_dict=True,
                                   zero_division=0)

    return avg_loss, accuracy, kappa, cm, report


def plot_confusion_matrix(cm, class_names, output_path, title):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))

    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def measure_inference_time(model, device, n_runs=100):
    """测量推理时间"""
    model.eval()
    dummy_eeg = torch.randn(1, 6, 3000).to(device)
    dummy_ppg = torch.randn(1, 6144).to(device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_eeg, dummy_ppg)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()
            _ = model(dummy_eeg, dummy_ppg)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - start_time) * 1000)

    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_model', type=str, required=True,
                        help='Path to pretrained fusion model (from MESA)')
    parser.add_argument('--eeg_model', type=str, required=True,
                        help='Path to pretrained EEG model')
    parser.add_argument('--ppg_model', type=str, required=True,
                        help='Path to pretrained PPG model')
    parser.add_argument('--dataset', type=str, required=True, choices=['cfs', 'abc'],
                        help='Dataset to fine-tune on')

    # CFS paths
    parser.add_argument('--cfs_eeg_folder', type=str,
                        default='../../data/cfs_eeg_c3m2_data')
    parser.add_argument('--cfs_ppg_h5', type=str,
                        default='../../data/cfs_ppg_with_labels.h5')
    parser.add_argument('--cfs_ppg_index', type=str,
                        default='../../data/cfs_subject_index.h5')

    # ABC paths
    parser.add_argument('--abc_eeg_folder', type=str,
                        default='../../data/eeg')
    parser.add_argument('--abc_ppg_h5', type=str,
                        default='../../data/abc_ppg_with_labels.h5')
    parser.add_argument('--abc_ppg_index', type=str,
                        default='../../data/abc_subject_index.h5')

    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Ratio of data for training')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 70)
    print(f"Fine-tuning Cross-Attention Fusion on {args.dataset.upper()}")
    print("=" * 70)

    # 选择数据路径
    if args.dataset == 'cfs':
        eeg_folder = args.cfs_eeg_folder
        ppg_h5 = args.cfs_ppg_h5
        ppg_index = args.cfs_ppg_index
    else:
        eeg_folder = args.abc_eeg_folder
        ppg_h5 = args.abc_ppg_h5
        ppg_index = args.abc_ppg_index

    # ==================== 加载模型 ====================
    print("\n[1/5] Loading models...")
    model = create_fusion_model(
        eeg_model_path=args.eeg_model,
        ppg_model_path=args.ppg_model,
        device=args.device,
        n_fusion_blocks=2,
        freeze_encoders=True
    )

    # 加载MESA预训练的融合层权重
    checkpoint = torch.load(args.fusion_model, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded pretrained fusion model from {args.fusion_model}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # ==================== 加载数据并划分 ====================
    print(f"\n[2/5] Loading {args.dataset.upper()} dataset...")

    # 获取所有subjects
    ppg_index_h5 = h5py.File(ppg_index, 'r')
    all_subjects = list(ppg_index_h5['subjects'].keys())
    ppg_index_h5.close()

    # 划分train/val/test
    np.random.seed(42)
    np.random.shuffle(all_subjects)

    n_train = int(len(all_subjects) * args.train_ratio)
    n_val = int(len(all_subjects) * 0.1)

    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train:n_train + n_val]
    test_subjects = all_subjects[n_train + n_val:]

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Val subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")

    train_dataset = CrossDatasetFusionDataset(
        eeg_folder=eeg_folder,
        ppg_h5_path=ppg_h5,
        ppg_index_path=ppg_index,
        subject_ids=train_subjects,
        window_minutes=3,
        dataset_name=args.dataset
    )

    val_dataset = CrossDatasetFusionDataset(
        eeg_folder=eeg_folder,
        ppg_h5_path=ppg_h5,
        ppg_index_path=ppg_index,
        subject_ids=val_subjects,
        window_minutes=3,
        dataset_name=args.dataset
    )

    test_dataset = CrossDatasetFusionDataset(
        eeg_folder=eeg_folder,
        ppg_h5_path=ppg_h5,
        ppg_index_path=ppg_index,
        subject_ids=test_subjects,
        window_minutes=3,
        dataset_name=args.dataset
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    # ==================== 计算类别权重 ====================
    print("\n[3/5] Calculating class weights...")
    class_counts = [0, 0, 0, 0]
    for sample in train_dataset.samples:
        for label in sample['labels']:
            class_counts[label] += 1

    total = sum(class_counts)
    class_weights = [np.sqrt(total / (4 * c)) if c > 0 else 1.0 for c in class_counts]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)

    print(f"  Class distribution:")
    class_names = ['Wake', 'Light', 'Deep', 'REM']
    for i, name in enumerate(class_names):
        pct = class_counts[i] / total * 100 if total > 0 else 0
        print(f"    {name}: {class_counts[i]:>8} ({pct:.1f}%) -> weight: {class_weights[i]:.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 只训练融合层（encoder已冻结）
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params_list, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ==================== 训练 ====================
    print("\n[4/5] Fine-tuning...")
    best_kappa = -1
    patience_counter = 0
    history = {'train_loss': [], 'train_kappa': [], 'val_loss': [], 'val_kappa': []}

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc, train_kappa = train_epoch(
            model, train_loader, criterion, optimizer, args.device
        )

        val_loss, val_acc, val_kappa, val_cm, val_report = validate(
            model, val_loader, criterion, args.device
        )

        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_kappa'].append(train_kappa)
        history['val_loss'].append(val_loss)
        history['val_kappa'].append(val_kappa)

        deep_recall = val_report['Deep']['recall']

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")
        print(f"        Deep Recall: {deep_recall:.4f}")

        if val_kappa > best_kappa:
            best_kappa = val_kappa
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kappa': val_kappa,
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))

            print(f"  -> New best model saved! (Kappa: {val_kappa:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # ==================== 测试 ====================
    print("\n[5/5] Evaluating on test set...")

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # 推理时间
    inference_mean, inference_std = measure_inference_time(model, args.device)
    print(f"Inference time: {inference_mean:.2f} ms ± {inference_std:.2f} ms")

    # 测试
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            eeg = batch['eeg'].to(args.device)
            ppg = batch['ppg'].to(args.device)
            labels = batch['labels']

            outputs = model(eeg, ppg)
            preds = outputs.argmax(dim=-1).cpu().numpy()

            for i in range(len(preds)):
                all_preds.extend(preds[i])
                all_labels.extend(labels[i].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    accuracy = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2, 3]
    )
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]

    # 打印结果
    print("\n" + "=" * 70)
    print(f"TEST RESULTS - {args.dataset.upper()} (Fine-tuned)")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nPer-class metrics:")
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 56)
    for i, name in enumerate(class_names):
        print(f"{name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")

    print("\nConfusion Matrix:")
    print(cm)

    # 保存结果
    results = {
        'dataset': args.dataset,
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'macro_f1': float(macro_f1),
        'per_class': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i in range(4)
        },
        'confusion_matrix': cm.tolist(),
        'total_samples': len(labels),
        'inference_time_ms': {
            'mean': round(inference_mean, 2),
            'std': round(inference_std, 2)
        }
    }

    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    plot_confusion_matrix(
        cm, class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png'),
        title=f'Cross-Attention Fusion - {args.dataset.upper()} (Fine-tuned)'
    )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()