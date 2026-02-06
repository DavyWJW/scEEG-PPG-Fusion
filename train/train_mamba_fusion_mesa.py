"""
Train and Evaluate EEG-PPG Cross-Attention + Mamba TCM Fusion Model on MESA

用法:
python train_mamba_fusion_mesa.py \
    --eeg_model ../../work/mesa-short_window/outputs/eeg_window_3min/best_model.pth \
    --ppg_model ../../work/mesa-short_window/outputs/short_window/3min_2/best_model.pth \
    --output_dir ./outputs/cross_attention_mamba_fusion
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

sys.path.insert(0, '.')

from cross_attention_mamba_fusion import create_mamba_fusion_model


class FusionDataset(Dataset):
    """EEG-PPG融合数据集"""
    
    def __init__(self,
                 eeg_folders: list,
                 ppg_h5_path: str,
                 ppg_index_path: str,
                 subject_ids: list,
                 window_minutes: int = 3,
                 samples_per_epoch: int = 1024):
        self.eeg_folders = eeg_folders
        self.ppg_h5_path = ppg_h5_path
        self.ppg_index_path = ppg_index_path
        self.window_minutes = window_minutes
        self.epochs_per_window = window_minutes * 2
        self.samples_per_epoch = samples_per_epoch
        
        # 构建EEG文件映射（多个文件夹）
        self.eeg_file_map = {}
        for folder in eeg_folders:
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    if fname.endswith('.npz'):
                        sid = fname.replace('.npz', '')
                        if sid.startswith('mesa-sleep-'):
                            sid = sid.replace('mesa-sleep-', '')
                        self.eeg_file_map[sid] = os.path.join(folder, fname)
        
        self.ppg_h5 = h5py.File(ppg_h5_path, 'r')
        self.ppg_data = self.ppg_h5['ppg']
        self.ppg_labels = self.ppg_h5['labels']
        self.ppg_index_h5 = h5py.File(ppg_index_path, 'r')
        
        self.samples = []
        self._build_samples(subject_ids)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _build_samples(self, subject_ids):
        matched = 0
        for sid in subject_ids:
            sid_str = str(sid).zfill(4) if isinstance(sid, int) else sid
            
            if sid_str not in self.eeg_file_map:
                continue
            eeg_path = self.eeg_file_map[sid_str]
            
            ppg_key = f'subjects/{sid_str}/window_indices'
            if ppg_key not in self.ppg_index_h5:
                continue
            
            matched += 1
            
            eeg_data = np.load(eeg_path)
            eeg_signals = eeg_data['x']
            eeg_labels = eeg_data['y']
            ppg_indices = self.ppg_index_h5[ppg_key][:]
            n_epochs = min(len(eeg_signals), len(ppg_indices))
            
            mapped_labels = []
            for label in eeg_labels[:n_epochs]:
                if isinstance(label, bytes):
                    label = label.decode()
                label = str(label)
                if label == '0':
                    mapped_labels.append(0)
                elif label in ['1', '2']:
                    mapped_labels.append(1)
                elif label == '3':
                    mapped_labels.append(2)
                elif label in ['4', '5']:
                    mapped_labels.append(3)
                else:
                    mapped_labels.append(-1)
            mapped_labels = np.array(mapped_labels)
            
            n_windows = n_epochs // self.epochs_per_window
            for win_idx in range(n_windows):
                start = win_idx * self.epochs_per_window
                end = start + self.epochs_per_window
                window_labels = mapped_labels[start:end]
                if np.any(window_labels == -1):
                    continue
                self.samples.append({
                    'subject_id': sid_str,
                    'eeg_path': eeg_path,
                    'eeg_start': start,
                    'ppg_indices': ppg_indices[start:end],
                    'labels': window_labels
                })
        print(f"  Matched subjects: {matched}")
    
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
        
        ppg_epochs = self.ppg_data[sample['ppg_indices']]
        ppg_window = ppg_epochs.flatten()
        ppg_window = (ppg_window - ppg_window.mean()) / (ppg_window.std() + 1e-8)
        
        return {
            'eeg': torch.FloatTensor(eeg_window),
            'ppg': torch.FloatTensor(ppg_window),
            'labels': torch.LongTensor(sample['labels']),
            'subject_id': sample['subject_id']
        }
    
    def __del__(self):
        if hasattr(self, 'ppg_h5'):
            self.ppg_h5.close()
        if hasattr(self, 'ppg_index_h5'):
            self.ppg_index_h5.close()


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for batch in tqdm(loader, desc="Training"):
        eeg = batch['eeg'].to(device)
        ppg = batch['ppg'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(eeg, ppg)
        loss = criterion(outputs.view(-1, 4), labels.view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend(outputs.argmax(-1).cpu().numpy().flatten())
        all_labels.extend(labels.cpu().numpy().flatten())
    
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    return total_loss / len(loader), acc, kappa


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            eeg = batch['eeg'].to(device)
            ppg = batch['ppg'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(eeg, ppg)
            loss = criterion(outputs.view(-1, 4), labels.view(-1))
            
            total_loss += loss.item()
            all_preds.extend(outputs.argmax(-1).cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    report = classification_report(all_labels, all_preds, labels=[0,1,2,3],
                                   target_names=['Wake','Light','Deep','REM'],
                                   output_dict=True, zero_division=0)
    return total_loss / len(loader), acc, kappa, cm, report


def plot_confusion_matrix(cm, output_path, title):
    plt.figure(figsize=(10, 8))
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(4):
        for j in range(4):
            annot[i, j] = f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Wake','Light','Deep','REM'],
                yticklabels=['Wake','Light','Deep','REM'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def measure_inference_time(model, device, n_runs=100):
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
            t0 = time.time()
            _ = model(dummy_eeg, dummy_ppg)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append((time.time() - t0) * 1000)
    
    return np.mean(times), np.std(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eeg_model', type=str, default='../../work/mesa-short_window/outputs/eeg_window_3min/best_model.pth')
    parser.add_argument('--ppg_model', type=str, default='../../work/mesa-short_window/outputs/short_window/3min_2/best_model.pth')
    parser.add_argument('--eeg_folders', nargs='+', 
                        default=['../../data/eeg-1', '../../data/eeg-2', '../../data/eeg-3'],
                        help='EEG data folders')
    parser.add_argument('--ppg_h5_path', default='../../data/mesa_ppg_with_labels.h5')
    parser.add_argument('--ppg_index_path', default='../../data/mesa_subject_index.h5')
    parser.add_argument('--testset_json', default='./testset.json')
    parser.add_argument('--output_dir', default='./outputs/cross_attention_mamba_fusion')
    parser.add_argument('--n_fusion_blocks', type=int, default=2)
    parser.add_argument('--n_mamba_layers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 70)
    print("Cross-Attention + Mamba TCM Fusion - MESA")
    print("=" * 70)
    
    # 创建模型
    print("\n[1/5] Creating model...")
    model = create_mamba_fusion_model(
        args.eeg_model, args.ppg_model, args.device,
        n_fusion_blocks=args.n_fusion_blocks,
        n_mamba_layers=args.n_mamba_layers,
        freeze_encoders=True
    )
    
    # 加载数据
    print("\n[2/5] Loading data...")
    with open(args.testset_json) as f:
        test_subjects = json.load(f)
    test_set = set(test_subjects)
    
    # 从多个EEG文件夹获取所有subjects
    all_subjects = []
    for folder in args.eeg_folders:
        if os.path.exists(folder):
            for fname in os.listdir(folder):
                if fname.endswith('.npz'):
                    sid = fname.replace('.npz', '').replace('mesa-sleep-', '')
                    if sid not in test_set and sid not in all_subjects:
                        all_subjects.append(sid)
    
    np.random.seed(42)
    np.random.shuffle(all_subjects)
    n_val = max(1, int(len(all_subjects) * 0.1))
    val_subjects = all_subjects[:n_val]
    train_subjects = all_subjects[n_val:]
    
    print(f"Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")
    
    train_dataset = FusionDataset(args.eeg_folders, args.ppg_h5_path, args.ppg_index_path, train_subjects)
    val_dataset = FusionDataset(args.eeg_folders, args.ppg_h5_path, args.ppg_index_path, val_subjects)
    test_dataset = FusionDataset(args.eeg_folders, args.ppg_h5_path, args.ppg_index_path, test_subjects)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 类别权重
    print("\n[3/5] Calculating class weights...")
    class_counts = [0, 0, 0, 0]
    for s in train_dataset.samples:
        for l in s['labels']:
            class_counts[l] += 1
    total = sum(class_counts)
    class_weights = torch.tensor([np.sqrt(total/(4*c)) for c in class_counts], dtype=torch.float).to(args.device)
    print(f"  Wake: {class_counts[0]} ({class_counts[0]/total*100:.1f}%) w={class_weights[0]:.2f}")
    print(f"  Light: {class_counts[1]} ({class_counts[1]/total*100:.1f}%) w={class_weights[1]:.2f}")
    print(f"  Deep: {class_counts[2]} ({class_counts[2]/total*100:.1f}%) w={class_weights[2]:.2f}")
    print(f"  REM: {class_counts[3]} ({class_counts[3]/total*100:.1f}%) w={class_weights[3]:.2f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 训练
    print("\n[4/5] Training...")
    best_kappa = -1
    patience_cnt = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)
        
        train_loss, train_acc, train_kappa = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, val_kappa, val_cm, val_report = validate(model, val_loader, criterion, args.device)
        scheduler.step()
        
        deep_recall = val_report['Deep']['recall']
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}, Deep Recall: {deep_recall:.4f}")
        
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            patience_cnt = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_kappa': val_kappa
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"  -> New best! (Kappa: {val_kappa:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # 测试
    print("\n[5/5] Testing...")
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(ckpt['model_state_dict'])
    
    inference_mean, inference_std = measure_inference_time(model, args.device)
    print(f"Inference time: {inference_mean:.2f} ± {inference_std:.2f} ms")
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            eeg = batch['eeg'].to(args.device)
            ppg = batch['ppg'].to(args.device)
            outputs = model(eeg, ppg)
            all_preds.extend(outputs.argmax(-1).cpu().numpy().flatten())
            all_labels.extend(batch['labels'].numpy().flatten())
    
    preds, labels = np.array(all_preds), np.array(all_labels)
    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0,1,2,3])
    prec, rec, f1, supp = precision_recall_fscore_support(labels, preds, labels=[0,1,2,3], zero_division=0)
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[2]
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Kappa: {kappa:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"\n{'Class':<10} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 50)
    for i, name in enumerate(['Wake','Light','Deep','REM']):
        print(f"{name:<10} {prec[i]:<10.4f} {rec[i]:<10.4f} {f1[i]:<10.4f} {supp[i]:<10}")
    
    # 保存结果
    params = model.get_trainable_params()
    results = {
        'accuracy': float(acc),
        'kappa': float(kappa),
        'macro_f1': float(macro_f1),
        'per_class': {name: {'precision': float(prec[i]), 'recall': float(rec[i]), 'f1': float(f1[i]), 'support': int(supp[i])}
                      for i, name in enumerate(['Wake','Light','Deep','REM'])},
        'confusion_matrix': cm.tolist(),
        'model_params': params,
        'inference_time_ms': {'mean': round(inference_mean, 2), 'std': round(inference_std, 2)}
    }
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    plot_confusion_matrix(cm, os.path.join(args.output_dir, 'confusion_matrix.png'),
                          'Cross-Attention + Mamba TCM Fusion - MESA Test')
    
    # 对比
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"{'Method':<35} {'Kappa':<10} {'Params':<12} {'Inference':<15}")
    print("-" * 70)
    print(f"{'Cross-Attention (no Mamba)':<35} {'0.794':<10} {'14.5M':<12} {'14.5ms':<15}")
    print(f"{'Cross-Attention + Mamba TCM':<35} {kappa:<10.4f} {params['total']/1e6:<12.2f}M {inference_mean:<.2f}±{inference_std:<.2f}ms")
    print("-" * 70)
    
    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()
