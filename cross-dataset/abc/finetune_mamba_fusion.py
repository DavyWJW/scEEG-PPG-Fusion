"""
Fine-tune Cross-Attention + Mamba TCM Fusion Model on CFS/ABC


python finetune_mamba_fusion.py --dataset cfs --fusion_model ./outputs/mamba_fusion/best_model.pth
python finetune_mamba_fusion.py --dataset abc --fusion_model ./outputs/mamba_fusion/best_model.pth
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

from cross_attention_mamba_fusion import EEGPPGCrossAttentionMambaFusion, MambaTCM


DATASET_CONFIG = {
    'cfs': {
        'ppg_data': '../../data/cfs_ppg_with_labels.h5',
        'ppg_index': '../../data/cfs_subject_index.h5',
        'eeg_folder': '../../data/cfs_eeg_c3m2_data',
        'label_format': '5class'  # 0=W, 1=N1, 2=N2, 3=N3, 4/5=REM
    },
    'abc': {
        'ppg_data': '../../data/abc_ppg_with_labels.h5',
        'ppg_index': '../../data/abc_subject_index.h5',
        'eeg_folder': '../../data/eeg',
        'label_format': '4class'  # 0=W, 1=Light, 2=Deep, 3=REM
    }
}


class CrossDatasetFusion(Dataset):

    def __init__(self,
                 eeg_folder: str,
                 ppg_h5_path: str,
                 ppg_index_path: str,
                 subject_ids: list,
                 dataset_name: str = 'cfs',
                 window_minutes: int = 3,
                 samples_per_epoch: int = 1024):
        self.eeg_folder = eeg_folder
        self.dataset_name = dataset_name
        self.window_minutes = window_minutes
        self.epochs_per_window = window_minutes * 2
        self.samples_per_epoch = samples_per_epoch


        self.eeg_file_map = self._build_eeg_file_map(eeg_folder)


        self.ppg_h5 = h5py.File(ppg_h5_path, 'r')
        self.ppg_data = self.ppg_h5['ppg']
        self.ppg_index_h5 = h5py.File(ppg_index_path, 'r')

        self.samples = []
        self._build_samples(subject_ids)

        print(f"  [{dataset_name.upper()}] Loaded {len(self.samples)} samples from {len(subject_ids)} subjects")

    def _build_eeg_file_map(self, eeg_folder):

        eeg_map = {}
        if not os.path.exists(eeg_folder):
            print(f"Warning: EEG folder not found: {eeg_folder}")
            return eeg_map

        for fname in os.listdir(eeg_folder):
            if not fname.endswith('.npz'):
                continue

            if self.dataset_name == 'abc':
                #  abc-baseline-900001.npz -> 900001_baseline
                if fname.startswith('abc-baseline-'):
                    subject_num = fname.replace('abc-baseline-', '').replace('.npz', '')
                    sid = f"{subject_num}_baseline"
                    eeg_map[sid] = os.path.join(eeg_folder, fname)
            else:

                sid = fname.replace('.npz', '')
                eeg_map[sid] = os.path.join(eeg_folder, fname)

        return eeg_map

    def _build_samples(self, subject_ids):

        matched = 0

        for sid in subject_ids:

            if sid not in self.eeg_file_map:
                continue
            eeg_path = self.eeg_file_map[sid]


            ppg_key = f'subjects/{sid}/window_indices'
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
                label_int = int(label)

                if self.dataset_name == 'abc':

                    if label_int in [0, 1, 2, 3]:
                        mapped_labels.append(label_int)
                    else:
                        mapped_labels.append(-1)
                else:

                    if label_int == 0:
                        mapped_labels.append(0)
                    elif label_int in [1, 2]:
                        mapped_labels.append(1)
                    elif label_int == 3:
                        mapped_labels.append(2)
                    elif label_int in [4, 5]:
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
                    'subject_id': sid,
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
            'labels': torch.LongTensor(sample['labels'])
        }

    def __del__(self):
        if hasattr(self, 'ppg_h5'):
            self.ppg_h5.close()
        if hasattr(self, 'ppg_index_h5'):
            self.ppg_index_h5.close()


def load_mamba_fusion_model(fusion_model_path, eeg_model_path, ppg_model_path, device):

    from short_window_eeg_model import ShortWindowAttnSleep
    from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow


    eeg_model = ShortWindowAttnSleep(window_minutes=3, num_classes=4)
    eeg_state = torch.load(eeg_model_path, map_location=device)
    if 'model_state_dict' in eeg_state:
        eeg_state = eeg_state['model_state_dict']
    eeg_model.load_state_dict(eeg_state)
    eeg_model.to(device)
    eeg_model.eval()


    ppg_model = PPGCrossAttnShortWindow(window_size='3min', n_classes=4)
    ppg_state = torch.load(ppg_model_path, map_location=device)
    if 'model_state_dict' in ppg_state:
        ppg_state = ppg_state['model_state_dict']
    ppg_model.load_state_dict(ppg_state)
    ppg_model.to(device)
    ppg_model.eval()


    model = EEGPPGCrossAttentionMambaFusion(
        eeg_model=eeg_model,
        ppg_model=ppg_model,
        d_model=256,
        n_heads=8,
        n_fusion_blocks=2,
        n_mamba_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        n_classes=4,
        dropout=0.1,
        freeze_encoders=True
    )


    fusion_state = torch.load(fusion_model_path, map_location=device)
    if 'model_state_dict' in fusion_state:
        fusion_state = fusion_state['model_state_dict']
    model.load_state_dict(fusion_state)
    model.to(device)

    print(f"Loaded Mamba fusion model from {fusion_model_path}")

    return model


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
        for batch in tqdm(loader, desc="Validating", leave=False):
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
    return total_loss / len(loader), acc, kappa


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Testing"):
            eeg = batch['eeg'].to(device)
            ppg = batch['ppg'].to(device)
            outputs = model(eeg, ppg)
            all_preds.extend(outputs.argmax(-1).cpu().numpy().flatten())
            all_labels.extend(batch['labels'].numpy().flatten())

    preds, labels = np.array(all_preds), np.array(all_labels)

    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])
    prec, rec, f1, supp = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2, 3], zero_division=0)
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)[2]

    return {
        'accuracy': acc,
        'kappa': kappa,
        'macro_f1': macro_f1,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'support': supp,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, output_path, title):
    plt.figure(figsize=(10, 8))
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    annot = np.empty_like(cm, dtype=object)
    for i in range(4):
        for j in range(4):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=['Wake', 'Light', 'Deep', 'REM'],
                yticklabels=['Wake', 'Light', 'Deep', 'REM'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['cfs', 'abc'])
    parser.add_argument('--fusion_model', type=str, default='./outputs/mamba_fusion/best_model.pth')
    parser.add_argument('--eeg_model', type=str,
                        default='../../work/mesa-short_window/outputs/eeg_window_3min/best_model.pth')
    parser.add_argument('--ppg_model', type=str,
                        default='../../work/mesa-short_window/outputs/short_window/3min_2/best_model.pth')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f'./outputs/mamba_fusion_finetune_{args.dataset}'
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"Fine-tuning Mamba Fusion on {args.dataset.upper()}")
    print("=" * 70)


    config = DATASET_CONFIG[args.dataset]


    print("\n[1/5] Loading model...")
    model = load_mamba_fusion_model(
        args.fusion_model, args.eeg_model, args.ppg_model, args.device
    )


    for param in model.fusion_blocks.parameters():
        param.requires_grad = True
    for param in model.mamba_tcm.parameters():
        param.requires_grad = True
    for param in model.fusion_projection.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")


    print("\n[2/5] Loading data...")


    eeg_folder = config['eeg_folder']
    all_subjects = []

    if os.path.exists(eeg_folder):
        for fname in os.listdir(eeg_folder):
            if fname.endswith('.npz'):
                if args.dataset == 'abc':
                    if fname.startswith('abc-baseline-'):
                        subject_num = fname.replace('abc-baseline-', '').replace('.npz', '')
                        all_subjects.append(f"{subject_num}_baseline")
                else:
                    all_subjects.append(fname.replace('.npz', ''))

    print(f"  Found {len(all_subjects)} subjects")


    np.random.seed(42)
    np.random.shuffle(all_subjects)
    n_train = int(len(all_subjects) * args.train_ratio)
    n_val = int(len(all_subjects) * 0.1)

    train_subjects = all_subjects[:n_train]
    val_subjects = all_subjects[n_train:n_train + n_val]
    test_subjects = all_subjects[n_train + n_val:]

    print(f"  Train: {len(train_subjects)}, Val: {len(val_subjects)}, Test: {len(test_subjects)}")


    train_dataset = CrossDatasetFusion(
        eeg_folder, config['ppg_data'], config['ppg_index'],
        train_subjects, args.dataset
    )
    val_dataset = CrossDatasetFusion(
        eeg_folder, config['ppg_data'], config['ppg_index'],
        val_subjects, args.dataset
    )
    test_dataset = CrossDatasetFusion(
        eeg_folder, config['ppg_data'], config['ppg_index'],
        test_subjects, args.dataset
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)


    print("\n[3/5] Calculating class weights...")
    class_counts = [0, 0, 0, 0]
    for s in train_dataset.samples:
        for l in s['labels']:
            class_counts[l] += 1
    total = sum(class_counts)

    if total > 0:
        class_weights = torch.tensor([np.sqrt(total / (4 * max(c, 1))) for c in class_counts], dtype=torch.float).to(
            args.device)
        for i, name in enumerate(['Wake', 'Light', 'Deep', 'REM']):
            print(f"  {name}: {class_counts[i]} ({class_counts[i] / total * 100:.1f}%) w={class_weights[i]:.2f}")
    else:
        class_weights = torch.ones(4).to(args.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    print("\n[4/5] Training...")
    best_kappa = -1
    patience_cnt = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_kappa = train_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc, val_kappa = validate(model, val_loader, criterion, args.device)
        scheduler.step()

        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Kappa: {train_kappa:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Kappa: {val_kappa:.4f}")

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
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break


    print("\n[5/5] Testing...")
    ckpt = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
    model.load_state_dict(ckpt['model_state_dict'])

    results = evaluate(model, test_loader, args.device)

    print("\n" + "=" * 70)
    print(f"TEST RESULTS - {args.dataset.upper()}")
    print("=" * 70)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"Kappa: {results['kappa']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"\n{'Class':<10} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 50)
    for i, name in enumerate(['Wake', 'Light', 'Deep', 'REM']):
        print(
            f"{name:<10} {results['precision'][i]:<10.4f} {results['recall'][i]:<10.4f} {results['f1'][i]:<10.4f} {results['support'][i]:<10}")


    save_results = {
        'dataset': args.dataset,
        'accuracy': float(results['accuracy']),
        'kappa': float(results['kappa']),
        'macro_f1': float(results['macro_f1']),
        'per_class': {
            name: {
                'precision': float(results['precision'][i]),
                'recall': float(results['recall'][i]),
                'f1': float(results['f1'][i]),
                'support': int(results['support'][i])
            }
            for i, name in enumerate(['Wake', 'Light', 'Deep', 'REM'])
        },
        'confusion_matrix': results['confusion_matrix'].tolist()
    }

    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    plot_confusion_matrix(
        results['confusion_matrix'],
        os.path.join(args.output_dir, 'confusion_matrix.png'),
        f'Mamba Fusion Fine-tuned on {args.dataset.upper()}'
    )

    print(f"\nResults saved to {args.output_dir}")


if __name__ == "__main__":
    main()