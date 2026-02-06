"""
Fine-tuning a MESA-pretrained EEG model on the ABC dataset


Usage:
    python finetune_eeg_on_abc.py \
        --abc_data_dir ../../data/eeg \
        --pretrained_path ./mesa_eeg_model.pth \
        --strategy discriminative \
        --lr 1e-4
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
from datetime import datetime
from collections import Counter, defaultdict
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import gc
import math
import copy
from copy import deepcopy
import torch.nn.functional as F

warnings.filterwarnings('ignore')


# ============================================================================
# Model components (AttnSleep, 4-class)
# ============================================================================

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
    """AttnSleep model (4-class version)"""

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
# ABC EEG dataset
# ============================================================================

class ABCEEGDataset(Dataset):
    """ABC EEG dataset (epoch-level)"""

    def __init__(self, npz_files, split='train', seed=42, train_ratio=0.6, val_ratio=0.2):
        """
        Args:
            npz_files: list of NPZ files
            split: 'train', 'val', or 'test'
            seed: random seed
            train_ratio: training split ratio
            val_ratio: validation split ratio
        """
        # ABC labels are already 4-class; no remapping needed
        # 0: Wake, 1: Light, 2: Deep, 3: REM
        self.label_map = {
            0: 0,  # Wake -> Wake
            1: 1,  # Light -> Light
            2: 2,  # Deep -> Deep
            3: 3,  # REM -> REM
        }

        # Get all subject IDs
        all_subjects = [Path(f).stem for f in npz_files]

        # Split dataset
        train_subjects, temp_subjects = train_test_split(
            all_subjects, test_size=1 - train_ratio, random_state=seed
        )
        val_subjects, test_subjects = train_test_split(
            temp_subjects, test_size=0.5, random_state=seed
        )

        if split == 'train':
            self.subjects = train_subjects
        elif split == 'val':
            self.subjects = val_subjects
        else:
            self.subjects = test_subjects

        # Create file path mapping
        self.subject_to_file = {Path(f).stem: f for f in npz_files}

        # Collect all samples
        self.samples = []  # (file_path, sample_idx, subject_id)
        class_counts = np.zeros(4, dtype=np.int64)

        print(f"\nLoading {split} dataset...")
        for subj in tqdm(self.subjects, desc=f"Loading {split} data"):
            if subj not in self.subject_to_file:
                continue

            file_path = self.subject_to_file[subj]
            try:
                data = np.load(file_path)
                y = data['y']

                for idx in range(len(y)):
                    original_label = int(y[idx])
                    if original_label in self.label_map:
                        new_label = self.label_map[original_label]
                        self.samples.append((file_path, idx, subj))
                        class_counts[new_label] += 1

                data.close()
            except Exception as e:
                print(f"  Warning: failed to load {subj}: {e}")
                continue

        self.class_counts = class_counts
        print(f"  {split} set: {len(self.subjects)} subjects, {len(self.samples)} samples")

        class_names = ['Wake', 'Light', 'Deep', 'REM']
        for i, name in enumerate(class_names):
            pct = class_counts[i] / len(self.samples) * 100 if len(self.samples) > 0 else 0
            print(f"    {name}: {class_counts[i]} ({pct:.1f}%)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        file_path, sample_idx, subject_id = self.samples[index]

        data = np.load(file_path)
        x = data['x'][sample_idx]
        y = data['y'][sample_idx]
        data.close()

        original_label = int(y)
        y_label = self.label_map[original_label]

        x_tensor = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        y_tensor = torch.tensor(y_label, dtype=torch.long)

        # Z-score normalization (more stable)
        mean = x_tensor.mean()
        std = x_tensor.std()
        if std > 1e-6:
            x_tensor = (x_tensor - mean) / std
        else:
            x_tensor = x_tensor - mean

        # Clip extreme values
        x_tensor = torch.clamp(x_tensor, -10, 10)

        return x_tensor, y_tensor, subject_id


# ============================================================================
# Fine-tuning trainer
# ============================================================================

class EEGFineTuner:
    """EEG fine-tuning trainer"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            config['output_dir'],
            f"finetune_eeg_{config['strategy']}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        # Save configuration
        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

        # Mixed precision
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None

    def load_pretrained_model(self):
        """Load pretrained model"""
        model = AttnSleep4Class()

        if self.config['pretrained_path'] and os.path.exists(self.config['pretrained_path']):
            print(f"\nLoading pretrained model: {self.config['pretrained_path']}")
            state_dict = torch.load(self.config['pretrained_path'], map_location=self.device)

            # Handle different checkpoint formats
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            model.load_state_dict(state_dict)
            print("✅ pretrained model loaded successfully")
        else:
            print("⚠️  No pretrained model specified or file not found — training from scratch")

        return model.to(self.device)

    def setup_finetune_strategy(self, model):
        """Set fine-tuning strategy"""
        strategy = self.config['strategy']

        if strategy == 'full':
            print("\nStrategy: Full model fine-tuning")
            for param in model.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        elif strategy == 'head_only':
            print("\nStrategy: Fine-tune classifier head only")

            # Freeze feature extractor
            for param in model.mrcnn.parameters():
                param.requires_grad = False
            for param in model.tce.parameters():
                param.requires_grad = False

            # Train fc layer only
            optimizer = optim.Adam(
                model.fc.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

        elif strategy == 'progressive':
            print("\nStrategy: Progressive unfreezing")

            for param in model.parameters():
                param.requires_grad = False

            for param in model.fc.parameters():
                param.requires_grad = True

            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        elif strategy == 'discriminative':
            print("\nStrategy: Differential learning rates")

            for param in model.parameters():
                param.requires_grad = True

            base_lr = self.config['learning_rate']
            param_groups = [
                {'params': model.mrcnn.parameters(), 'lr': base_lr * 0.01},
                {'params': model.tce.parameters(), 'lr': base_lr * 0.1},
                {'params': model.fc.parameters(), 'lr': base_lr},
            ]

            optimizer = optim.Adam(param_groups, weight_decay=self.config['weight_decay'])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return optimizer

    def unfreeze_layer(self, model, layer_name):
        """Unfreeze a specific layer"""
        layer = getattr(model, layer_name, None)
        if layer is not None:
            for param in layer.parameters():
                param.requires_grad = True
            print(f"  Unfroze: {layer_name}")

    def calculate_class_weights(self, dataset):
        """Compute class weights"""
        class_counts = dataset.class_counts
        total = class_counts.sum()

        # Weighting: total / (num_classes * class_count)
        weights = total / (len(class_counts) * class_counts + 1e-6)
        weights = weights / weights.sum() * len(class_counts)  # Normalize

        print(f"\nClass weights: {weights}")
        return torch.FloatTensor(weights).to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion):
        """Train one epoch"""
        model.train()
        running_loss = 0.0
        total_samples = 0
        nan_count = 0

        for batch_idx, (data, target, _) in enumerate(tqdm(dataloader, desc="Training")):
            data = data.to(self.device)
            target = target.to(self.device)

            # Check input NaNs
            if torch.isnan(data).any():
                nan_count += 1
                continue

            optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)

                # Check loss NaNs/Infs
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)

                # Check loss NaNs/Infs
                if torch.isnan(loss) or torch.isinf(loss):
                    nan_count += 1
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item() * data.size(0)
            total_samples += data.size(0)

            # Periodically clear GPU cache
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        if nan_count > 0:
            print(f"  Warning: skipped {nan_count} batches containing NaNs")

        return running_loss / total_samples if total_samples > 0 else float('nan')

    def evaluate(self, model, dataloader, criterion):
        """Evaluate"""
        model.eval()
        running_loss = 0.0
        total_samples = 0

        all_preds = []
        all_labels = []
        patient_results = defaultdict(lambda: {'preds': [], 'labels': []})

        with torch.no_grad():
            for data, target, subject_ids in tqdm(dataloader, desc="Evaluating"):
                data = data.to(self.device)
                target = target.to(self.device)

                output = model(data)
                loss = criterion(output, target)

                _, predicted = torch.max(output, 1)

                running_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

                # Aggregate by subject
                for i, subj in enumerate(subject_ids):
                    patient_results[subj]['preds'].append(predicted[i].cpu().item())
                    patient_results[subj]['labels'].append(target[i].cpu().item())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Overall metrics
        accuracy = accuracy_score(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        # Per-patient metrics
        patient_kappas = []
        patient_accuracies = []
        for subj, data in patient_results.items():
            preds = np.array(data['preds'])
            labels = np.array(data['labels'])

            patient_accuracies.append(accuracy_score(labels, preds))
            if len(np.unique(labels)) > 1:
                patient_kappas.append(cohen_kappa_score(labels, preds))

        median_kappa = np.median(patient_kappas) if patient_kappas else 0
        median_accuracy = np.median(patient_accuracies) if patient_accuracies else 0

        return {
            'loss': running_loss / total_samples,
            'accuracy': accuracy,
            'kappa': kappa,
            'f1': f1,
            'median_kappa': median_kappa,
            'median_accuracy': median_accuracy,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'patient_kappas': patient_kappas
        }

    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("Starting EEG fine-tuning")
        print("=" * 70)

        # Load data
        npz_files = sorted(glob.glob(os.path.join(self.config['abc_data_dir'], '*.npz')))
        if len(npz_files) == 0:
            raise FileNotFoundError(f"No NPZ files found in {self.config['abc_data_dir']}")

        print(f"\nFound {len(npz_files)} NPZ files")

        # Create datasets
        train_dataset = ABCEEGDataset(npz_files, split='train', seed=self.config.get('seed', 42))
        val_dataset = ABCEEGDataset(npz_files, split='val', seed=self.config.get('seed', 42))
        test_dataset = ABCEEGDataset(npz_files, split='test', seed=self.config.get('seed', 42))

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )

        # Load model
        model = self.load_pretrained_model()

        # Set fine-tuning strategy
        optimizer = self.setup_finetune_strategy(model)

        # Class weights
        class_weights = self.calculate_class_weights(train_dataset)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )

        # Training loop
        best_kappa = 0
        patience_counter = 0
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_kappa': [],
            'val_accuracy': []
        }

        for epoch in range(1, self.config['num_epochs'] + 1):
            print(f"\nEpoch {epoch}/{self.config['num_epochs']}")
            print("-" * 50)

            # Progressive unfreezing
            if self.config['strategy'] == 'progressive':
                if epoch == self.config.get('unfreeze_tce_epoch', 5):
                    print("\n🔓 Unfreezing TCE layers")
                    self.unfreeze_layer(model, 'tce')
                    optimizer = optim.Adam(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=self.config['learning_rate'] * 0.1,
                        weight_decay=self.config['weight_decay']
                    )

                elif epoch == self.config.get('unfreeze_all_epoch', 10):
                    print("\n🔓 Unfreezing all layers")
                    for param in model.parameters():
                        param.requires_grad = True
                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config['learning_rate'] * 0.01,
                        weight_decay=self.config['weight_decay']
                    )

            # Train
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_results = self.evaluate(model, val_loader, criterion)

            # Record
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_results['loss'])
            history['val_kappa'].append(val_results['kappa'])
            history['val_accuracy'].append(val_results['accuracy'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")
            print(f"Val Acc: {val_results['accuracy']:.4f}, Kappa: {val_results['kappa']:.4f}, "
                  f"Median Kappa: {val_results['median_kappa']:.4f}")

            # Learning rate scheduling
            scheduler.step(val_results['kappa'])

            # Saved best model
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

                print(f"✅ Saved best model (Kappa: {best_kappa:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break

        # Testing best model
        print("\n" + "=" * 70)
        print("Testing best model")
        print("=" * 70)

        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_results = self.evaluate(model, test_loader, criterion)

        print(f"\nTest results:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Kappa: {test_results['kappa']:.4f}")
        print(f"  Median Kappa: {test_results['median_kappa']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")

        # Classification report
        print("\nClassification report:")
        print(classification_report(
            test_results['all_labels'],
            test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM']
        ))

        # Save results
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

        # Plot confusion matrix
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

        # Plot training curves
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

        print(f"\n✅ Results saved to: {self.output_dir}")

        return results


# ============================================================================
# Main entrypoint
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a MESA-pretrained EEG model on ABC')

    # Data paths
    parser.add_argument('--abc_data_dir', type=str, required=True,
                        help='Directory containing ABC EEG NPZ files')
    parser.add_argument('--pretrained_path', type=str, default='',
                        help='Path to the MESA-pretrained model')

    # Fine-tuning strategy
    parser.add_argument('--strategy', type=str, default='discriminative',
                        choices=['full', 'head_only', 'progressive', 'discriminative'],
                        help='Fine-tuning strategy')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-5)

    # Progressive unfreezing
    parser.add_argument('--unfreeze_tce_epoch', type=int, default=5)
    parser.add_argument('--unfreeze_all_epoch', type=int, default=10)

    # Misc
    parser.add_argument('--output_dir', type=str, default='./finetune_eeg_outputs')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--no_amp', action='store_true', help='Disable mixed precision training')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    # Build configuration
    config = {
        'abc_data_dir': args.abc_data_dir,
        'pretrained_path': args.pretrained_path,
        'strategy': args.strategy,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'patience': args.patience,
        'weight_decay': args.weight_decay,
        'unfreeze_tce_epoch': args.unfreeze_tce_epoch,
        'unfreeze_all_epoch': args.unfreeze_all_epoch,
        'output_dir': args.output_dir,
        'gpu_id': args.gpu_id,
        'num_workers': args.num_workers,
        'use_amp': not args.no_amp,
        'seed': args.seed
    }

    print("\n" + "=" * 70)
    print("MESA → ABC EEG Fine-tuning")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  pretrained model: {config['pretrained_path'] or 'None (training from scratch)'}")
    print(f"  ABC data directory: {config['abc_data_dir']}")
    print(f"  Mixed precision: {config['use_amp']}")

    # Start fine-tuning
    finetuner = EEGFineTuner(config)
    results = finetuner.train()

    print("\n" + "=" * 70)
    print("Fine-tuning completed!")
    print("=" * 70)
    print(f"\nFinal test results:")
    print(f"  Accuracy: {results['test_accuracy']:.4f}")
    print(f"  Kappa: {results['test_kappa']:.4f}")
    print(f"  Median Kappa: {results['test_median_kappa']:.4f}")


if __name__ == '__main__':
    main()