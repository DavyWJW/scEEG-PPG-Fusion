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


sys.path.append('..')
try:
    from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
except ImportError:
    from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention



class ABCDualPPGDataset(Dataset):


    def __init__(self, ppg_file, index_file, split='train', seed=42,
                 train_ratio=0.6, val_ratio=0.2, min_windows=600):

        self.ppg_file = ppg_file
        self.index_file = index_file
        self.split = split
        self.target_windows = 1200
        self.samples_per_window = 1024


        with h5py.File(index_file, 'r') as f:
            all_subjects = list(f['subjects'].keys())
            

            valid_subjects = []
            self.subject_n_windows = {}
            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows >= min_windows:
                    valid_subjects.append(subj)
                    self.subject_n_windows[subj] = n_windows

        print(f"Valid subjects: {len(valid_subjects)}/{len(all_subjects)} (min_windows={min_windows})")

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


        if n_windows < self.target_windows:
            ppg_padded = np.zeros((self.target_windows, self.samples_per_window), dtype=np.float32)
            labels_padded = np.full(self.target_windows, -1, dtype=np.int64)
            
            ppg_padded[:n_windows] = ppg_windows
            labels_padded[:n_windows] = labels
        else:
            ppg_padded = ppg_windows[:self.target_windows].astype(np.float32)
            labels_padded = labels[:self.target_windows].astype(np.int64)


        ppg = ppg_padded.reshape(-1)  # [1228800]


        ppg_tensor = torch.FloatTensor(ppg).unsqueeze(0)  # [1, 1228800]
        labels_tensor = torch.LongTensor(labels_padded)  # [1200]

        return ppg_tensor, labels_tensor



class DualPPGFineTuner:


    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.set_per_process_memory_fraction(0.95)


        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None


        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)


        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            config['output_dir'],
            f"abc_dual_finetune_{config['strategy']}_{timestamp}"
        )
        os.makedirs(self.output_dir, exist_ok=True)


        with open(os.path.join(self.output_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)

    def load_pretrained_model(self):

        model = PPGUnfilteredCrossAttention()

        if self.config['pretrained_path'] and os.path.exists(self.config['pretrained_path']):
            print(f"\nLoading pretrained model: {self.config['pretrained_path']}")
            checkpoint = torch.load(self.config['pretrained_path'], map_location=self.device)

            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            print("✅ Pretrained dual-stream model loaded successfully")
        else:
            print("⚠️  No pretrained model specified or file not found — training from scratch")

        return model.to(self.device)

    def setup_finetune_strategy(self, model):

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

            print("\nStrategy: Fine-tune classification head and fusion layers only")


            for param in model.clean_ppg_encoder.parameters():
                param.requires_grad = False
            for param in model.noisy_ppg_encoder.parameters():
                param.requires_grad = False


            trainable_params = []


            for param in model.fusion_blocks.parameters():
                param.requires_grad = True
            trainable_params.extend(model.fusion_blocks.parameters())


            for param in model.modality_weighting.parameters():
                param.requires_grad = True
            trainable_params.extend(model.modality_weighting.parameters())


            for param in model.feature_aggregation.parameters():
                param.requires_grad = True
            trainable_params.extend(model.feature_aggregation.parameters())


            for param in model.temporal_blocks.parameters():
                param.requires_grad = True
            trainable_params.extend(model.temporal_blocks.parameters())


            for param in model.feature_refinement.parameters():
                param.requires_grad = True
            trainable_params.extend(model.feature_refinement.parameters())


            for param in model.classifier.parameters():
                param.requires_grad = True
            trainable_params.extend(model.classifier.parameters())

            optimizer = optim.Adam(
                trainable_params,
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )

        elif strategy == 'progressive':

            print("\nStrategy: Progressive unfreezing")

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

            print("\nStrategy: Differential learning rates")

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
            raise ValueError(f"ValueError: {strategy}")


        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"parameters: {trainable:,} / {total:,} ({trainable / total * 100:.1f}%)")

        return optimizer

    def unfreeze_layer(self, model, layer_name):

        if hasattr(model, layer_name):
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
            print(f"  unfreeze: {layer_name}")

    def calculate_class_weights(self, dataloader):

        print("\nclass_weights...")
        all_labels = []

        for ppg, labels in tqdm(dataloader, desc="label", leave=False):
            valid_labels = labels[labels >= 0].numpy().flatten()
            all_labels.extend(valid_labels.tolist())

        label_counts = Counter(all_labels)
        
        print("label:")
        label_names = ['Wake', 'Light', 'Deep', 'REM']
        class_counts = []
        for i in range(4):
            count = label_counts.get(i, 1)
            class_counts.append(count)
            print(f"  {label_names[i]}: {count} ({100 * count / len(all_labels):.1f}%)")


        weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
        weights = weights / weights.sum() * 4

        print(f"weights: {[f'{w:.3f}' for w in weights.tolist()]}")

        return weights.to(self.device)

    def train_epoch(self, model, dataloader, optimizer, criterion):

        model.train()
        running_loss = 0.0
        total = 0

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            ppg, labels = batch
            ppg = ppg.to(self.device)
            labels = labels.to(self.device)


            if self.use_amp:
                with autocast():
                    outputs = model(ppg)
                    outputs = outputs.permute(0, 2, 1)

                    loss = criterion(
                        outputs.reshape(-1, 4),
                        labels.reshape(-1)
                    )


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


            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()


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


                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()

        gc.collect()
        torch.cuda.empty_cache()


        overall_acc = accuracy_score(all_labels, all_preds) if all_labels else 0
        overall_kappa = cohen_kappa_score(all_labels, all_preds) if all_labels else 0
        overall_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0


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

        print("\n" + "=" * 70)
        print("=" * 70)


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


        model = self.load_pretrained_model()


        optimizer = self.setup_finetune_strategy(model)


        class_weights = self.calculate_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)


        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )


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


            if self.config['strategy'] == 'progressive':
                if epoch == self.config.get('unfreeze_tcn_epoch', 5):
                    print(" Unfreezing TCN and fusion layers")
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
                    print(" Unfreezing all layers")
                    for param in model.parameters():
                        param.requires_grad = True

                    optimizer = optim.Adam(
                        model.parameters(),
                        lr=self.config['learning_rate'] * 0.01,
                        weight_decay=self.config['weight_decay']
                    )


            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)


            val_results = self.evaluate(model, val_loader, criterion)


            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_results['loss'])
            history['val_kappa'].append(val_results['kappa'])
            history['val_accuracy'].append(val_results['accuracy'])

            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")
            print(f"Val Acc: {val_results['accuracy']:.4f}, Kappa: {val_results['kappa']:.4f}, "
                  f"Median Kappa: {val_results['median_kappa']:.4f}")


            scheduler.step(val_results['kappa'])


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

                print(f"✅ best model (Kappa: {best_kappa:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"\n⏹️  Early stopping at epoch {epoch}")
                    break


        print("\n" + "=" * 70)
        print("=" * 70)

        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        test_results = self.evaluate(model, test_loader, criterion)

        print(f"\ntest:")
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Kappa: {test_results['kappa']:.4f}")
        print(f"  Median Kappa: {test_results['median_kappa']:.4f}")
        print(f"  F1: {test_results['f1']:.4f}")

        print("\nClassification report:")
        print(classification_report(
            test_results['all_labels'],
            test_results['all_preds'],
            target_names=['Wake', 'Light', 'Deep', 'REM']
        ))

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

        print(f" result: {self.output_dir}")

        return results



def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune a MESA-pretrained dual-stream PPG model on ABC'
    )

    # Data paths
    parser.add_argument('--abc_ppg_file', type=str,
                        default='../../data/abc_ppg_with_labels.h5')
    parser.add_argument('--abc_index_file', type=str,
                        default='../../data/abc_subject_index.h5')
    parser.add_argument('--pretrained_path', type=str,
                        default='./dual_best_model.pth',
                        help='Path to the MESA-pretrained dual-stream model')

    # Fine-tuning strategy
    parser.add_argument('--strategy', type=str, default='discriminative',
                        choices=['full', 'head_only', 'progressive', 'discriminative'],
                        help='Fine-tuning strategy')

    # Training hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (recommended: 1 to reduce memory usage)')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--min_windows', type=int, default=600,
                        help='Minimum number of windows; subjects with fewer windows will be excluded')

    # Memory optimization
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch_size = batch_size * steps)')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision training')

    # Progressive unfreezing
    parser.add_argument('--unfreeze_tcn_epoch', type=int, default=5)
    parser.add_argument('--unfreeze_all_epoch', type=int, default=10)

    # Output
    parser.add_argument('--output_dir', type=str, default='./abc_dual_finetune_outputs')

    args = parser.parse_args()

    # Build config
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
    print("MESA → ABC Dual-stream PPG Fine-tuning (PPGUnfilteredCrossAttention)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Strategy: {config['strategy']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Gradient accumulation: {config['gradient_accumulation_steps']} "
          f"(effective batch={config['batch_size'] * config['gradient_accumulation_steps']})")
    print(f"  Mixed precision: {'enabled' if config['use_amp'] else 'disabled'}")
    print(f"  Pretrained model: {config['pretrained_path']}")
    print(f"  ABC data: {config['abc_ppg_file']}")
    print(f"  Minimum windows: {config['min_windows']}")



    finetuner = DualPPGFineTuner(config)
    results = finetuner.train()

    print("\n" + "=" * 70)
    print("=" * 70)


if __name__ == '__main__':
    main()
