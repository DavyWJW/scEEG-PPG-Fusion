import os
import sys
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
import gc

warnings.filterwarnings('ignore')


sys.path.append('..')
try:
    from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention
except ImportError:
    from ppg_unfiltered_crossattn import PPGUnfilteredCrossAttention



class ABCPPGTestDataset(Dataset):


    def __init__(self, data_path, min_windows=600, verbose=True):

        self.data_path = data_path
        self.min_windows = min_windows
        self.verbose = verbose


        self.ppg_file = os.path.join(data_path, 'abc_ppg_with_labels.h5')
        self.index_file = os.path.join(data_path, 'abc_subject_index.h5')

        if not os.path.exists(self.ppg_file):
            raise FileNotFoundError(f"PPG file not found: {self.ppg_file}")
        if not os.path.exists(self.index_file):
            raise FileNotFoundError(f"Index file not found: {self.index_file}")

        self.target_windows = 1200
        self.samples_per_window = 1024

        self._load_subjects()

    def _load_subjects(self):

        if self.verbose:
            print(f"🔧 Loading ABC PPG data...")
            print(f"   PPG file: {self.ppg_file}")
            print(f"   Index file: {self.index_file}")


        with h5py.File(self.index_file, 'r') as f:
            all_subjects = list(f['subjects'].keys())

            self.subjects = []
            self.subject_indices = {}
            self.subject_n_windows = {}

            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows >= self.min_windows:
                    indices = f[f'subjects/{subj}/window_indices'][:]
                    self.subjects.append(subj)
                    self.subject_indices[subj] = indices[0]
                    self.subject_n_windows[subj] = n_windows

        if self.verbose:
            total_windows = sum(self.subject_n_windows.values())
            avg_windows = total_windows / len(self.subjects) if self.subjects else 0

            print(f"\n✅ Loading complete:")
            print(f"   Valid subjects: {len(self.subjects)} / {len(all_subjects)}")
            print(f"   Minimum window requirement: {self.min_windows}")
            print(f"   Average windows per subject: {avg_windows:.1f}")
            print(f"   Total valid windows: {total_windows:,}")

        self._compute_label_distribution()

    def _compute_label_distribution(self):

        class_counts = np.zeros(4, dtype=np.int64)
        total_valid = 0

        with h5py.File(self.ppg_file, 'r') as f:
            for subj in self.subjects:
                start_idx = self.subject_indices[subj]
                n_windows = self.subject_n_windows[subj]
                labels = f['labels'][start_idx:start_idx + n_windows]

                for label in labels:
                    if 0 <= label < 4:
                        class_counts[label] += 1
                        total_valid += 1

        self.class_counts = class_counts
        self.total_valid_epochs = total_valid

        if self.verbose:
            class_names = ['Wake', 'Light', 'Deep', 'REM']
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                pct = count / total_valid * 100 if total_valid > 0 else 0
                print(f"   {i}: {name}: {count:,} ({pct:.1f}%)")

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


        ppg_continuous = ppg_padded.reshape(-1)  # [1228800]


        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)  # [1, 1228800]
        labels_tensor = torch.LongTensor(labels_padded)  # [1200]

        return ppg_tensor, labels_tensor, subject_id



def cross_dataset_ppg_evaluation(model_path, abc_data_dir, output_dir, config):


    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")

    print(f"\n📦 Loading MESA-trained dual-stream PPG model: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = PPGUnfilteredCrossAttention(
        n_classes=4,
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_fusion_blocks=config.get('n_fusion_blocks', 3),
        noise_config=config.get('noise_config', None)
    ).to(device)


    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'best_kappa' in checkpoint:
            print(f"   Best validation Kappa (MESA): {checkpoint['best_kappa']:.4f}")
        if 'epoch' in checkpoint:
            print(f"   Training epoch: {checkpoint['epoch']}")


    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f" Model loaded successfully")
    print(f" Parameter count: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n📂 Loading ABC PPG data: {abc_data_dir}")

    test_dataset = ABCPPGTestDataset(
        abc_data_dir,
        min_windows=config.get('min_windows', 600),
        verbose=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"   Running inference on the ABC dataset...")
    print(f"   被试数: {len(test_dataset)}")
    print(f"   Batch size: {config['batch_size']}")

    all_preds = []
    all_labels = []


    patient_predictions = defaultdict(list)
    patient_labels = defaultdict(list)
    patient_ids = {}


    clean_weights_all = []
    noisy_weights_all = []

    use_amp = config.get('use_amp', True) and torch.cuda.is_available()

    with torch.no_grad():
        for batch_idx, (ppg, labels, subject_ids) in enumerate(tqdm(test_loader, desc="inference")):
            ppg = ppg.to(device)


            if use_amp:
                with autocast():
                    outputs = model(ppg)
            else:
                outputs = model(ppg)


            clean_weight, noisy_weight = model.get_modality_weights()
            if clean_weight is not None:
                clean_weights_all.append(clean_weight.mean().item() if hasattr(clean_weight, 'mean') else clean_weight)
                noisy_weights_all.append(noisy_weight.mean().item() if hasattr(noisy_weight, 'mean') else noisy_weight)


            outputs = outputs.permute(0, 2, 1)  # [B, 1200, 4]

            batch_size = outputs.shape[0]
            for i in range(batch_size):
                patient_idx = batch_idx * config['batch_size'] + i
                subj_id = subject_ids[i]
                patient_ids[patient_idx] = subj_id


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


            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()

    gc.collect()
    torch.cuda.empty_cache()

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print(f" Computing evaluation metrics...")

    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    f1_macro = f1_score(all_labels, all_preds, average='macro')


    patient_kappas = []
    patient_accuracies = []
    patient_f1s = []
    patient_details = {}

    for patient_idx in patient_predictions:
        if len(patient_predictions[patient_idx]) > 0:
            p_preds = np.array(patient_predictions[patient_idx])
            p_labels = np.array(patient_labels[patient_idx])
            subj_id = patient_ids.get(patient_idx, f"subject_{patient_idx}")

            patient_acc = accuracy_score(p_labels, p_preds)
            patient_accuracies.append(patient_acc)


            if len(np.unique(p_labels)) > 1:
                patient_kappa = cohen_kappa_score(p_labels, p_preds)
                patient_kappas.append(patient_kappa)
            else:
                patient_kappa = 0.0

            patient_f1 = f1_score(p_labels, p_preds, average='weighted', zero_division=0)
            patient_f1s.append(patient_f1)

            patient_details[subj_id] = {
                'accuracy': float(patient_acc),
                'kappa': float(patient_kappa),
                'f1': float(patient_f1),
                'n_epochs': len(p_labels)
            }

    median_accuracy = np.median(patient_accuracies) if patient_accuracies else 0
    median_kappa = np.median(patient_kappas) if patient_kappas else 0
    median_f1 = np.median(patient_f1s) if patient_f1s else 0


    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])


    class_names = ['Wake', 'Light', 'Deep', 'REM']
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
        labels=[0, 1, 2, 3]
    )


    print(f"\n{'=' * 70}")
    print(f"Cross-dataset evaluation results: MESA → ABC (dual-stream PPG model, zero-shot)")
    print(f"{'=' * 70}")

    print(f"\n📈 Overall metrics:")
    print(f"   Accuracy: {accuracy * 100:.2f}%")
    print(f"   Kappa: {kappa:.4f}")
    print(f"   F1 (weighted): {f1_weighted:.4f}")
    print(f"   F1 (macro): {f1_macro:.4f}")

    print(f"\n📈 Per-patient median metrics:")
    print(f"   Accuracy: {median_accuracy * 100:.2f}%")
    print(f"   Kappa: {median_kappa:.4f}")
    print(f"   F1: {median_f1:.4f}")

    if patient_kappas:
        print(f"\n   Kappa distribution:")
        print(f"     Min: {np.min(patient_kappas):.4f}")
        print(f"     25%: {np.percentile(patient_kappas, 25):.4f}")
        print(f"     Median: {median_kappa:.4f}")
        print(f"     75%: {np.percentile(patient_kappas, 75):.4f}")
        print(f"     Max: {np.max(patient_kappas):.4f}")


    if clean_weights_all:
        avg_clean = np.mean(clean_weights_all)
        avg_noisy = np.mean(noisy_weights_all)
        print(f"\n🔀 Modality weights (average):")
        print(f"   Clean PPG: {avg_clean:.3f}")
        print(f"   Noisy PPG: {avg_noisy:.3f}")

    print(f"\n📊 Per-class performance:")
    print(f"{'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)

    for name in class_names:
        if name in report:
            p = report[name]['precision']
            r = report[name]['recall']
            f = report[name]['f1-score']
            s = report[name]['support']
            print(f"{name:<15} {p:>10.3f} {r:>10.3f} {f:>10.3f} {int(s):>10}")


    print(classification_report(all_labels, all_preds, target_names=class_names,
                                zero_division=0, labels=[0, 1, 2, 3]))



    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix (Counts)\nMESA → ABC (Dual-Stream PPG, Zero-Shot)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')

    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Confusion Matrix (Normalized)\nMESA → ABC (Dual-Stream PPG, Zero-Shot)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


    if patient_kappas:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].hist(patient_kappas, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(median_kappa, color='red', linestyle='--', linewidth=2,
                        label=f'Median: {median_kappa:.3f}')
        axes[0].axvline(np.mean(patient_kappas), color='orange', linestyle='--', linewidth=2,
                        label=f'Mean: {np.mean(patient_kappas):.3f}')
        axes[0].set_xlabel('Cohen\'s Kappa')
        axes[0].set_ylabel('Number of Patients')
        axes[0].set_title('Per-Patient Kappa Distribution\nMESA → ABC (Zero-Shot)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Box plot
        axes[1].boxplot(patient_kappas, vert=True)
        axes[1].set_ylabel('Kappa')
        axes[1].set_title('Per-Patient Kappa Box Plot')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'kappa_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


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
    ax.set_title('Per-Class Performance: MESA → ABC (Dual-Stream PPG, Zero-Shot)')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_performance.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


    results = {
        'experiment': 'Cross-Dataset Evaluation: MESA → ABC (Dual-Stream PPG, Zero-Shot)',
        'model_path': model_path,
        'abc_data_dir': abc_data_dir,
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

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


    import pandas as pd
    patient_df = pd.DataFrame([
        {'subject_id': k, **v} for k, v in patient_details.items()
    ])
    patient_df.to_csv(os.path.join(output_dir, 'patient_results.csv'), index=False)

    print(f"\n💾 result: {output_dir}")

    return results



def main():
    parser = argparse.ArgumentParser(
        description='Cross-dataset evaluation: MESA → ABC (Dual-stream PPG)'
    )

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the MESA-trained dual-stream PPG model (.pth file)')

    parser.add_argument('--abc_data_dir', type=str, required=True,
                        help='ABC dataset directory (contains abc_ppg_with_labels.h5 and abc_subject_index.h5)')

    parser.add_argument('--output_dir', type=str, default='./cross_dataset_abc_results',
                        help='Output directory')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')

    parser.add_argument('--min_windows', type=int, default=600,
                        help='Minimum number of windows')

    parser.add_argument('--no_amp', action='store_true',
                        help='Disable mixed precision')


    args = parser.parse_args()

    print("=" * 70)
    print("🔄 Cross-dataset evaluation: MESA → ABC (dual-stream PPG model, zero-shot)")
    print("=" * 70)
    print("\nEvaluating the generalization of the MESA-trained PPG + unfiltered PPG model on ABC")
    print("Note: This is a zero-shot test with no fine-tuning")


    config = {
        'batch_size': args.batch_size,
        'gpu_id': args.gpu_id,
        'num_workers': args.num_workers,
        'min_windows': args.min_windows,
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


    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'mesa_to_abc_dualppg_{timestamp}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)


    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    try:
        results = cross_dataset_ppg_evaluation(
            args.model_path,
            args.abc_data_dir,
            output_dir,
            config
        )

        print(f"\n{'=' * 70}")
        print(f" Cross-dataset evaluation completed!")
        print(f"{'=' * 70}")
        print(f"\nFinal results (MESA dual-stream PPG → ABC, zero-shot):")
        print(f"   Overall: {results['overall_metrics']['accuracy'] * 100:.2f}%")
        print(f"   Overall Kappa: {results['overall_metrics']['kappa']:.4f}")
        print(f"   Median Kappa: {results['per_patient_median_metrics']['kappa']:.4f}")
        print(f"   F1 (weighted): {results['overall_metrics']['f1_weighted']:.4f}")

    except Exception as e:
        print(f"\n❌ {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "=" * 70)
    print("=" * 70)


if __name__ == '__main__':
    main()