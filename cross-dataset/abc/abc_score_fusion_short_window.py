import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import glob
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import os
import time
import argparse
import sys
import warnings

warnings.filterwarnings('ignore')


try:
    from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow, create_model_for_window
    from short_window_eeg_model import ShortWindowAttnSleep
except ImportError:
    print("Make sure ppg_crossattn_shortwindow.py和short_window_eeg_model.py are in the current directory or PYTHONPATH")
    sys.exit(1)


def load_abc_test_data(ppg_path, ppg_index_path, eeg_folder, seed=42):

    with h5py.File(ppg_index_path, 'r') as f:
        all_subjects = list(f['subjects'].keys())
        ppg_valid_subjects = []
        ppg_indices = {}
        for subj in all_subjects:
            n_windows = f[f'subjects/{subj}'].attrs['n_windows']
            if n_windows >= 100:  # 至少100个epoch
                ppg_valid_subjects.append(subj)
                indices = f[f'subjects/{subj}/window_indices'][:]
                ppg_indices[subj] = (indices[0], n_windows)

    print(f"Number of valid PPG subjects: {len(ppg_valid_subjects)}")
    print(f"Example PPG IDs: {ppg_valid_subjects[:3]}")

    eeg_files_raw = {}
    files = glob.glob(str(Path(eeg_folder) / "*.npz"))

    for file in files:
        filename = Path(file).stem

        parts = filename.split('-')
        eeg_id = parts[-1] if len(parts) >= 3 else filename
        eeg_files_raw[eeg_id] = file

    print(f"Number of EEG files found: {len(eeg_files_raw)}")
    print(f"Example EEG IDs: {list(eeg_files_raw.keys())[:3]}")

    eeg_files = {}
    for ppg_id in ppg_valid_subjects:

        eeg_id = ppg_id.split('_')[0]
        if eeg_id in eeg_files_raw:
            eeg_files[ppg_id] = eeg_files_raw[eeg_id]


    common_subjects = [s for s in ppg_valid_subjects if s in eeg_files]

    train_subjects, temp_subjects = train_test_split(common_subjects, test_size=0.4, random_state=seed)
    val_subjects, test_subjects = train_test_split(temp_subjects, test_size=0.5, random_state=seed)


    return test_subjects, ppg_indices, eeg_files


def fusion_inference(ppg_model, eeg_model, test_subjects, ppg_indices,
                     eeg_files, ppg_path, device, window_minutes, alpha=0.3):

    ppg_model.eval()
    eeg_model.eval()

    epochs_per_window = window_minutes * 2
    samples_per_epoch_ppg = 1024
    samples_per_epoch_eeg = 3000

    all_preds = []
    all_labels = []
    inference_times = []
    skipped_subjects = 0

    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for subj in tqdm(test_subjects, desc=f"Alpha={alpha}"):
            try:

                start_idx, n_windows = ppg_indices[subj]


                with h5py.File(ppg_path, 'r') as f:
                    ppg_windows = f['ppg'][start_idx:start_idx + n_windows]
                    ppg_labels = f['labels'][start_idx:start_idx + n_windows]


                if subj not in eeg_files:
                    skipped_subjects += 1
                    continue

                eeg_data = np.load(eeg_files[subj])
                eeg_epochs = eeg_data['x']
                n_eeg = len(eeg_epochs)


                min_len = min(n_windows, n_eeg)
                if min_len < epochs_per_window:
                    skipped_subjects += 1
                    continue


                ppg_windows = ppg_windows[:min_len]
                ppg_labels = ppg_labels[:min_len]
                eeg_epochs = eeg_epochs[:min_len]


                valid_mask = (ppg_labels >= 0) & (ppg_labels <= 3)
                ppg_windows = ppg_windows[valid_mask]
                ppg_labels = ppg_labels[valid_mask]
                eeg_epochs = eeg_epochs[valid_mask]

                if len(ppg_labels) < epochs_per_window:
                    skipped_subjects += 1
                    continue


                n_windows_total = len(ppg_labels) // epochs_per_window

                for win_idx in range(n_windows_total):
                    start = win_idx * epochs_per_window
                    end = start + epochs_per_window


                    ppg_win = ppg_windows[start:end]
                    labels_win = ppg_labels[start:end]
                    eeg_win = eeg_epochs[start:end]


                    ppg_continuous = ppg_win.reshape(-1).astype(np.float32)
                    ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0).unsqueeze(0).to(device)


                    eeg_tensor = torch.FloatTensor(eeg_win.astype(np.float32)).unsqueeze(0).to(device)


                    eeg_mean = eeg_tensor.mean(dim=-1, keepdim=True)
                    eeg_std = eeg_tensor.std(dim=-1, keepdim=True)
                    eeg_std = torch.where(eeg_std > 1e-6, eeg_std, torch.ones_like(eeg_std))
                    eeg_tensor = (eeg_tensor - eeg_mean) / eeg_std
                    eeg_tensor = torch.clamp(eeg_tensor, -10, 10)


                    if use_cuda:
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()


                    ppg_logits = ppg_model(ppg_tensor)
                    ppg_probs = F.softmax(ppg_logits, dim=1)
                    ppg_probs = ppg_probs[0].transpose(0, 1)  # (n_epochs, 4)


                    eeg_logits = eeg_model(eeg_tensor)
                    eeg_probs = F.softmax(eeg_logits, dim=-1)
                    eeg_probs = eeg_probs[0]  # (n_epochs, 4)


                    fused_probs = alpha * ppg_probs + (1 - alpha) * eeg_probs
                    preds = torch.argmax(fused_probs, dim=1).cpu().numpy()


                    if use_cuda:
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    inference_times.append((end_time - start_time) * 1000)  # ms

                    all_preds.extend(preds)
                    all_labels.extend(labels_win)

            except Exception as e:
                skipped_subjects += 1
                continue

    if skipped_subjects > 0:
        print(f"  Skipped {skipped_subjects} subjects (incomplete data or errors)")

    inf_mean = np.mean(inference_times) if inference_times else 0
    inf_std = np.std(inference_times) if inference_times else 0

    return np.array(all_preds), np.array(all_labels), inf_mean, inf_std


def evaluate_fusion(window_minutes, ppg_model_path, eeg_model_path,
                    ppg_path, index_path, eeg_folder, output_dir, gpu_id=0):


    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    if device.type == 'cpu':
        print("Warning: Running on CPU — execution may be slow")
    print(f"Window length: {window_minutes}")


    window_size_str = f"{window_minutes}min"


    print("\nDual PPG ...")
    ppg_model = create_model_for_window(window_size_str).to(device)

    checkpoint = torch.load(ppg_model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ppg_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        best_kappa = checkpoint.get('best_kappa', None)
        if best_kappa is not None:
            print(f"  load epoch {epoch}, best_kappa={best_kappa:.4f}")
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        ppg_model.load_state_dict(checkpoint['model'])
    else:
        ppg_model.load_state_dict(checkpoint)

    ppg_params = sum(p.numel() for p in ppg_model.parameters())
    print(f"  {ppg_params:,} ({ppg_params / 1e6:.2f}M)")


    eeg_model = ShortWindowAttnSleep(
        window_minutes=window_minutes,
        num_classes=4
    ).to(device)

    eeg_checkpoint = torch.load(eeg_model_path, map_location=device, weights_only=False)
    if isinstance(eeg_checkpoint, dict) and 'model_state_dict' in eeg_checkpoint:
        eeg_model.load_state_dict(eeg_checkpoint['model_state_dict'])
        epoch = eeg_checkpoint.get('epoch', '?')
        best_kappa = eeg_checkpoint.get('best_kappa', None)
        if best_kappa is not None:
            print(f"  load epoch {epoch}, best_kappa={best_kappa:.4f}")
    else:
        eeg_model.load_state_dict(eeg_checkpoint)

    eeg_params = sum(p.numel() for p in eeg_model.parameters())
    print(f"   {eeg_params:,} ({eeg_params / 1e6:.2f}M)")

    total_params = ppg_params + eeg_params
    print(f" {total_params:,} ({total_params / 1e6:.2f}M)")


    test_subjects, ppg_indices, eeg_files = load_abc_test_data(
        ppg_path, index_path, eeg_folder
    )


    alpha = 0.4
    print(f"\n{'=' * 70}")
    print(f"test Alpha={alpha} (PPG:{alpha * 100:.0f}%, EEG:{(1 - alpha) * 100:.0f}%)")
    print(f"{'=' * 70}")

    class_names = ['Wake', 'Light', 'Deep', 'REM']

    preds, labels, inf_mean, inf_std = fusion_inference(
        ppg_model, eeg_model, test_subjects, ppg_indices,
        eeg_files, ppg_path, device, window_minutes, alpha
    )

    if len(preds) == 0:
        print("❌")
        return None


    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])


    print(f"\n{'#' * 70}")
    print(f"##  ({window_minutes})")
    print(f"{'#' * 70}")
    print(f"\n{'Metric':<25} {'Value':<20}")
    print(f"{'-' * 45}")
    print(f"{'κ (Kappa)':<25} {kappa:.4f}")
    print(f"{'Acc (Accuracy)':<25} {acc * 100:.2f}%")
    print(f"{'Model Size':<25} {total_params / 1e6:.2f}M")
    print(f"{'Inference (ms)':<25} {inf_mean:.2f}±{inf_std:.2f}")
    print(f"{'-' * 45}")

    print(f"\nConfusion matrix:")
    print(f"{'-' * 70}")
    print(f"{'True/Pred':<10}", end="")
    for name in class_names:
        print(f"{name:<12}", end="")
    print(f"{'Recall':<10}")
    print(f"{'-' * 70}")
    for i, name in enumerate(class_names):
        print(f"{name:<10}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:<12}", end="")
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
        print(f"{recall:.2%}")
    print(f"{'-' * 70}")


    results = {
        'dataset': 'ABC',
        'window_minutes': window_minutes,
        'alpha': alpha,
        'metrics': {
            'kappa': float(kappa),
            'accuracy': float(acc),
            'f1_macro': float(f1_macro),
            'model_size_M': total_params / 1e6,
            'inference_mean_ms': float(inf_mean),
            'inference_std_ms': float(inf_std)
        },
        'confusion_matrix': cm.tolist(),
        'per_class_recall': {
            name: float(cm[i, i] / cm[i].sum()) if cm[i].sum() > 0 else 0
            for i, name in enumerate(class_names)
        },
        'model_params': {
            'ppg': ppg_params,
            'eeg': eeg_params,
            'total': total_params
        },
        'transfer_strategy': {
            'ppg': 'MESA pretrain -> ABC zero-shot',
            'eeg': 'MESA pretrain -> ABC zero-shot'
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f'abc_fusion_{window_minutes}min_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f" {result_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Short-window PPG+EEG fusion evaluation on the ABC dataset')

    parser.add_argument('--window', type=int, required=True, choices=[3, 5, 10, 30],
                        help='Window length (minutes)')

    parser.add_argument('--ppg_model', type=str, required=True,
                        help='Path to the MESA-trained short-window PPG model')

    parser.add_argument('--eeg_model', type=str, required=True,
                        help='Path to the MESA-trained short-window EEG model')

    parser.add_argument('--ppg_data', type=str, required=True,
                        help='Path to the ABC PPG dataset (abc_ppg_with_labels.h5)')

    parser.add_argument('--ppg_index', type=str, required=True,
                        help='Path to the ABC PPG index file (abc_subject_index.h5)')

    parser.add_argument('--eeg_folder', type=str, required=True,
                        help='ABC EEG data folder')

    parser.add_argument('--output', type=str, default='./abc_fusion_short_results',
                        help='Output directory')

    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID')

    args = parser.parse_args()

    evaluate_fusion(
        window_minutes=args.window,
        ppg_model_path=args.ppg_model,
        eeg_model_path=args.eeg_model,
        ppg_path=args.ppg_data,
        index_path=args.ppg_index,
        eeg_folder=args.eeg_folder,
        output_dir=args.output,
        gpu_id=args.gpu
    )


if __name__ == '__main__':
    main()