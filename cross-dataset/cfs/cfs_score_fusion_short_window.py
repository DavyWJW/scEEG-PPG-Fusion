"""
CFS数据集上 短窗口 Dual PPG + EEG 概率层融合评估

- Dual PPG模型: MESA预训练 → CFS微调
- EEG模型: MESA预训练，零样本迁移（不微调）
- 融合策略: 加权概率融合，搜索最优alpha

支持窗口长度: 3分钟、5分钟、10分钟、30分钟
"""
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

# 导入模型
sys.path.append('..')
from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow, create_model_for_window
from short_window_eeg_model import ShortWindowAttnSleep


def load_cfs_test_data(ppg_path, eeg_folder, index_path, seed=42):
    """
    加载CFS测试集数据

    与微调时使用相同的划分方式
    """
    # 获取所有有效被试
    with h5py.File(index_path, 'r') as f:
        all_subjects = list(f['subjects'].keys())
        valid_subjects = []
        for subj in all_subjects:
            n_windows = f[f'subjects/{subj}'].attrs['n_windows']
            if n_windows == 1200:
                valid_subjects.append(subj)

    print(f"CFS有效被试总数: {len(valid_subjects)}")

    # 划分训练/测试集 (与微调时相同的划分)
    train_subjects, test_subjects = train_test_split(
        valid_subjects, test_size=0.2, random_state=seed
    )

    print(f"测试集被试数: {len(test_subjects)}")

    # 获取PPG索引
    ppg_indices = {}
    with h5py.File(index_path, 'r') as f:
        for subj in test_subjects:
            indices = f[f'subjects/{subj}/window_indices'][:]
            ppg_indices[subj] = indices[0]

    # 加载EEG文件映射
    eeg_files = {}
    eeg_pattern = str(Path(eeg_folder) / "*.npz")
    files = glob.glob(eeg_pattern)

    for file in files:
        # CFS EEG文件名格式: 800002.npz
        filename = Path(file).stem
        # 处理可能的格式: cfs-visit5-800002 或 800002
        subj_id = filename.split('-')[-1] if '-' in filename else filename

        if subj_id in test_subjects:
            eeg_files[subj_id] = file

    print(f"找到EEG文件: {len(eeg_files)}")

    # 找出同时有PPG和EEG的被试
    common_subjects = [s for s in test_subjects if s in eeg_files]
    print(f"同时有PPG和EEG的测试被试: {len(common_subjects)}")

    return common_subjects, ppg_indices, eeg_files


def fusion_inference(ppg_model, eeg_model, test_subjects, ppg_indices,
                     eeg_files, ppg_path, device, window_minutes, alpha=0.5):
    """
    短窗口融合推理

    Args:
        window_minutes: 窗口长度（分钟）
        alpha: 融合权重 (0=纯EEG, 1=纯PPG)

    Returns:
        preds, labels, avg_inference_time_ms, std_inference_time_ms
    """
    ppg_model.eval()
    eeg_model.eval()

    epochs_per_window = window_minutes * 2  # 每分钟2个epoch (30秒/epoch)
    samples_per_epoch = 1024  # PPG采样点/epoch

    all_preds = []
    all_labels = []
    inference_times = []
    skipped_subjects = 0

    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for subj in tqdm(test_subjects, desc=f"Alpha={alpha:.1f}"):
            # 加载PPG数据
            with h5py.File(ppg_path, 'r') as f:
                start_idx = ppg_indices[subj]
                ppg_windows = f['ppg'][start_idx:start_idx + 1200]
                ppg_labels_raw = f['labels'][start_idx:start_idx + 1200]

            # 加载EEG数据
            if subj not in eeg_files:
                skipped_subjects += 1
                continue

            eeg_data = np.load(eeg_files[subj])
            eeg_epochs = eeg_data['x']
            n_eeg = len(eeg_epochs)

            # 取最小长度
            min_len = min(1200, n_eeg)
            if min_len < epochs_per_window:
                skipped_subjects += 1
                continue

            # 截取到相同长度
            ppg_windows = ppg_windows[:min_len]
            ppg_labels_raw = ppg_labels_raw[:min_len]
            eeg_epochs = eeg_epochs[:min_len]

            # 过滤无效标签 (PPG标签: 0-3有效, -1无效)
            valid_mask = ppg_labels_raw >= 0
            ppg_windows = ppg_windows[valid_mask]
            ppg_labels = ppg_labels_raw[valid_mask]
            eeg_epochs = eeg_epochs[valid_mask]

            if len(ppg_labels) < epochs_per_window:
                skipped_subjects += 1
                continue

            # 分割成短窗口
            n_windows = len(ppg_labels) // epochs_per_window

            for win_idx in range(n_windows):
                start = win_idx * epochs_per_window
                end = start + epochs_per_window

                # 提取窗口数据
                ppg_win = ppg_windows[start:end]
                labels_win = ppg_labels[start:end]
                eeg_win = eeg_epochs[start:end]

                # 准备PPG数据: (1, 1, samples)
                ppg_continuous = ppg_win.reshape(-1).astype(np.float32)
                ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0).unsqueeze(0).to(device)

                # 准备EEG数据: (1, n_epochs, signal_length)
                eeg_tensor = torch.FloatTensor(eeg_win.astype(np.float32)).unsqueeze(0).to(device)
                # EEG标准化
                eeg_mean = eeg_tensor.mean(dim=-1, keepdim=True)
                eeg_std = eeg_tensor.std(dim=-1, keepdim=True) + 1e-8
                eeg_tensor = (eeg_tensor - eeg_mean) / eeg_std

                # GPU同步后开始计时
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                # PPG推理: 输出 (1, 4, n_epochs)
                ppg_logits = ppg_model(ppg_tensor)
                ppg_probs = F.softmax(ppg_logits, dim=1)
                ppg_probs = ppg_probs[0].transpose(0, 1)  # (n_epochs, 4)

                # EEG推理: 输出 (1, n_epochs, 4)
                eeg_logits = eeg_model(eeg_tensor)
                eeg_probs = F.softmax(eeg_logits, dim=-1)
                eeg_probs = eeg_probs[0]  # (n_epochs, 4)

                # 融合
                fused_probs = alpha * ppg_probs + (1 - alpha) * eeg_probs
                preds = torch.argmax(fused_probs, dim=1).cpu().numpy()

                # GPU同步后结束计时
                if use_cuda:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                inference_times.append((end_time - start_time) * 1000)  # ms

                all_preds.extend(preds)
                all_labels.extend(labels_win)

    if skipped_subjects > 0:
        print(f"  跳过 {skipped_subjects} 个被试（数据不完整）")

    inf_mean = np.mean(inference_times) if inference_times else 0
    inf_std = np.std(inference_times) if inference_times else 0

    return np.array(all_preds), np.array(all_labels), inf_mean, inf_std


def evaluate_fusion(window_minutes, ppg_model_path, eeg_model_path,
                    ppg_path, index_path, eeg_folder, output_dir):
    """评估融合模型"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    if device.type == 'cpu':
        print("⚠️  警告: 使用CPU运行，速度较慢")
    print(f"窗口长度: {window_minutes} 分钟")

    # 加载PPG模型
    print("\n加载Dual PPG模型 (CFS微调)...")
    # 窗口大小格式转换: 30 -> "30min"
    window_size_str = f"{window_minutes}min"
    ppg_model = create_model_for_window(window_size_str).to(device)

    checkpoint = torch.load(ppg_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ppg_model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        best_kappa = checkpoint.get('best_kappa', None)
        if best_kappa is not None:
            print(f"  加载自 epoch {epoch}, best_kappa={best_kappa:.4f}")
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        ppg_model.load_state_dict(checkpoint['model'])
    else:
        ppg_model.load_state_dict(checkpoint)

    ppg_params = sum(p.numel() for p in ppg_model.parameters())
    print(f"  PPG参数量: {ppg_params:,} ({ppg_params / 1e6:.2f}M)")

    # 加载EEG模型 (MESA训练，零样本迁移)
    print("\n加载EEG模型 (MESA训练，零样本迁移)...")
    eeg_model = ShortWindowAttnSleep(
        window_minutes=window_size_str,
        num_classes=4
    ).to(device)

    eeg_checkpoint = torch.load(eeg_model_path, map_location=device)
    if isinstance(eeg_checkpoint, dict) and 'model_state_dict' in eeg_checkpoint:
        eeg_model.load_state_dict(eeg_checkpoint['model_state_dict'])
        epoch = eeg_checkpoint.get('epoch', '?')
        best_kappa = eeg_checkpoint.get('best_kappa', None)
        if best_kappa is not None:
            print(f"  加载自 epoch {epoch}, best_kappa={best_kappa:.4f}")
    else:
        eeg_model.load_state_dict(eeg_checkpoint)

    eeg_params = sum(p.numel() for p in eeg_model.parameters())
    print(f"  EEG参数量: {eeg_params:,} ({eeg_params / 1e6:.2f}M)")

    total_params = ppg_params + eeg_params
    print(f"\n总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")

    # 加载测试数据
    print("\n加载CFS测试数据...")
    test_subjects, ppg_indices, eeg_files = load_cfs_test_data(
        ppg_path, eeg_folder, index_path
    )

    if len(test_subjects) == 0:
        print("❌ 没有找到有效的测试被试！")
        return None

    # 只测试Alpha=0.3
    results = {}
    best_kappa = -1
    best_alpha = 0.3
    best_result = None
    best_cm = None

    alphas = [0.3]  # 只测试0.3

    print("\n" + "=" * 70)
    print("测试融合权重 Alpha=0.3...")
    print("=" * 70)

    class_names = ['Wake', 'Light', 'Deep', 'REM']

    for alpha in alphas:
        preds, labels, inf_mean, inf_std = fusion_inference(
            ppg_model, eeg_model, test_subjects, ppg_indices,
            eeg_files, ppg_path, device, window_minutes, alpha
        )

        if len(preds) == 0:
            continue

        acc = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        f1_macro = f1_score(labels, preds, average='macro')
        cm = confusion_matrix(labels, preds, labels=[0, 1, 2, 3])

        results[f'alpha_{alpha}'] = {
            'accuracy': float(acc),
            'kappa': float(kappa),
            'f1_macro': float(f1_macro),
            'inference_mean_ms': float(inf_mean),
            'inference_std_ms': float(inf_std),
            'confusion_matrix': cm.tolist()
        }

        # 打印每个alpha的结果
        print(f"\n{'=' * 70}")
        if alpha == 0.0:
            print(f"Alpha={alpha:.1f} (EEG Only)")
        elif alpha == 1.0:
            print(f"Alpha={alpha:.1f} (Dual PPG Only)")
        else:
            print(f"Alpha={alpha:.1f} (PPG:{alpha * 100:.0f}%, EEG:{(1 - alpha) * 100:.0f}%)")
        print(f"{'=' * 70}")
        print(f"κ={kappa:.4f}, Acc={acc * 100:.2f}%, F1={f1_macro:.4f}, Inference={inf_mean:.2f}±{inf_std:.2f}ms")

        # 打印混淆矩阵
        print(f"\n混淆矩阵:")
        print(f"{'-' * 70}")
        print(f"{'真实-预测':<10}", end="")
        for name in class_names:
            print(f"{name:<12}", end="")
        print(f"{'召回率':<10}")
        print(f"{'-' * 70}")
        for i, name in enumerate(class_names):
            print(f"{name:<10}", end="")
            for j in range(len(class_names)):
                print(f"{cm[i, j]:<12}", end="")
            recall = cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0
            print(f"{recall:.2%}")
        print(f"{'-' * 70}")

        if kappa > best_kappa:
            best_kappa = kappa
            best_alpha = alpha
            best_result = {
                'accuracy': acc,
                'kappa': kappa,
                'f1_macro': f1_macro,
                'inference_mean_ms': inf_mean,
                'inference_std_ms': inf_std
            }
            best_cm = cm

    # 打印最终最佳结果
    print(f"\n\n{'#' * 70}")
    print(f"{'#' * 70}")
    print(f"##  CFS最佳融合结果 ({window_minutes}分钟窗口)")
    print(f"{'#' * 70}")
    print(f"{'#' * 70}")
    print(f"\n最佳 Alpha: {best_alpha} (PPG:{best_alpha * 100:.0f}%, EEG:{(1 - best_alpha) * 100:.0f}%)")
    print(f"\n{'指标':<25} {'值':<20}")
    print(f"{'-' * 45}")
    print(f"{'κ (Kappa)':<25} {best_result['kappa']:.4f}")
    print(f"{'Acc (Accuracy)':<25} {best_result['accuracy'] * 100:.2f}%")
    print(f"{'F1 (Macro)':<25} {best_result['f1_macro']:.4f}")
    print(f"{'Model Size':<25} {total_params / 1e6:.2f}M")
    print(f"{'Inference (ms)':<25} {best_result['inference_mean_ms']:.2f}±{best_result['inference_std_ms']:.2f}")
    print(f"{'-' * 45}")

    # 打印最佳结果的混淆矩阵
    print(f"\n最佳结果混淆矩阵 (Alpha={best_alpha}):")
    print(f"{'-' * 70}")
    print(f"{'真实-预测':<10}", end="")
    for name in class_names:
        print(f"{name:<12}", end="")
    print(f"{'召回率':<10}")
    print(f"{'-' * 70}")
    for i, name in enumerate(class_names):
        print(f"{name:<10}", end="")
        for j in range(len(class_names)):
            print(f"{best_cm[i, j]:<12}", end="")
        recall = best_cm[i, i] / best_cm[i].sum() if best_cm[i].sum() > 0 else 0
        print(f"{recall:.2%}")
    print(f"{'-' * 70}")
    print(f"{'#' * 70}")

    # 保存结果
    final_results = {
        'dataset': 'CFS',
        'window_minutes': window_minutes,
        'best_alpha': best_alpha,
        'metrics': {
            'kappa': float(best_result['kappa']),
            'accuracy': float(best_result['accuracy']),
            'f1_macro': float(best_result['f1_macro']),
            'model_size_M': total_params / 1e6,
            'inference_mean_ms': float(best_result['inference_mean_ms']),
            'inference_std_ms': float(best_result['inference_std_ms'])
        },
        'confusion_matrix': best_cm.tolist(),
        'per_class_recall': {
            name: float(best_cm[i, i] / best_cm[i].sum()) if best_cm[i].sum() > 0 else 0
            for i, name in enumerate(class_names)
        },
        'model_params': {
            'ppg': ppg_params,
            'eeg': eeg_params,
            'total': total_params
        },
        'transfer_strategy': {
            'ppg': 'MESA pretrain -> CFS finetune',
            'eeg': 'MESA pretrain -> CFS zero-shot'
        },
        'all_alphas': results
    }

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f'cfs_fusion_{window_minutes}min_results.json')
    with open(result_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n结果已保存: {result_path}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description='CFS数据集短窗口PPG+EEG融合评估')
    parser.add_argument('--window', type=int, required=True, choices=[3, 5, 10, 30],
                        help='窗口长度（分钟）')
    parser.add_argument('--ppg_model', type=str, required=True,
                        help='CFS微调后的PPG模型路径')
    parser.add_argument('--eeg_model', type=str, required=True,
                        help='MESA训练的EEG模型路径')
    parser.add_argument('--ppg_data', type=str, default='../../data/cfs_ppg_with_labels.h5',
                        help='CFS PPG数据路径')
    parser.add_argument('--index', type=str, default='../../data/cfs_subject_index.h5',
                        help='CFS索引文件路径')
    parser.add_argument('--eeg_folder', type=str, default='../../data/cfs_eeg_c3m2_data',
                        help='CFS EEG数据文件夹')
    parser.add_argument('--output', type=str, default='./cfs_fusion_results',
                        help='输出目录')
    args = parser.parse_args()

    evaluate_fusion(
        window_minutes=args.window,
        ppg_model_path=args.ppg_model,
        eeg_model_path=args.eeg_model,
        ppg_path=args.ppg_data,
        index_path=args.index,
        eeg_folder=args.eeg_folder,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()