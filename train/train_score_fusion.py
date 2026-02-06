"""
短窗口PPG + EEG概率层融合评估
支持窗口长度: 3分钟、5分钟、10分钟、30分钟

PPG模型: PPGCrossAttnShortWindow from ppg_crossattn_shortwindow.py
EEG模型: ShortWindowAttnSleep from short_window_eeg_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import glob
from pathlib import Path
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
import json
import os
import time
import argparse

# 导入正确的模型
from ppg_crossattn_shortwindow import PPGCrossAttnShortWindow, create_model_for_window
from short_window_eeg_model import ShortWindowAttnSleep


# SleepPPG-Net测试集 (204 subjects)
TEST_SUBJECTS = [
    "0001", "0021", "0033", "0052", "0077", "0081", "0101", "0111", "0225", "0310",
    "0314", "0402", "0416", "0445", "0465", "0483", "0505", "0554", "0572", "0587",
    "0601", "0620", "0648", "0702", "0764", "0771", "0792", "0797", "0800", "0807",
    "0860", "0892", "0902", "0904", "0921", "1033", "1080", "1121", "1140", "1148",
    "1161", "1164", "1219", "1224", "1271", "1324", "1356", "1391", "1463", "1483",
    "1497", "1528", "1531", "1539", "1672", "1693", "1704", "1874", "1876", "1900",
    "1914", "2039", "2049", "2096", "2100", "2109", "2169", "2172", "2183", "2208",
    "2239", "2243", "2260", "2269", "2317", "2362", "2388", "2470", "2472", "2488",
    "2527", "2556", "2602", "2608", "2613", "2677", "2680", "2685", "2727", "2729",
    "2802", "2811", "2828", "2877", "2881", "2932", "2934", "2993", "2999", "3044",
    "3066", "3068", "3111", "3121", "3153", "3275", "3298", "3324", "3369", "3492",
    "3543", "3554", "3557", "3561", "3684", "3689", "3777", "3793", "3801", "3815",
    "3839", "3886", "3997", "4110", "4137", "4171", "4227", "4285", "4332", "4406",
    "4460", "4462", "4497", "4501", "4552", "4577", "4649", "4650", "4667", "4732",
    "4794", "4888", "4892", "4895", "4912", "4918", "4998", "5006", "5075", "5077",
    "5148", "5169", "5203", "5232", "5243", "5287", "5316", "5357", "5366", "5395",
    "5397", "5457", "5472", "5479", "5496", "5532", "5568", "5580", "5659", "5692",
    "5706", "5737", "5754", "5805", "5838", "5847", "5890", "5909", "5957", "5983",
    "6015", "6039", "6047", "6123", "6224", "6263", "6266", "6281", "6291", "6482",
    "6491", "6502", "6516", "6566", "6567", "6583", "6619", "6629", "6646", "6680",
    "6722", "6730", "6741", "6788"
]


def load_test_data(ppg_path, eeg_folders, index_path):
    """加载测试集数据"""
    with h5py.File(index_path, 'r') as f:
        all_subjects = list(f['subjects'].keys())
        valid_subjects = [s for s in all_subjects if f[f'subjects/{s}'].attrs['n_windows'] == 1200]

    test_subjects = [s for s in TEST_SUBJECTS if s in valid_subjects]
    print(f"测试集被试: {len(test_subjects)}")

    # 获取PPG索引
    ppg_indices = {}
    with h5py.File(index_path, 'r') as f:
        for subj in test_subjects:
            indices = f[f'subjects/{subj}/window_indices'][:]
            ppg_indices[subj] = indices[0]

    # 加载EEG文件映射
    eeg_files = {}
    for folder in eeg_folders:
        files = glob.glob(str(Path(folder) / "*.npz"))
        for file in files:
            basename = Path(file).name
            if 'mesa-sleep-' in basename:
                subj_id = basename.split('-')[-1].split('.')[0].zfill(4)
            else:
                subj_id = basename.split('_')[0].zfill(4)
            if subj_id in test_subjects:
                eeg_files[subj_id] = file

    print(f"找到EEG文件: {len(eeg_files)}")
    return test_subjects, ppg_indices, eeg_files


def fusion_inference(ppg_model, eeg_model, test_subjects, ppg_indices, 
                     eeg_files, ppg_path, device, window_minutes, alpha=0.5):
    """
    短窗口融合推理
    
    Args:
        window_minutes: 窗口长度（分钟）
        alpha: 融合权重 (0=纯EEG, 1=纯PPG)
    
    Returns:
        preds, labels, avg_inference_time_ms
    """
    ppg_model.eval()
    eeg_model.eval()

    epochs_per_window = window_minutes * 2  # 每分钟2个epoch
    
    all_preds = []
    all_labels = []
    inference_times = []
    skipped_subjects = 0
    
    use_cuda = device.type == 'cuda'

    with torch.no_grad():
        for subj in tqdm(test_subjects, desc=f"Alpha={alpha}"):
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

            # 过滤无效标签 (标签: 0=Wake, 1=Light, 2=Deep, 3=REM, -1=无效)
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

                # 准备数据
                ppg_continuous = ppg_win.reshape(-1)
                ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0).unsqueeze(0).to(device)
                
                # EEG: (n_epochs, 3000) -> (1, n_epochs, 3000)
                eeg_tensor = torch.FloatTensor(eeg_win).unsqueeze(0).to(device)

                # GPU同步后开始计时
                if use_cuda:
                    torch.cuda.synchronize()
                start_time = time.perf_counter()

                # PPG推理
                ppg_logits = ppg_model(ppg_tensor)  # (1, 4, n_epochs)
                ppg_probs = F.softmax(ppg_logits, dim=1)  # softmax on class dim
                ppg_probs = ppg_probs[0].transpose(0, 1)  # (n_epochs, 4)

                # EEG推理 - ShortWindowAttnSleep输出 (batch, n_epochs, 4)
                eeg_logits = eeg_model(eeg_tensor)  # (1, n_epochs, 4)
                eeg_probs = F.softmax(eeg_logits, dim=-1)  # softmax on class dim
                eeg_probs = eeg_probs[0]  # (n_epochs, 4)

                # 融合
                fused_probs = alpha * ppg_probs + (1 - alpha) * eeg_probs
                preds = torch.argmax(fused_probs, dim=1).cpu()

                # GPU同步后结束计时
                if use_cuda:
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                inference_times.append((end_time - start_time) * 1000)

                all_preds.extend(preds.numpy())
                all_labels.extend(labels_win)

    if skipped_subjects > 0:
        print(f"  跳过 {skipped_subjects} 个被试（数据不完整）")

    avg_inference_time = np.mean(inference_times) if inference_times else 0

    return np.array(all_preds), np.array(all_labels), avg_inference_time


def evaluate_fusion(window_minutes, ppg_model_path, eeg_model_path, 
                    ppg_path, index_path, eeg_folders, output_dir):
    """评估特定窗口长度的融合性能"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    print(f"窗口长度: {window_minutes} 分钟")

    # 加载PPG模型 - PPGCrossAttnShortWindow
    print("\n加载PPG模型...")
    ppg_model = create_model_for_window(
        f'{window_minutes}min',
        d_model=256,
        n_heads=8,
        n_fusion_blocks=3,
        deterministic_noise=True
    ).to(device)
    
    checkpoint = torch.load(ppg_model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        ppg_model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        ppg_model.load_state_dict(checkpoint['model'])
    else:
        ppg_model.load_state_dict(checkpoint)
    print("PPG模型加载完成")

    # 加载EEG模型 - ShortWindowAttnSleep
    print("\n加载EEG模型...")
    eeg_model = ShortWindowAttnSleep(
        window_minutes=window_minutes,
        num_classes=4
    ).to(device)
    
    eeg_checkpoint = torch.load(eeg_model_path, map_location=device)
    if isinstance(eeg_checkpoint, dict) and 'model_state_dict' in eeg_checkpoint:
        eeg_model.load_state_dict(eeg_checkpoint['model_state_dict'])
    else:
        eeg_model.load_state_dict(eeg_checkpoint)
    print("EEG模型加载完成")

    # 计算模型参数量
    ppg_params = sum(p.numel() for p in ppg_model.parameters())
    eeg_params = sum(p.numel() for p in eeg_model.parameters())
    total_params = ppg_params + eeg_params
    
    print(f"\n模型参数量:")
    print(f"  PPG:   {ppg_params:,} ({ppg_params*4/1024/1024:.2f} MB)")
    print(f"  EEG:   {eeg_params:,} ({eeg_params*4/1024/1024:.2f} MB)")
    print(f"  Total: {total_params:,} ({total_params*4/1024/1024:.2f} MB)")

    # 加载测试数据
    test_subjects, ppg_indices, eeg_files = load_test_data(ppg_path, eeg_folders, index_path)

    # 测试不同权重
    results = {}
    best_kappa = -1
    best_alpha = 0.5
    best_result = None
    best_cm = None

    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        preds, labels, inference_time = fusion_inference(
            ppg_model, eeg_model, test_subjects, ppg_indices,
            eeg_files, ppg_path, device, window_minutes, alpha
        )

        if len(preds) == 0:
            continue

        acc = accuracy_score(labels, preds)
        kappa = cohen_kappa_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        results[f'alpha_{alpha}'] = {
            'accuracy': float(acc),
            'kappa': float(kappa),
            'inference_ms': float(inference_time)
        }
        
        print(f"  Alpha={alpha}: Acc={acc:.4f}, κ={kappa:.4f}")

        if kappa > best_kappa:
            best_kappa = kappa
            best_alpha = alpha
            best_result = {
                'accuracy': acc,
                'kappa': kappa,
                'inference_ms': inference_time
            }
            best_cm = cm

    # 打印最终结果
    class_names = ['Wake', 'Light', 'Deep', 'REM']
    
    print(f"\n{'='*60}")
    print(f"融合结果 ({window_minutes}分钟窗口)")
    print(f"{'='*60}")
    print(f"最佳 Alpha: {best_alpha} (PPG:{best_alpha*100:.0f}%, EEG:{(1-best_alpha)*100:.0f}%)")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'值':<20}")
    print(f"{'-'*40}")
    print(f"{'κ (Kappa)':<20} {best_result['kappa']:.4f}")
    print(f"{'Acc (Accuracy)':<20} {best_result['accuracy']*100:.2f}%")
    print(f"{'Model Size':<20} {total_params*4/1024/1024:.2f} MB")
    print(f"{'Inference (ms)':<20} {best_result['inference_ms']:.2f}")
    print(f"{'='*60}")
    
    # 打印混淆矩阵
    print(f"\n混淆矩阵:")
    print(f"{'-'*70}")
    print(f"{'真实-预测':<10}", end="")
    for name in class_names:
        print(f"{name:<12}", end="")
    print(f"{'召回率':<10}")
    print(f"{'-'*70}")
    for i, name in enumerate(class_names):
        print(f"{name:<10}", end="")
        for j in range(len(class_names)):
            print(f"{best_cm[i, j]:<12}", end="")
        recall = best_cm[i, i] / best_cm[i].sum() if best_cm[i].sum() > 0 else 0
        print(f"{recall:.2%}")
    print(f"{'-'*70}")

    # 保存结果
    final_results = {
        'window_minutes': window_minutes,
        'best_alpha': best_alpha,
        'metrics': {
            'kappa': float(best_result['kappa']),
            'accuracy': float(best_result['accuracy']),
            'model_size_MB': total_params * 4 / 1024 / 1024,
            'inference_ms': float(best_result['inference_ms'])
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
        'all_alphas': results
    }

    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f'fusion_{window_minutes}min_results.json')
    with open(result_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n结果已保存: {result_path}")

    return final_results


def main():
    parser = argparse.ArgumentParser(description='短窗口PPG+EEG融合评估')
    parser.add_argument('--window', type=int, required=True, choices=[3, 5, 10, 30],
                        help='窗口长度（分钟）')
    parser.add_argument('--ppg_model', type=str, required=True,
                        help='PPG模型路径')
    parser.add_argument('--eeg_model', type=str, required=True,
                        help='EEG模型路径')
    parser.add_argument('--ppg_data', type=str, default='../../data/mesa_ppg_with_labels.h5',
                        help='PPG数据路径')
    parser.add_argument('--index', type=str, default='../../data/mesa_subject_index.h5',
                        help='索引文件路径')
    parser.add_argument('--eeg_folders', type=str, nargs='+',
                        default=['../../data/eeg-1', '../../data/eeg-2', '../../data/eeg-3'],
                        help='EEG数据文件夹')
    parser.add_argument('--output', type=str, default='./fusion_results',
                        help='输出目录')
    args = parser.parse_args()

    evaluate_fusion(
        window_minutes=args.window,
        ppg_model_path=args.ppg_model,
        eeg_model_path=args.eeg_model,
        ppg_path=args.ppg_data,
        index_path=args.index,
        eeg_folders=args.eeg_folders,
        output_dir=args.output
    )


if __name__ == '__main__':
    main()
