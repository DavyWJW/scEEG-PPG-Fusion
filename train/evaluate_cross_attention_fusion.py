"""
Evaluate EEG-PPG Cross-Attention Fusion Model

用法:
python evaluate_cross_attention_fusion.py \
    --fusion_model ./outputs/cross_attention_fusion/best_model.pth \
    --eeg_model ../../work/mesa-short_window/outputs/eeg_window_3min/best_model.pth \
    --ppg_model ../../work/mesa-short_window/outputs/short_window/3min_2/best_model.pth \
    --output_dir ./outputs/cross_attention_fusion/evaluation
"""

import os
import sys
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    cohen_kappa_score, accuracy_score, confusion_matrix, 
    classification_report, precision_recall_fscore_support
)
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# 添加模型路径
sys.path.insert(0, '../../work/mesa-short_window')

from cross_attention_fusion import create_fusion_model, EEGPPGCrossAttentionFusion


class FusionTestDataset(Dataset):
    """EEG-PPG融合测试数据集"""
    
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
        self.epochs_per_window = window_minutes * 2
        self.samples_per_epoch = samples_per_epoch
        
        # 构建EEG文件路径映射
        self.eeg_file_map = {}
        for folder in eeg_folders:
            if os.path.exists(folder):
                for fname in os.listdir(folder):
                    if fname.endswith('.npz'):
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
        
        print(f"Loaded {len(self.samples)} test samples from {len(subject_ids)} subjects")
    
    def _build_samples(self, subject_ids):
        """构建测试样本索引"""
        matched_subjects = 0
        
        for sid in subject_ids:
            sid_str = str(sid).zfill(4) if isinstance(sid, int) else sid
            
            # 检查EEG数据
            if sid_str not in self.eeg_file_map:
                continue
            eeg_path = self.eeg_file_map[sid_str]
            
            # 检查PPG索引数据
            ppg_key = f'subjects/{sid_str}/window_indices'
            if ppg_key not in self.ppg_index_h5:
                continue
            
            matched_subjects += 1
            
            # 加载EEG数据
            eeg_data = np.load(eeg_path)
            eeg_signals = eeg_data['x']
            eeg_labels = eeg_data['y']
            
            # 加载PPG索引
            ppg_indices = self.ppg_index_h5[ppg_key][:]
            
            # 确保数据长度匹配
            n_epochs = min(len(eeg_signals), len(ppg_indices))
            
            # 标签映射
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
            
            # 创建窗口样本
            n_windows = n_epochs // self.epochs_per_window
            
            for win_idx in range(n_windows):
                start_epoch = win_idx * self.epochs_per_window
                end_epoch = start_epoch + self.epochs_per_window
                
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
        
        print(f"  Matched subjects: {matched_subjects}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 加载EEG数据
        eeg_data = np.load(sample['eeg_path'])
        eeg_signals = eeg_data['x']
        start = sample['eeg_start']
        end = start + self.epochs_per_window
        eeg_window = eeg_signals[start:end]
        
        # Z-score标准化
        eeg_window = (eeg_window - eeg_window.mean()) / (eeg_window.std() + 1e-8)
        
        # 加载PPG数据
        ppg_indices = sample['ppg_indices']
        ppg_epochs = self.ppg_data[ppg_indices]
        ppg_window = ppg_epochs.flatten()
        
        # Z-score标准化
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


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_subjects = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            eeg = batch['eeg'].to(device)
            ppg = batch['ppg'].to(device)
            labels = batch['labels']
            subjects = batch['subject_id']
            
            outputs = model(eeg, ppg)
            preds = outputs.argmax(dim=-1).cpu().numpy()
            
            # 展平
            for i in range(len(preds)):
                all_preds.extend(preds[i])
                all_labels.extend(labels[i].numpy())
                all_subjects.extend([subjects[i]] * len(preds[i]))
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return all_preds, all_labels, all_subjects


def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    
    # 计算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # 创建标注文本
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - Cross-Attention Fusion (3min)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fusion_model', type=str, required=True,
                        help='Path to trained fusion model')
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
                        default='./outputs/cross_attention_fusion/evaluation',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("EEG-PPG Cross-Attention Fusion Evaluation")
    print("=" * 60)
    
    # 创建融合模型
    print("\nLoading models...")
    model = create_fusion_model(
        eeg_model_path=args.eeg_model,
        ppg_model_path=args.ppg_model,
        device=args.device,
        n_fusion_blocks=2,
        freeze_encoders=True
    )
    
    # 加载训练好的融合层权重
    checkpoint = torch.load(args.fusion_model, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded fusion model from {args.fusion_model}")
    print(f"  Best validation Kappa: {checkpoint.get('val_kappa', 'N/A')}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eeg_params = sum(p.numel() for p in model.eeg_model.parameters())
    ppg_params = sum(p.numel() for p in model.ppg_model.parameters())
    fusion_params = total_params - eeg_params - ppg_params
    
    print(f"\nModel Parameters:")
    print(f"  EEG Encoder: {eeg_params/1e6:.2f}M")
    print(f"  PPG Encoder: {ppg_params/1e6:.2f}M")
    print(f"  Fusion Layers: {fusion_params/1e6:.2f}M")
    print(f"  Total: {total_params/1e6:.2f}M")
    
    # 测量推理时间
    print("\nMeasuring inference time...")
    model.eval()
    dummy_eeg = torch.randn(1, 6, 3000).to(args.device)
    dummy_ppg = torch.randn(1, 6144).to(args.device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_eeg, dummy_ppg)
    
    # 测量多次
    import time
    n_runs = 100
    times = []
    
    with torch.no_grad():
        for _ in range(n_runs):
            if args.device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            _ = model(dummy_eeg, dummy_ppg)
            
            if args.device == 'cuda':
                torch.cuda.synchronize()
            
            times.append((time.time() - start_time) * 1000)  # ms
    
    inference_mean = np.mean(times)
    inference_std = np.std(times)
    print(f"  Inference time: {inference_mean:.2f} ms ± {inference_std:.2f} ms per window")
    
    # 加载测试集
    print("\nLoading test dataset...")
    with open(args.testset_json, 'r') as f:
        test_subjects = json.load(f)
    
    test_dataset = FusionTestDataset(
        eeg_folders=args.eeg_folders,
        ppg_h5_path=args.ppg_h5_path,
        ppg_index_path=args.ppg_index_path,
        subject_ids=test_subjects,
        window_minutes=3
    )
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)
    
    # 评估
    print("\nEvaluating on test set...")
    preds, labels, subjects = evaluate(model, test_loader, args.device)
    
    # 计算指标
    accuracy = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, labels=[0, 1, 2, 3]
    )
    
    macro_f1 = precision_recall_fscore_support(labels, preds, average='macro')[2]
    
    # 打印结果
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    print("\nPer-class metrics:")
    class_names = ['Wake', 'Light', 'Deep', 'REM']
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}")
    print("-" * 56)
    for i, name in enumerate(class_names):
        print(f"{name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # 保存结果
    results = {
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
        'model_path': args.fusion_model,
        'model_params': {
            'eeg_encoder': eeg_params,
            'ppg_encoder': ppg_params,
            'fusion_layers': fusion_params,
            'total': total_params,
            'total_M': round(total_params / 1e6, 2)
        },
        'inference_time_ms': {
            'mean': round(inference_mean, 2),
            'std': round(inference_std, 2)
        }
    }
    
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制混淆矩阵
    plot_confusion_matrix(
        cm, class_names,
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # 保存详细分类报告
    report = classification_report(labels, preds, target_names=class_names)
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Cross-Attention Fusion Model - Test Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    print(f"\nResults saved to {args.output_dir}")
    
    # 对比之前的结果
    print("\n" + "=" * 70)
    print("COMPARISON WITH PREVIOUS METHODS")
    print("=" * 70)
    print(f"{'Method':<30} {'Kappa':<10} {'Params':<12} {'Inference (ms)':<15}")
    print("-" * 70)
    print(f"{'EEG-only (3min)':<30} {'0.706':<10} {'2.71M':<12} {'<2':<15}")
    print(f"{'PPG-only (3min)':<30} {'0.643':<10} {'10.26M':<12} {'<2':<15}")
    print(f"{'Score-level Fusion (α=0.3)':<30} {'0.727':<10} {'12.97M':<12} {'<15':<15}")
    inference_str = f"{inference_mean:.2f} ± {inference_std:.2f}"
    print(f"{'Cross-Attention Fusion':<30} {kappa:<10.4f} {total_params/1e6:<12.2f}M {inference_str:<15}")
    print("-" * 70)
    improvement = (kappa - 0.727) / 0.727 * 100
    print(f"Improvement over score-level fusion: {improvement:+.2f}%")


if __name__ == "__main__":
    main()
