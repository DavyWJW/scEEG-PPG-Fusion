"""
短窗口EEG数据集类
支持不同长度的窗口：3分钟、5分钟、10分钟、30分钟
将连续的30秒epochs组合成更长的窗口
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import glob
from tqdm import tqdm


class ShortWindowEEGDataset(Dataset):
    """
    短窗口EEG数据集 - 懒加载版本
    将连续的30秒EEG epochs组合成更长的窗口（30秒、1分钟、3分钟、5分钟、10分钟、30分钟）
    """

    def __init__(self, npz_files, window_minutes=5):
        """
        Args:
            npz_files: npz文件列表
            window_minutes: 窗口长度（分钟）: 0.5(30s), 1, 3, 5, 10, 30
        """
        self.window_minutes = window_minutes
        self.epochs_per_window = int(window_minutes * 2)  # 每分钟2个30秒epoch
        self.npz_files = npz_files

        print(f"\nShort Window EEG Configuration:")
        print(f"  Window length: {window_minutes} minutes ({self.epochs_per_window} epochs)")
        print(f"  Epochs per window: {self.epochs_per_window}")

        # 修正的MESA标签映射
        self.mesa_label_map = {
            '0': 0,  # Wake
            '1': 1,  # Light Sleep (NREM1)
            '2': 1,  # Light Sleep (NREM2)
            '3': 2,  # Deep Sleep (NREM3)
            '4': 3,  # REM
            '9': 4,  # Unknown
        }

        # 预扫描文件，建立窗口索引（不加载数据）
        self.window_index = []  # [(file_path, start_epoch_idx), ...]

        print(f"Scanning {len(npz_files)} EEG files...")

        all_labels_sample = []

        for npz_file in tqdm(npz_files, desc="Indexing files"):
            try:
                # 只读取必要的元数据
                data = np.load(npz_file, mmap_mode='r')

                # 获取标签
                y = data['y']

                # 转换标签并检查有效性
                valid_count = 0
                valid_indices = []

                for idx, label in enumerate(y):
                    if isinstance(label, (int, float, np.integer, np.floating)):
                        label_str = str(int(float(label)))
                    else:
                        label_str = str(label).strip()

                    if label_str in self.mesa_label_map and self.mesa_label_map[label_str] < 4:
                        valid_indices.append(idx)
                        if len(all_labels_sample) < 10000:  # 采样一些标签用于统计
                            all_labels_sample.append(self.mesa_label_map[label_str])

                # 计算可以创建多少个窗口
                n_valid = len(valid_indices)
                if n_valid >= self.epochs_per_window:
                    n_windows = n_valid // self.epochs_per_window

                    for i in range(n_windows):
                        self.window_index.append((npz_file, i))

                # 不保持文件打开
                del data

            except Exception as e:
                continue

        print(f"\nTotal windows indexed: {len(self.window_index)}")

        if len(self.window_index) == 0:
            raise RuntimeError("No valid windows created!")

        # 统计类别分布（基于采样）
        if all_labels_sample:
            class_names = ['Wake', 'Light', 'Deep', 'REM']
            class_counts = np.bincount(all_labels_sample, minlength=4)

            print(f"\nLabel distribution (sampled):")
            for i, (name, count) in enumerate(zip(class_names, class_counts)):
                if count > 0:
                    print(f"  {name}: {count} samples ({count / len(all_labels_sample) * 100:.2f}%)")

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, idx):
        npz_file, window_idx = self.window_index[idx]

        # 懒加载：只在需要时读取数据
        data = np.load(npz_file)

        # 获取数据
        if 'x_psd' in data:
            x = data['x_psd']
        elif 'x' in data:
            x = data['x']
        else:
            raise KeyError(f"No valid data key found in {npz_file}")

        y = data['y']

        # 转换标签并过滤
        valid_x = []
        valid_y = []

        for i, label in enumerate(y):
            if isinstance(label, (int, float, np.integer, np.floating)):
                label_str = str(int(float(label)))
            else:
                label_str = str(label).strip()

            if label_str in self.mesa_label_map:
                mapped_label = self.mesa_label_map[label_str]
                if mapped_label < 4:  # 排除unknown
                    valid_x.append(x[i])
                    valid_y.append(mapped_label)

        valid_x = np.array(valid_x)
        valid_y = np.array(valid_y, dtype=np.int64)

        # 提取特定窗口
        start_idx = window_idx * self.epochs_per_window
        end_idx = start_idx + self.epochs_per_window

        window_x = valid_x[start_idx:end_idx]  # (epochs_per_window, signal_length)
        window_y = valid_y[start_idx:end_idx]  # (epochs_per_window,)

        data.close()

        # 转换为tensor
        x_tensor = torch.from_numpy(window_x.astype(np.float32))
        y_tensor = torch.from_numpy(window_y.astype(np.int64))

        # 标准化每个epoch
        x_mean = x_tensor.mean(dim=1, keepdim=True)
        x_std = x_tensor.std(dim=1, keepdim=True) + 1e-8
        x_tensor = (x_tensor - x_mean) / x_std

        return x_tensor, y_tensor


def load_eeg_files_by_split(data_folders, testset_path='testset.json', val_ratio=0.2):
    """
    根据testset.json划分训练、验证、测试文件

    Args:
        data_folders: 数据文件夹列表
        testset_path: 测试集配置文件路径
        val_ratio: 验证集比例

    Returns:
        train_files, val_files, test_files
    """
    # 加载测试集ID
    if os.path.exists(testset_path):
        with open(testset_path, 'r') as f:
            test_json = json.load(f)
        if isinstance(test_json, list):
            test_subject_ids = [str(x).zfill(4) for x in test_json]
        elif "test_subjects" in test_json:
            test_subject_ids = [str(x).zfill(4) for x in test_json["test_subjects"]]
        else:
            test_subject_ids = [str(x).zfill(4) for x in test_json.values()]
        print(f"Loaded {len(test_subject_ids)} test subject IDs from {testset_path}")
    else:
        print(f"Warning: {testset_path} not found, using random split")
        test_subject_ids = []

    # 收集所有文件
    all_files = []
    for folder in data_folders:
        if os.path.exists(folder):
            files = glob.glob(os.path.join(folder, '*.npz'))
            all_files.extend(files)
            print(f"Found {len(files)} files in {folder}")

    print(f"Total files found: {len(all_files)}")

    # 按被试ID分组
    subject_files = {}
    for f in all_files:
        basename = os.path.basename(f)
        # 提取被试ID
        # 支持格式: mesa-sleep-0001.npz, 0001_xxx.npz, subject_0001.npz等

        # 尝试不同的解析方式
        subject_id = None

        # 格式1: mesa-sleep-XXXX.npz
        if 'mesa-sleep-' in basename:
            parts = basename.replace('.npz', '').split('-')
            if len(parts) >= 3:
                subject_id = parts[-1].zfill(4)

        # 格式2: XXXX_xxx.npz
        if subject_id is None and '_' in basename:
            subject_id = basename.split('_')[0].zfill(4)

        # 格式3: 纯数字开头
        if subject_id is None:
            # 尝试提取数字部分
            import re
            match = re.search(r'(\d+)', basename)
            if match:
                subject_id = match.group(1).zfill(4)

        if subject_id is None:
            subject_id = basename.replace('.npz', '').zfill(4)

        if subject_id not in subject_files:
            subject_files[subject_id] = []
        subject_files[subject_id].append(f)

    print(f"Total subjects: {len(subject_files)}")

    # 划分数据集
    test_files = []
    train_val_files = []

    for subject_id, files in subject_files.items():
        if subject_id in test_subject_ids:
            test_files.extend(files)
        else:
            train_val_files.extend(files)

    # 划分训练和验证集
    np.random.seed(42)
    np.random.shuffle(train_val_files)
    val_size = int(len(train_val_files) * val_ratio)
    val_files = train_val_files[:val_size]
    train_files = train_val_files[val_size:]

    print(f"\nData split:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")
    print(f"  Test files: {len(test_files)}")

    return train_files, val_files, test_files


def get_short_window_eeg_dataloaders(data_folders, window_minutes=5, batch_size=32,
                                     num_workers=4, testset_path='testset.json'):
    """
    获取短窗口EEG数据加载器
    """
    train_files, val_files, test_files = load_eeg_files_by_split(
        data_folders, testset_path
    )

    train_dataset = ShortWindowEEGDataset(train_files, window_minutes=window_minutes)
    val_dataset = ShortWindowEEGDataset(val_files, window_minutes=window_minutes)
    test_dataset = ShortWindowEEGDataset(test_files, window_minutes=window_minutes)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # 测试数据集
    data_folders = [
        '../../data/eeg-1',
        '../../data/eeg-2',
        '../../data/eeg-3'
    ]

    for window_min in [3, 5, 10, 30]:
        print(f"\n{'=' * 60}")
        print(f"Testing {window_min}-minute windows")
        print('=' * 60)

        train_files, val_files, test_files = load_eeg_files_by_split(data_folders)

        if len(test_files) > 0:
            dataset = ShortWindowEEGDataset(test_files[:10], window_minutes=window_min)

            if len(dataset) > 0:
                x, y = dataset[0]
                print(f"X shape: {x.shape}")  # (epochs_per_window, signal_length)
                print(f"Y shape: {y.shape}")  # (epochs_per_window,)
                print(f"Expected epochs: {window_min * 2}")