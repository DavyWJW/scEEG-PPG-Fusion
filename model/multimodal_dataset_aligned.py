"""
多模态睡眠数据集类 - 与SleepPPG-Net测试集对齐版本
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os

# SleepPPG-Net使用的204个测试用户
SLEEPPPG_TEST_SUBJECTS = [
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


class MultiModalSleepDataset(Dataset):
    def __init__(self, data_path, split='train', use_generated_ecg=False,
                 transform=None, seed=42, use_sleepppg_test_set=True):
        """
        多模态睡眠数据集 - 返回10小时连续信号

        Args:
            data_path: 数据目录路径或文件路径字典
            split: 'train', 'val', or 'test'
            use_generated_ecg: 是否使用生成的ECG
            transform: 数据增强
            seed: 随机种子
            use_sleepppg_test_set: 是否使用SleepPPG-Net的测试集划分
        """
        self.split = split
        self.use_generated_ecg = use_generated_ecg
        self.transform = transform
        self.seed = seed
        self.use_sleepppg_test_set = use_sleepppg_test_set

        # 每个受试者的窗口数和样本数
        self.windows_per_subject = 1200
        self.samples_per_window = 1024
        self.total_samples = self.windows_per_subject * self.samples_per_window

        # 支持两种输入方式
        if isinstance(data_path, dict):
            self.ppg_file_path = data_path['ppg']
            self.ecg_file_path = data_path.get('ecg', data_path.get('real_ecg'))
            self.index_file_path = data_path['index']
        else:
            self.ppg_file_path = os.path.join(data_path, 'mesa_ppg_with_labels.h5')
            if use_generated_ecg:
                self.ecg_file_path = os.path.join(data_path, 'mesa_generated_ecg.h5')
            else:
                self.ecg_file_path = os.path.join(data_path, 'mesa_real_ecg.h5')
            self.index_file_path = os.path.join(data_path, 'mesa_subject_index.h5')

        # 检查文件是否存在
        if not os.path.exists(self.ppg_file_path):
            raise FileNotFoundError(f"PPG file not found: {self.ppg_file_path}")
        if not os.path.exists(self.ecg_file_path):
            raise FileNotFoundError(f"ECG file not found: {self.ecg_file_path}")
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file not found: {self.index_file_path}")

        print(f"Loading data from:")
        print(f"  PPG: {self.ppg_file_path}")
        print(f"  ECG: {self.ecg_file_path}")
        print(f"  Index: {self.index_file_path}")

        # 准备受试者列表
        self._prepare_subjects()

    def _prepare_subjects(self):
        """准备受试者列表"""
        # 获取所有受试者
        with h5py.File(self.index_file_path, 'r') as f:
            all_subjects = list(f['subjects'].keys())

            # 只保留有完整1200个窗口的受试者
            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows == self.windows_per_subject:
                    valid_subjects.append(subj)

        if self.use_sleepppg_test_set:
            # 使用SleepPPG-Net的测试集划分
            test_subjects = [s for s in SLEEPPPG_TEST_SUBJECTS if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]

            # 打印统计信息
            print(f"Using SleepPPG-Net test set split:")
            print(f"  Total valid subjects: {len(valid_subjects)}")
            print(f"  Test subjects (from SleepPPG-Net): {len(test_subjects)}")
            print(f"  Train+Val subjects: {len(train_val_subjects)}")

            # 验证是否所有204个测试用户都存在
            missing_subjects = [s for s in SLEEPPPG_TEST_SUBJECTS if s not in valid_subjects]
            if missing_subjects:
                print(f"  WARNING: {len(missing_subjects)} test subjects not found in data:")
                print(f"    {missing_subjects[:10]}..." if len(missing_subjects) > 10 else f"    {missing_subjects}")

            # 进一步划分训练集和验证集
            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            # 原始的随机划分
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=0.2, random_state=self.seed
            )
            train_subjects, val_subjects = train_test_split(
                train_subjects, test_size=0.2, random_state=self.seed
            )

        # 选择对应的受试者
        if self.split == 'train':
            self.subjects = train_subjects
        elif self.split == 'val':
            self.subjects = val_subjects
        else:  # test
            self.subjects = test_subjects

        print(f"{self.split} set: {len(self.subjects)} subjects")

        # 获取每个受试者的起始索引
        self.subject_indices = {}
        with h5py.File(self.index_file_path, 'r') as f:
            for subj in self.subjects:
                indices = f[f'subjects/{subj}/window_indices'][:]
                if len(indices) == self.windows_per_subject:
                    self.subject_indices[subj] = indices[0]  # 起始索引

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        # 获取受试者ID和起始索引
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]

        # 读取该受试者的所有窗口
        with h5py.File(self.ppg_file_path, 'r') as f_ppg:
            ppg_windows = f_ppg['ppg'][start_idx:start_idx + self.windows_per_subject]
            labels = f_ppg['labels'][start_idx:start_idx + self.windows_per_subject]

        with h5py.File(self.ecg_file_path, 'r') as f_ecg:
            ecg_windows = f_ecg['ecg'][start_idx:start_idx + self.windows_per_subject]

        # 将窗口拼接成连续信号
        ppg_continuous = ppg_windows.reshape(-1)
        ecg_continuous = ecg_windows.reshape(-1)

        # 数据增强
        if self.transform:
            ppg_continuous, ecg_continuous = self.transform(ppg_continuous, ecg_continuous)

        # 转换为tensor
        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        ecg_tensor = torch.FloatTensor(ecg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, ecg_tensor, labels_tensor


class PPGOnlyDataset(Dataset):
    """仅PPG的数据集 - 10小时连续信号版本"""

    def __init__(self, data_path, split='train', transform=None, seed=42,
                 use_sleepppg_test_set=True):
        self.split = split
        self.transform = transform
        self.seed = seed
        self.use_sleepppg_test_set = use_sleepppg_test_set

        self.windows_per_subject = 1200
        self.samples_per_window = 1024
        self.total_samples = self.windows_per_subject * self.samples_per_window

        # 支持两种输入方式
        if isinstance(data_path, dict):
            self.ppg_file_path = data_path['ppg']
            self.index_file_path = data_path['index']
        else:
            self.ppg_file_path = os.path.join(data_path, 'mesa_ppg_with_labels.h5')
            self.index_file_path = os.path.join(data_path, 'mesa_subject_index.h5')

        # 检查文件存在
        if not os.path.exists(self.ppg_file_path):
            raise FileNotFoundError(f"PPG file not found: {self.ppg_file_path}")
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file not found: {self.index_file_path}")

        print(f"Loading PPG data from: {self.ppg_file_path}")

        self._prepare_subjects()

    def _prepare_subjects(self):
        """准备受试者列表"""
        with h5py.File(self.index_file_path, 'r') as f:
            all_subjects = list(f['subjects'].keys())

            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows == self.windows_per_subject:
                    valid_subjects.append(subj)

        if self.use_sleepppg_test_set:
            # 使用SleepPPG-Net的测试集划分
            test_subjects = [s for s in SLEEPPPG_TEST_SUBJECTS if s in valid_subjects]
            train_val_subjects = [s for s in valid_subjects if s not in test_subjects]

            print(f"Using SleepPPG-Net test set split:")
            print(f"  Test subjects: {len(test_subjects)}")
            print(f"  Train+Val subjects: {len(train_val_subjects)}")

            # 进一步划分训练集和验证集
            train_subjects, val_subjects = train_test_split(
                train_val_subjects, test_size=0.2, random_state=self.seed
            )
        else:
            # 原始的随机划分
            train_subjects, test_subjects = train_test_split(
                valid_subjects, test_size=0.2, random_state=self.seed
            )
            train_subjects, val_subjects = train_test_split(
                train_subjects, test_size=0.2, random_state=self.seed
            )

        if self.split == 'train':
            self.subjects = train_subjects
        elif self.split == 'val':
            self.subjects = val_subjects
        else:
            self.subjects = test_subjects

        print(f"{self.split} set: {len(self.subjects)} subjects")

        # 获取起始索引
        self.subject_indices = {}
        with h5py.File(self.index_file_path, 'r') as f:
            for subj in self.subjects:
                indices = f[f'subjects/{subj}/window_indices'][:]
                if len(indices) == self.windows_per_subject:
                    self.subject_indices[subj] = indices[0]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        start_idx = self.subject_indices[subject_id]

        with h5py.File(self.ppg_file_path, 'r') as f:
            ppg_windows = f['ppg'][start_idx:start_idx + self.windows_per_subject]
            labels = f['labels'][start_idx:start_idx + self.windows_per_subject]

        # 拼接成连续信号
        ppg_continuous = ppg_windows.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor


def get_dataloaders(data_path, batch_size=1, num_workers=0,
                    use_generated_ecg=False, model_type='ppg_only',
                    use_sleepppg_test_set=True):
    """
    获取数据加载器

    Args:
        use_sleepppg_test_set: 是否使用SleepPPG-Net的204个测试用户
    """

    if model_type == 'ppg_only':
        train_dataset = PPGOnlyDataset(
            data_path, split='train',
            use_sleepppg_test_set=use_sleepppg_test_set
        )
        val_dataset = PPGOnlyDataset(
            data_path, split='val',
            use_sleepppg_test_set=use_sleepppg_test_set
        )
        test_dataset = PPGOnlyDataset(
            data_path, split='test',
            use_sleepppg_test_set=use_sleepppg_test_set
        )
    else:
        train_dataset = MultiModalSleepDataset(
            data_path, split='train',
            use_generated_ecg=use_generated_ecg,
            use_sleepppg_test_set=use_sleepppg_test_set
        )
        val_dataset = MultiModalSleepDataset(
            data_path, split='val',
            use_generated_ecg=use_generated_ecg,
            use_sleepppg_test_set=use_sleepppg_test_set
        )
        test_dataset = MultiModalSleepDataset(
            data_path, split='test',
            use_generated_ecg=use_generated_ecg,
            use_sleepppg_test_set=use_sleepppg_test_set
        )

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


def verify_test_set():
    """验证测试集是否与SleepPPG-Net一致"""
    data_paths = {
        'ppg': "../../data/mesa_ppg_with_labels.h5",
        'real_ecg': "../../data/mesa_real_ecg.h5",
        'index': "../../data/mesa_subject_index.h5"
    }

    # 创建测试数据集
    test_dataset = PPGOnlyDataset(
        data_paths, split='test',
        use_sleepppg_test_set=True
    )

    print(f"\nTest set verification:")
    print(f"Number of test subjects: {len(test_dataset.subjects)}")
    print(f"Test subjects (first 10): {test_dataset.subjects[:10]}")

    # 检查是否与SleepPPG-Net的测试集一致
    matched = set(test_dataset.subjects) == set(SLEEPPPG_TEST_SUBJECTS)
    print(f"Matches SleepPPG-Net test set: {matched}")

    if not matched:
        missing = set(SLEEPPPG_TEST_SUBJECTS) - set(test_dataset.subjects)
        extra = set(test_dataset.subjects) - set(SLEEPPPG_TEST_SUBJECTS)
        if missing:
            print(f"Missing subjects: {missing}")
        if extra:
            print(f"Extra subjects: {extra}")


if __name__ == "__main__":
    verify_test_set()