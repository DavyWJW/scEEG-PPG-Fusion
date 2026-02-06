"""
CFS dataset loader — compatible with MESA format
Used to evaluate model generalization on the CFS dataset
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import os


class CFSPPGOnlyDataset(Dataset):
    """CFS PPG dataset — 10-hour continuous signal version"""

    def __init__(self, data_path, split='train', transform=None, seed=42):
        self.split = split
        self.transform = transform
        self.seed = seed

        self.windows_per_subject = 1200
        self.samples_per_window = 1024
        self.total_samples = self.windows_per_subject * self.samples_per_window

        # Support two input formats
        if isinstance(data_path, dict):
            self.ppg_file_path = data_path['ppg']
            self.index_file_path = data_path['index']
        else:
            # Check whether data_path is a directory
            if os.path.isdir(data_path):
                # Try CFS naming first
                ppg_path = os.path.join(data_path, 'cfs_ppg_with_labels.h5')
                if not os.path.exists(ppg_path):
                    # Fall back to MESA naming
                    ppg_path = os.path.join(data_path, 'mesa_ppg_with_labels.h5')
                self.ppg_file_path = ppg_path

                index_path = os.path.join(data_path, 'cfs_subject_index.h5')
                if not os.path.exists(index_path):
                    index_path = os.path.join(data_path, 'mesa_subject_index.h5')
                self.index_file_path = index_path
            else:
                raise ValueError(f"data_path should be a directory: {data_path}")

        if not os.path.exists(self.ppg_file_path):
            raise FileNotFoundError(f"PPG file not found: {self.ppg_file_path}")
        if not os.path.exists(self.index_file_path):
            raise FileNotFoundError(f"Index file not found: {self.index_file_path}")

        print("Loading CFS data from:")
        print(f"  PPG: {self.ppg_file_path}")
        print(f"  Index: {self.index_file_path}")

        self._prepare_subjects()

    def _prepare_subjects(self):
        """Prepare subject list using an 80/20 split"""
        with h5py.File(self.index_file_path, 'r') as f:
            all_subjects = list(f['subjects'].keys())
            valid_subjects = []
            for subj in all_subjects:
                n_windows = f[f'subjects/{subj}'].attrs['n_windows']
                if n_windows == self.windows_per_subject:
                    valid_subjects.append(subj)

        print(f"Total valid subjects in CFS: {len(valid_subjects)}")

        # 80/20 split
        train_val_subjects, test_subjects = train_test_split(
            valid_subjects, test_size=0.2, random_state=self.seed
        )
        train_subjects, val_subjects = train_test_split(
            train_val_subjects, test_size=0.2, random_state=self.seed
        )

        print("Dataset split:")
        print(f"  Train: {len(train_subjects)} ({len(train_subjects) / len(valid_subjects) * 100:.1f}%)")
        print(f"  Val:   {len(val_subjects)} ({len(val_subjects) / len(valid_subjects) * 100:.1f}%)")
        print(f"  Test:  {len(test_subjects)} ({len(test_subjects) / len(valid_subjects) * 100:.1f}%)")

        if self.split == 'train':
            self.subjects = train_subjects
        elif self.split == 'val':
            self.subjects = val_subjects
        else:
            self.subjects = test_subjects

        print(f"{self.split} set: {len(self.subjects)} subjects")

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

        ppg_continuous = ppg_windows.reshape(-1)

        if self.transform:
            ppg_continuous = self.transform(ppg_continuous)

        ppg_tensor = torch.FloatTensor(ppg_continuous).unsqueeze(0)
        labels_tensor = torch.LongTensor(labels)

        return ppg_tensor, labels_tensor


def get_cfs_dataloaders(data_path, batch_size=1, num_workers=0, seed=42):
    """Create CFS dataloaders"""
    train_dataset = CFSPPGOnlyDataset(data_path, split='train', seed=seed)
    val_dataset = CFSPPGOnlyDataset(data_path, split='val', seed=seed)
    test_dataset = CFSPPGOnlyDataset(data_path, split='test', seed=seed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def verify_cfs_data(data_path):
    """Verify CFS data format"""
    print("=" * 80)
    print("Verifying CFS data format")
    print("=" * 80)

    if isinstance(data_path, dict):
        ppg_file = data_path['ppg']
        index_file = data_path['index']
    else:
        # Auto-detect files
        ppg_file = os.path.join(data_path, 'cfs_ppg_with_labels.h5')
        if not os.path.exists(ppg_file):
            ppg_file = os.path.join(data_path, 'mesa_ppg_with_labels.h5')

        index_file = os.path.join(data_path, 'cfs_subject_index.h5')
        if not os.path.exists(index_file):
            index_file = os.path.join(data_path, 'mesa_subject_index.h5')

    print(f"\n📊 PPG file: {ppg_file}")
    with h5py.File(ppg_file, 'r') as f:
        print(f"  Keys: {list(f.keys())}")
        if 'ppg' in f:
            print(f"  PPG shape: {f['ppg'].shape}")
        if 'labels' in f:
            print(f"  Label shape: {f['labels'].shape}")
            labels = f['labels'][:]
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            print("\n  Label distribution:")
            stage_names = ['Wake', 'Light', 'Deep', 'REM']
            for u, c in zip(unique, counts):
                if u < len(stage_names):
                    pct = c / np.sum(counts) * 100
                    print(f"    {stage_names[int(u)]}: {c:,} ({pct:.2f}%)")

    print(f"\n📇 Index file: {index_file}")
    with h5py.File(index_file, 'r') as f:
        if 'subjects' in f:
            subjects = list(f['subjects'].keys())
            print(f"  Total subjects: {len(subjects)}")

    print("\n✅ Verification completed!")


if __name__ == "__main__":
    import sys

    data_path = sys.argv[1] if len(sys.argv) > 1 else "../../data"
    verify_cfs_data(data_path)

    print("\nTesting data loading...")
    train_loader, val_loader, test_loader, _, _, _ = get_cfs_dataloaders(data_path)
    print(f"✅ Success! Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
