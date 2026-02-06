# Combining scEEG and PPG for Reliable Sleep Staging Using Lightweight Wearables

This repository contains the official implementation for ours paper:

> **Combining scEEG and PPG for reliable sleep staging using lightweight wearables**

## Overview
We investigate the fusion of single-channel EEG (scEEG) and photoplethysmography (PPG) for 4-class sleep staging (Wake, Light Sleep, Deep Sleep, REM) under short-window (30s–30min) constraints, with a particular focus on improving Light Sleep classification—a historically challenging stage due to its transitional nature and subtle physiological signatures. Three fusion strategies are proposed and compared:

- **Score-level fusion**: Weighted probability combination of independently trained scEEG and PPG models
- **Cross-attention fusion**: Bidirectional feature-level interactions between modalities via multi-head cross-attention
- **Mamba-enhanced fusion**: Cross-attention fusion augmented with bidirectional Mamba temporal context modeling

The Mamba-enhanced fusion achieves the best performance on the MESA dataset (Cohen's κ = 0.798, Accuracy = 86.9%), with notably improved light sleep classification (F1: 85.63% vs. 77.76% compared to scEEG alone).

## Repository Structure

```
scEEG/
│
├── model/                                    # Model architectures
│   ├── short_window_eeg_model.py             # scEEG AttnSleep model
│   ├── ppg_unfiltered_crossattn.py           # Dual-stream PPG model
│   ├── cross_attention_fusion.py             # Cross-attention fusion model
│   ├── cross_attention_mamba_fusion.py       # Mamba-enhanced fusion model
│   ├── short_window_eeg_dataset.py           # EEG dataset loader
│   ├── multimodal_dataset_aligned.py         # Multi-modal dataset loader
│   ├── multimodal_sleep_model.py             # SleepPPG-Net baseline
│   └── model.py                              # Base model components
│
├── train/                                    # Training & Evaluation on MESA
│   ├── train_evaluate_single_eeg_window.py   # Train scEEG model
│   ├── ppg_crossattn_shortwindow.py          # Train dual-stream PPG model
│   ├── train_cross_attention_fusion.py       # Train cross-attention fusion
│   ├── train_mamba_fusion_mesa.py            # Train Mamba-enhanced fusion
│   ├── train_score_fusion.py                 # Score-level fusion training
│   └── evaluate_cross_attention_fusion.py    # Evaluate cross-attention fusion
│
├── cross-dataset/                            # Cross-dataset validation
│   ├── cfs/                                  # CFS dataset (719 subjects)
│   │   ├── cfs_dataset.py                    # CFS data loader
│   │   ├── eeg_data_processing.py            # CFS EEG preprocessing (C3-M2)
│   │   ├── cross_dataset_eeg_evaluation.py   # Zero-shot scEEG evaluation
│   │   ├── cross_dataset_dualppg_evalidation.py  # Zero-shot PPG evaluation
│   │   ├── finetune_eeg_on_cfs.py            # Fine-tune scEEG on CFS
│   │   ├── finetune_dual_ppg_on_cfs.py       # Fine-tune PPG on CFS
│   │   ├── finetune_cross_atten.py           # Fine-tune cross-attention fusion
│   │   ├── finetune_mamba_fusion.py          # Fine-tune Mamba fusion
│   │   └── cfs_score_fusion_short_window.py  # CFS score-level fusion
│   │
│   └── abc/                                  # ABC dataset (49 subjects)
│       ├── prepare_abc_eeg.py                # ABC EEG preprocessing
│       ├── prepare_abc_ppg.py                # ABC PPG preprocessing
│       ├── eval_eeg__zeroshot_abc.py         # Zero-shot scEEG evaluation
│       ├── eval_dual_ppg_zeroshot_abc.py     # Zero-shot PPG evaluation
│       ├── finetune_eeg_on_abc.py            # Fine-tune scEEG on ABC
│       ├── finetune_dual_ppg_on_abc.py       # Fine-tune PPG on ABC
│       ├── finetune_cross_atten.py           # Fine-tune cross-attention fusion
│       ├── finetune_mamba_fusion.py          # Fine-tune Mamba fusion
│       └── abc_score_fusion_short_window.py  # ABC score-level fusion
│
└── README.md
```

## Datasets

We use three publicly available datasets from the [National Sleep Research Resource (NSRR)](https://sleepdata.org/):

| Dataset | Subjects | Age (mean) | Description |
|---------|----------|------------|-------------|
| [MESA](https://sleepdata.org/datasets/mesa) | 2,056 | 69.4 yrs | Training & primary evaluation |
| [CFS](https://sleepdata.org/datasets/cfs) | 719 | 41.2 yrs | Cross-dataset validation |
| [ABC](https://sleepdata.org/datasets/abc) | 49 | 48.2 yrs | Cross-dataset validation (clinical) |

> **Note**: Raw data must be downloaded from NSRR with an approved data access agreement.

### Data Preprocessing

**scEEG**:
- Bandpass filter: 0.3–35 Hz
- Resample to 100 Hz → 3,000 samples/epoch (30s)
- Derivation: C4-M1 (MESA), C3-M2 (CFS, ABC)

**PPG**:
- Lowpass filter: 8 Hz (8th-order Chebyshev Type II)
- Resample to 34.13 Hz → 1,024 samples/epoch (30s)
- Clip values beyond ±3 SD
- Recording-level z-score normalization

## Requirements

```
torch>=2.0.0
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
h5py>=3.0.0
mne>=1.0.0
tqdm>=4.60.0
pandas>=1.3.0
mamba-ssm>=1.0.0  # Optional, for Mamba-enhanced fusion
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train Individual Models on MESA

```bash
# Train scEEG AttnSleep model
python train/train_evaluate_single_eeg_window.py --window_minutes 3

# Train dual-stream PPG model
python train/ppg_crossattn_shortwindow.py --window_minutes 3
```

### 2. Train Fusion Models on MESA

```bash
# Train cross-attention fusion
python train/train_cross_attention_fusion.py

# Train Mamba-enhanced fusion
python train/train_mamba_fusion_mesa.py
```

### 3. Evaluate Score-level Fusion

```bash
python train/train_score_fusion.py \
    --ppg_model path/to/ppg_model.pth \
    --eeg_model path/to/eeg_model.pth
```

### 4. Cross-dataset Validation

```bash
# Preprocess CFS data
python cross-dataset/cfs/eeg_data_processing.py

# Zero-shot evaluation on CFS
python cross-dataset/cfs/cross_dataset_eeg_evaluation.py

# Fine-tune on CFS
python cross-dataset/cfs/finetune_eeg_on_cfs.py
```

## Results

### MESA Dataset (3-minute window)

  <p align="center">
    <img src="figures/architecture.png" width="90%" alt="Comparison of fusion models and SOTA methods under the 3-minute window"/>
  </p>


### Cross-dataset Generalization

| Dataset | Method | Zero-shot κ | Fine-tuned κ |
|---------|--------|-------------|--------------|
| CFS | scEEG | 0.621 | 0.689 |
| CFS | PPG | 0.583 | 0.652 |
| CFS | Mamba fusion | 0.658 | 0.721 |
| ABC | scEEG | 0.589 | 0.667 |
| ABC | PPG | 0.561 | 0.634 |
| ABC | Mamba fusion | 0.623 | 0.698 |

## Citation

If you find this work useful, please cite:

```bibtex
@article{wang2025sceeg,
  title={Combining scEEG and PPG for reliable sleep staging using lightweight wearables},
  author={Wang, Jiawei and Xu, Liang and Zheng, Shuntian and Guan, Yu and Wang, Kaichen and Zhang, Ziqing and Chen, Chen and Yang, Laurence T. and Gu, Sai},
  journal={IEEE Transactions and Journals},
  year={2025}
}
```

## License

This project is licensed under the MIT License.

## Acknowledgments

- Data provided by the [National Sleep Research Resource (NSRR)](https://sleepdata.org/)
  
- EEG baseline from [Eldele et al., 2021](https://github.com/emadeldeen24/AttnSleep)
  ```bibtex
@article{wang2025sceeg,
  title={Combining scEEG and PPG for reliable sleep staging using lightweight wearables},
  author={Wang, Jiawei and Xu, Liang and Zheng, Shuntian and Guan, Yu and Wang, Kaichen and Zhang, Ziqing and Chen, Chen and Yang, Laurence T. and Gu, Sai},
  journal={IEEE Transactions and Journals},
  year={2025}
}
```


- PPG baseline from [Kotzen et al., 2023](https://github.com/eth-siplab/SleepPPG-Net)
```bibtex
@article{wang2025sceeg,
  title={Combining scEEG and PPG for reliable sleep staging using lightweight wearables},
  author={Wang, Jiawei and Xu, Liang and Zheng, Shuntian and Guan, Yu and Wang, Kaichen and Zhang, Ziqing and Chen, Chen and Yang, Laurence T. and Gu, Sai},
  journal={IEEE Transactions and Journals},
  year={2025}
}
```

