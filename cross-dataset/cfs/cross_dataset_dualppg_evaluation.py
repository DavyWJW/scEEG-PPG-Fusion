#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-dataset evaluation script — Dual-stream PPG model

Test a PPG + Unfiltered PPG cross-attention model trained on MESA
using the CFS dataset to evaluate generalization.

Usage:
    python cross_dataset_ppg_evaluation.py \
        --model_path ./outputs/.../best_model.pth \
        --cfs_data_dir ../../data
"""

# NOTE:
# For brevity, this English version keeps the original logic unchanged.
# Only user-facing text (prints, comments, docstrings) has been translated.

