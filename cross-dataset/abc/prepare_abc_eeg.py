# -*- coding: utf-8 -*-
"""
ABC数据集单通道EEG数据预处理脚本
参考prepare_mesa.py，提取EEG信号并保存为npz格式

输出格式与MESA EEG保持一致：
- x: [epochs, 3000] - EEG信号，重采样到100Hz，每个epoch 30秒
- y: [epochs] - 睡眠分期标签 (0:W, 1:N1, 2:N2, 3:N3, 4:REM)
"""

import os
import glob
import numpy as np
from scipy.signal import resample
import mne
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    # ABC数据目录
    'data_dir': r"H:\sleepdata\abc\polysomnography\edfs\baseline",
    'annotations_dir': r"H:\sleepdata\abc\polysomnography\annotations-events-nsrr\baseline",

    # 输出目录
    'save_dir': r"H:\sleepdata\abc-eeg",

    # EEG差分通道配置（按优先级排序）
    # 格式: (信号电极, 参考电极) 或 (单通道名,) 如果已经是差分信号
    'eeg_bipolar_priority': [
        ('C3', 'M2'),  # 首选：左中央-右乳突
        ('C4', 'M1'),  # 备选：右中央-左乳突
        ('F3', 'M2'),  # 备选：左额叶-右乳突
        ('F4', 'M1'),  # 备选：右额叶-左乳突
    ],

    # 如果数据集已经有差分通道，按此优先级查找
    'eeg_single_priority': ['C3-M2', 'C4-M1', 'EEG', 'EEG1', 'EEG2'],

    # 滤波参数
    'lowcut': 0.3,
    'highcut': 35,

    # 重采样目标
    'target_samples_per_epoch': 3000,  # 100Hz * 30s = 3000
    'epoch_length_sec': 30,
}

# 睡眠分期映射 (与PPG保持一致，4类)
# 0: Wake, 1: Light (N1+N2), 2: Deep (N3), 3: REM
STAGE_MAPPING = {
    'Wake|0': 0,
    'Stage 1 sleep|1': 1,  # N1 -> Light
    'Stage 2 sleep|2': 1,  # N2 -> Light
    'Stage 3 sleep|3': 2,  # N3 -> Deep
    'Stage 4 sleep|4': 2,  # N4 -> Deep (如果有)
    'REM sleep|5': 3,  # REM
    'Movement|6': -1,  # 无效
    'Unscored': -1,  # 无效
}


def find_eeg_channel(raw, priority_list):
    """
    根据优先级列表查找EEG通道（仅用于已有差分通道的情况）
    """
    channel_names = raw.info['ch_names']
    channel_names_upper = [ch.upper() for ch in channel_names]

    for priority_ch in priority_list:
        for idx, ch in enumerate(channel_names_upper):
            if priority_ch.upper() == ch or priority_ch.upper() in ch:
                return channel_names[idx]

    return None


def find_bipolar_channels(raw, bipolar_priority):
    """
    查找双极导联的两个电极

    返回:
        (signal_ch, ref_ch): 信号电极和参考电极的通道名
        如果找不到返回 (None, None)
    """
    channel_names = raw.info['ch_names']
    channel_names_upper = [ch.upper() for ch in channel_names]

    for signal_name, ref_name in bipolar_priority:
        signal_ch = None
        ref_ch = None

        # 查找信号电极
        for idx, ch in enumerate(channel_names_upper):
            if signal_name.upper() == ch:
                signal_ch = channel_names[idx]
                break

        # 查找参考电极
        for idx, ch in enumerate(channel_names_upper):
            if ref_name.upper() == ch:
                ref_ch = channel_names[idx]
                break

        if signal_ch and ref_ch:
            return signal_ch, ref_ch

    return None, None


def parse_nsrr_xml_stages(xml_path):
    """
    解析NSRR格式的XML文件，提取睡眠分期

    返回按epoch排列的标签列表
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取epoch长度
    epoch_length_elem = root.find('EpochLength')
    epoch_length = int(epoch_length_elem.text) if epoch_length_elem is not None else 30

    # 收集所有睡眠分期事件
    stage_events = []

    for event in root.findall('.//ScoredEvent'):
        event_type = event.find('EventType')
        event_concept = event.find('EventConcept')
        start = event.find('Start')
        duration = event.find('Duration')

        # 检查是否有效
        if event_type is None or event_concept is None or start is None or duration is None:
            continue
        if event_type.text is None or event_concept.text is None:
            continue

        # 只处理睡眠分期事件
        if 'Stages' in event_type.text:
            stage_events.append({
                'concept': event_concept.text,
                'start': float(start.text),
                'duration': float(duration.text)
            })

    if not stage_events:
        return None

    # 找到最大时间确定总epoch数
    max_time = max(e['start'] + e['duration'] for e in stage_events)
    n_epochs = int(np.ceil(max_time / epoch_length))

    # 初始化标签数组（-1表示无效）
    labels = np.full(n_epochs, -1, dtype=np.int64)

    # 填充标签
    for event in stage_events:
        concept = event['concept']
        start_sec = event['start']
        duration_sec = event['duration']

        # 映射标签
        mapped_label = STAGE_MAPPING.get(concept, -1)

        # 计算影响的epoch范围
        start_epoch = int(start_sec // epoch_length)
        end_epoch = int((start_sec + duration_sec) // epoch_length)

        for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
            labels[epoch_idx] = mapped_label

    return labels


def load_one_subject_data(data_file_path, label_file_path, config):
    """
    加载单个被试的EEG数据和标签

    返回:
        data: [epochs, 1, samples] - EEG数据
        label: [epochs] - 标签列表
        channel_name: 使用的通道名
    """
    # 加载EDF文件
    raw = mne.io.read_raw_edf(data_file_path, preload=True, verbose=False)

    # 首先尝试查找已有的差分通道
    eeg_channel = find_eeg_channel(raw, config['eeg_single_priority'])

    if eeg_channel:
        # 找到已有的差分通道
        print(f"  使用已有EEG通道: {eeg_channel}")
        raw_picked = raw.copy().pick_channels([eeg_channel])
        channel_name = eeg_channel
    else:
        # 需要计算双极导联差分
        signal_ch, ref_ch = find_bipolar_channels(raw, config['eeg_bipolar_priority'])

        if signal_ch is None or ref_ch is None:
            raise ValueError(f"未找到EEG通道。可用通道: {raw.info['ch_names']}")

        print(f"  计算双极导联: {signal_ch} - {ref_ch}")
        channel_name = f"{signal_ch}-{ref_ch}"

        # 选择两个通道
        raw_picked = raw.copy().pick_channels([signal_ch, ref_ch])

        # 获取数据并计算差分
        data_array = raw_picked.get_data()  # [2, samples]

        # 找到通道索引
        picked_names = raw_picked.info['ch_names']
        signal_idx = picked_names.index(signal_ch)
        ref_idx = picked_names.index(ref_ch)

        # 计算差分信号
        bipolar_data = data_array[signal_idx] - data_array[ref_idx]  # [samples]

        # 创建新的Raw对象（单通道）
        info = mne.create_info([channel_name], raw.info['sfreq'], ch_types=['eeg'])
        raw_picked = mne.io.RawArray(bipolar_data.reshape(1, -1), info, verbose=False)

    # 滤波
    raw_picked.filter(config['lowcut'], config['highcut'],
                      fir_design='firwin', skip_by_annotation='edge', verbose=False)

    # 获取数据
    raw_data = raw_picked.get_data()
    sampling_rate = raw_picked.info['sfreq']

    print(f"  采样率: {sampling_rate} Hz")
    print(f"  数据形状: {raw_data.shape}")

    # 分割为epochs
    epoch_length_samples = int(config['epoch_length_sec'] * sampling_rate)
    epoch_num = int(raw_data.shape[1] // epoch_length_samples)

    data = []
    for epoch_id in range(epoch_num):
        start_sample = int(epoch_id * epoch_length_samples)
        end_sample = int((epoch_id + 1) * epoch_length_samples)
        data.append(raw_data[:, start_sample:end_sample])
    data = np.array(data)  # [epochs, 1, samples]

    print(f"  分割后epochs: {data.shape[0]}")

    # 加载标签
    labels = parse_nsrr_xml_stages(label_file_path)
    if labels is None:
        raise ValueError("无法解析睡眠分期标签")

    print(f"  标签数量: {len(labels)}")

    # 对齐数据和标签长度
    min_len = min(len(labels), data.shape[0])
    labels = labels[:min_len]
    data = data[:min_len, :, :]

    print(f"  对齐后长度: {min_len}")

    # 重采样到目标采样率 (100Hz -> 3000 samples per 30s epoch)
    data = resample(data, config['target_samples_per_epoch'], axis=2)

    print(f"  重采样后数据形状: {data.shape}")

    return data, labels, eeg_channel


def preprocess_abc_eeg():
    """
    预处理ABC数据集的EEG数据
    """
    config = CONFIG

    print("=" * 80)
    print("ABC数据集EEG预处理")
    print("=" * 80)

    # 创建输出目录
    save_dir = Path(config['save_dir'])
    eeg_save_dir = save_dir / 'eeg'
    eeg_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据目录: {config['data_dir']}")
    print(f"标注目录: {config['annotations_dir']}")
    print(f"输出目录: {eeg_save_dir}")
    print()

    # 查找所有EDF文件
    data_dir = Path(config['data_dir'])
    annotations_dir = Path(config['annotations_dir'])

    edf_files = list(data_dir.glob("*.edf"))
    edf_files.sort()

    print(f"找到 {len(edf_files)} 个EDF文件")
    print("-" * 80)

    # 已处理文件列表
    preprocessed_list = [f.stem for f in eeg_save_dir.glob("*.npz")]

    success_count = 0
    failed_count = 0
    failed_files = []

    for edf_file in tqdm(edf_files, desc="处理进度"):
        file_name = edf_file.stem  # abc-baseline-900001

        # 检查是否已处理
        if file_name in preprocessed_list:
            print(f"✓ {file_name} 已处理，跳过")
            success_count += 1
            continue

        print(f"\n处理: {file_name}")

        # 构建标注文件路径
        xml_file = annotations_dir / f"{file_name}-nsrr.xml"

        if not xml_file.exists():
            print(f"  ✗ 未找到标注文件: {xml_file.name}")
            failed_count += 1
            failed_files.append((file_name, "标注文件不存在"))
            continue

        try:
            # 加载数据
            data, labels, channel_name = load_one_subject_data(
                str(edf_file), str(xml_file), config
            )

            # labels已经是numpy数组

            # 统计标签分布
            unique, counts = np.unique(labels, return_counts=True)
            label_names = {-1: 'Invalid', 0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
            dist_str = ', '.join([f"{label_names.get(u, u)}:{c}" for u, c in zip(unique, counts)])
            print(f"  标签分布: {dist_str}")

            # 保存 (与MESA格式一致)
            # data[:, 0, :] 提取第一个通道，形状为 [epochs, samples]
            save_path = eeg_save_dir / f"{file_name}.npz"
            np.savez(save_path, x=data[:, 0, :], y=labels)

            print(f"  ✓ 保存成功: {save_path.name}")
            print(f"    数据形状: {data[:, 0, :].shape}, 标签形状: {labels.shape}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ 处理失败: {str(e)}")
            failed_count += 1
            failed_files.append((file_name, str(e)))

    # 打印统计
    print("\n" + "=" * 80)
    print("处理完成")
    print("=" * 80)
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")

    if failed_files:
        print("\n失败文件:")
        for name, error in failed_files[:10]:
            print(f"  - {name}: {error[:50]}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个")

    print(f"\n输出目录: {eeg_save_dir}")


def process_all_visits():
    """
    处理所有时间点 (baseline, 9-month, 18-month)
    """
    base_data_dir = r"H:\sleepdata\abc\polysomnography\edfs"
    base_annotations_dir = r"H:\sleepdata\abc\polysomnography\annotations-events-nsrr"
    base_save_dir = r"H:\sleepdata\abc-eeg"

    visits = ['baseline', '9-month', '18-month']

    for visit in visits:
        print("\n" + "=" * 80)
        print(f"处理 {visit}")
        print("=" * 80)

        data_dir = os.path.join(base_data_dir, visit)
        annotations_dir = os.path.join(base_annotations_dir, visit)

        if not os.path.exists(data_dir):
            print(f"目录不存在，跳过: {data_dir}")
            continue

        # 更新配置
        CONFIG['data_dir'] = data_dir
        CONFIG['annotations_dir'] = annotations_dir
        CONFIG['save_dir'] = os.path.join(base_save_dir, visit)

        preprocess_abc_eeg()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # 处理所有时间点
        process_all_visits()
    else:
        # 只处理baseline
        preprocess_abc_eeg()