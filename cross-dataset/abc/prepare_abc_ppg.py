"""
ABC数据集预处理工具
从NSRR下载的EDF/XML文件 → 标准化HDF5格式（与MESA兼容）

ABC数据集特点：
- 49名受试者，最多3个时间点（baseline, 9-month, 18-month）
- 包含PPG (Pleth通道)、EEG、ECG等信号
- XML标注格式与CFS类似（NSRR标准格式）
"""

import os
import sys
import numpy as np
import h5py
from pathlib import Path
import warnings
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET

warnings.filterwarnings('ignore')

try:
    import mne
    from scipy import signal
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install mne scipy")
    sys.exit(1)

print("=" * 80)
print("ABC数据集预处理工具")
print("EDF/XML → HDF5 (MESA兼容格式)")
print("=" * 80)

# ============================================================================
# 配置部分
# ============================================================================

CONFIG = {
    # ABC数据根目录
    'abc_root': "H:/sleepdata/abc",

    # 输出目录
    'output_dir': "./abc_processed",

    # 目标格式参数（与MESA保持一致）
    'target_fs': 34.13,  # SleepPPG-Net目标采样率
    'epoch_length_sec': 30,  # Epoch长度(秒)
    'samples_per_epoch': 1024,  # 每个epoch的采样点数 (34.13 * 30 ≈ 1024)
    'target_epochs': 1200,  # 目标epoch数 (10小时)
    'total_samples': 1228800,  # 总采样点数

    # 滤波参数
    'lowpass_cutoff': 8,  # 低通滤波截止频率 (Hz) - SleepPPG-Net使用8Hz
    'filter_order': 8,  # 滤波器阶数

    # 数据裁剪
    'clip_std': 3,  # 裁剪到±N个标准差

    # 处理选项
    'max_files_to_process': None,  # 限制处理数量 (None=全部)
    'verbose': True,  # 详细输出

    # 信号通道名称变体
    'ppg_channel_variants': ['Pleth', 'PLETH', 'PPG', 'SpO2', 'Pulse'],
}

# ============================================================================
# 睡眠分期映射
# ============================================================================

# ABC使用AASM标准，映射到4类
STAGE_MAPPING = {
    # Wake
    'Wake|0': 0,
    'Wake': 0,
    'W': 0,

    # Light Sleep (N1 + N2)
    'Stage 1 sleep|1': 1,
    'NREM1': 1,
    'N1': 1,
    'Stage 2 sleep|2': 1,
    'NREM2': 1,
    'N2': 1,

    # Deep Sleep (N3)
    'Stage 3 sleep|3': 2,
    'NREM3': 2,
    'N3': 2,
    'Stage 4 sleep|4': 2,  # 旧的R&K分期
    'NREM4': 2,

    # REM
    'REM sleep|5': 3,
    'REM': 3,
    'R': 3,

    # 忽略的标签
    'Movement|6': -1,
    'Movement': -1,
    'Unscored': -1,
    'Unknown': -1,
}


# ============================================================================
# 工具函数
# ============================================================================

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def format_time(seconds):
    """格式化时间"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


# ============================================================================
# 标注解析器
# ============================================================================

class ABCAnnotationParser:
    """解析ABC XML标注文件（NSRR格式）"""

    def __init__(self):
        self.label_mapping = STAGE_MAPPING

    def parse_nsrr_xml(self, xml_path):
        """
        解析NSRR格式的XML标注

        返回:
            events: 睡眠事件列表
            error: 错误信息（如果有）
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            events = []

            # 查找所有ScoredEvent
            for event in root.findall('.//ScoredEvent'):
                event_type = event.find('EventType')
                event_concept = event.find('EventConcept')
                start = event.find('Start')
                duration = event.find('Duration')

                # 检查所有必需元素是否存在且有文本内容
                if event_type is None or event_concept is None or start is None or duration is None:
                    continue

                # 检查文本内容是否存在
                if event_type.text is None or event_concept.text is None:
                    continue
                if start.text is None or duration.text is None:
                    continue

                # 检查是否是睡眠分期事件
                if 'Stages' in event_type.text:
                    events.append({
                        'concept': event_concept.text,
                        'start': float(start.text),
                        'duration': float(duration.text)
                    })

            if len(events) == 0:
                return None, f"未找到任何睡眠分期标注"

            return events, None

        except ET.ParseError as e:
            return None, f"XML解析错误: {str(e)}"
        except Exception as e:
            return None, f"解析失败: {str(e)}"

    def create_epoch_labels(self, events, total_duration_sec, epoch_length=30):
        """
        从事件创建epoch级别的标签数组

        返回:
            labels: 标签数组 [n_epochs]
        """
        n_epochs = int(total_duration_sec // epoch_length)
        labels = np.full(n_epochs, -1, dtype=np.int8)

        for event in events:
            concept = event['concept']
            start_sec = event['start']
            duration_sec = event['duration']

            # 映射标签
            label = self.label_mapping.get(concept, -1)

            # 计算影响的epoch范围
            start_epoch = int(start_sec // epoch_length)
            end_epoch = int((start_sec + duration_sec) // epoch_length)

            # 设置标签
            for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
                labels[epoch_idx] = label

        return labels


# ============================================================================
# PPG预处理器
# ============================================================================

class ABCPPGPreprocessor:
    """ABC PPG信号预处理（遵循SleepPPG-Net方法）"""

    def __init__(self, config):
        self.config = config

    def find_channel(self, channel_names, variants):
        """查找信号通道"""
        for variant in variants:
            for ch in channel_names:
                if variant.upper() in ch.upper():
                    return ch
        return None

    def load_signal(self, edf_path, channel_variants):
        """加载指定通道的信号"""
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

            # 查找通道
            channel = self.find_channel(raw.ch_names, channel_variants)

            if channel is None:
                return None, None, f"未找到通道: {channel_variants}"

            # 加载数据
            raw.pick_channels([channel])
            raw.load_data()

            data = raw.get_data()[0]
            fs = raw.info['sfreq']

            raw.close()

            return data, fs, None

        except Exception as e:
            return None, None, f"加载失败: {str(e)}"

    def preprocess_ppg(self, ppg_signal, original_fs):
        """
        预处理PPG信号（按照SleepPPG-Net方法）

        步骤:
            1. 低通滤波（8Hz）
            2. 下采样到34.13Hz
            3. Clip到±3σ
            4. Z-score标准化
            5. 填充/截断到10小时

        返回:
            processed_ppg: 处理后的PPG [1,228,800]
        """
        target_fs = self.config['target_fs']

        # 步骤1: 低通滤波
        nyq = 0.5 * original_fs
        cutoff = self.config['lowpass_cutoff'] / nyq

        # 确保cutoff < 1
        if cutoff >= 1:
            cutoff = 0.99

        sos = signal.cheby2(
            self.config['filter_order'],
            40,
            cutoff,
            btype='lowpass',
            output='sos'
        )
        filtered_ppg = signal.sosfiltfilt(sos, ppg_signal)

        # 步骤2: 下采样到34.13Hz
        downsample_factor = original_fs / target_fs
        n_samples_new = int(len(filtered_ppg) / downsample_factor)
        original_indices = np.arange(len(filtered_ppg))
        new_indices = np.linspace(0, len(filtered_ppg) - 1, n_samples_new)
        downsampled_ppg = np.interp(new_indices, original_indices, filtered_ppg)

        # 步骤3: Clip到±3σ
        mean = np.mean(downsampled_ppg)
        std = np.std(downsampled_ppg)
        clipped_ppg = np.clip(
            downsampled_ppg,
            mean - self.config['clip_std'] * std,
            mean + self.config['clip_std'] * std
        )

        # 步骤4: Z-score标准化
        standardized_ppg = (clipped_ppg - mean) / (std + 1e-8)

        # 步骤5: 填充/截断到10小时
        target_samples = self.config['total_samples']
        if len(standardized_ppg) < target_samples:
            pad_length = target_samples - len(standardized_ppg)
            processed_ppg = np.pad(
                standardized_ppg,
                (0, pad_length),
                mode='constant',
                constant_values=0
            )
        else:
            processed_ppg = standardized_ppg[:target_samples]

        return processed_ppg.astype(np.float32)

    def segment_into_windows(self, ppg_signal):
        """
        将连续信号分割成windows

        返回:
            windows: [1200, 1024]
        """
        samples_per_window = self.config['samples_per_epoch']
        n_windows = self.config['target_epochs']

        # 确保信号长度正确
        expected_length = n_windows * samples_per_window
        if len(ppg_signal) != expected_length:
            # 调整长度
            if len(ppg_signal) < expected_length:
                ppg_signal = np.pad(
                    ppg_signal,
                    (0, expected_length - len(ppg_signal)),
                    mode='constant'
                )
            else:
                ppg_signal = ppg_signal[:expected_length]

        # 重塑为windows
        windows = ppg_signal.reshape(n_windows, samples_per_window)

        return windows


# ============================================================================
# ABC数据集处理器
# ============================================================================

class ABCDatasetProcessor:
    """ABC数据集完整处理流程"""

    def __init__(self, config):
        self.config = config
        self.preprocessor = ABCPPGPreprocessor(config)
        self.parser = ABCAnnotationParser()
        self.results = []

    def find_edf_xml_pairs(self):
        """
        查找所有EDF和XML文件配对

        ABC数据集结构：
        H:\sleepdata\abc\polysomnography\
        ├── edfs\
        │   ├── baseline\
        │   │   └── abc-baseline-900001.edf
        │   ├── 9-month\
        │   └── 18-month\
        └── annotations-events-nsrr\
            ├── baseline\
            │   └── abc-baseline-900001-nsrr.xml
            ├── 9-month\
            └── 18-month\

        返回:
            pairs: [{subject_id, visit, edf_path, xml_path, full_id}, ...]
        """
        abc_root = Path(self.config['abc_root'])
        polysomnography_dir = abc_root / "polysomnography"

        pairs = []

        # 访问类型列表
        visits = ['baseline']

        for visit in visits:
            # EDF目录
            edf_dir = polysomnography_dir / "edfs" / visit
            # XML目录 - 每个visit有单独的子目录
            xml_dir = polysomnography_dir / "annotations-events-nsrr" / visit

            if not edf_dir.exists():
                if self.config['verbose']:
                    print(f"⚠️ EDF目录不存在: {edf_dir}")
                continue

            if not xml_dir.exists():
                if self.config['verbose']:
                    print(f"⚠️ XML目录不存在: {xml_dir}")
                continue

            # 遍历EDF文件
            for edf_file in edf_dir.glob("*.edf"):
                # 文件命名格式: abc-baseline-900001.edf
                filename = edf_file.stem  # abc-baseline-900001
                parts = filename.split('-')

                if len(parts) >= 3:
                    subject_id = parts[-1]  # 900001
                else:
                    subject_id = filename

                # 查找对应的XML文件: abc-baseline-900001-nsrr.xml
                xml_file = xml_dir / f"{filename}-nsrr.xml"

                if xml_file.exists():
                    pairs.append({
                        'subject_id': subject_id,
                        'visit': visit,
                        'edf_path': str(edf_file),
                        'xml_path': str(xml_file),
                        'full_id': f"{subject_id}_{visit}"
                    })
                else:
                    if self.config['verbose']:
                        print(f"⚠️ 未找到XML: {filename}-nsrr.xml")

        return pairs

    def process_single_file(self, pair_info):
        """处理单个文件"""
        subject_id = pair_info['full_id']
        edf_path = pair_info['edf_path']
        xml_path = pair_info['xml_path']

        try:
            # 1. 加载PPG信号
            ppg_signal, fs, error = self.preprocessor.load_signal(
                edf_path,
                self.config['ppg_channel_variants']
            )

            if error:
                return None, f"PPG加载失败: {error}"

            # 获取原始时长
            duration_sec = len(ppg_signal) / fs

            # 2. 解析标注
            events, error = self.parser.parse_nsrr_xml(xml_path)
            if error:
                return None, f"标注解析失败: {error}"

            # 3. 创建epoch标签
            epoch_labels = self.parser.create_epoch_labels(
                events,
                duration_sec,
                self.config['epoch_length_sec']
            )

            # 4. 预处理PPG信号
            processed_ppg = self.preprocessor.preprocess_ppg(ppg_signal, fs)

            # 5. 分割成windows
            ppg_windows = self.preprocessor.segment_into_windows(processed_ppg)

            # 6. 处理标签
            target_epochs = self.config['target_epochs']
            if len(epoch_labels) < target_epochs:
                # 填充为-1
                final_labels = np.full(target_epochs, -1, dtype=np.int64)
                final_labels[:len(epoch_labels)] = epoch_labels
            else:
                final_labels = epoch_labels[:target_epochs].astype(np.int64)

            # 7. 统计
            valid_mask = final_labels >= 0
            n_valid = np.sum(valid_mask)

            label_counts = np.zeros(4, dtype=np.int64)
            for i in range(4):
                label_counts[i] = np.sum(final_labels == i)

            # 8. 创建结果（只保留PPG相关数据）
            result = {
                'subject_id': subject_id,
                'ppg': ppg_windows,  # [1200, 1024]
                'labels': final_labels,  # [1200]
                'fs': self.config['target_fs'],
                'n_valid_epochs': int(n_valid),
                'label_distribution': label_counts.tolist(),
                'visit': pair_info['visit']
            }

            return result, None

        except Exception as e:
            import traceback
            return None, f"处理失败: {str(e)}\n{traceback.format_exc()}"

    def save_hdf5(self, result, output_dir):
        """保存为HDF5格式（只保留PPG信号，与MESA兼容）"""
        output_path = Path(output_dir) / f"{result['subject_id']}.h5"

        with h5py.File(output_path, 'w') as f:
            # 只保存PPG数据
            f.create_dataset(
                'ppg',
                data=result['ppg'],
                compression='gzip',
                compression_opts=4
            )

            # 保存标签
            f.create_dataset('labels', data=result['labels'])

            # 保存元数据
            f.attrs['subject_id'] = result['subject_id']
            f.attrs['fs'] = result['fs']
            f.attrs['n_valid_epochs'] = result['n_valid_epochs']
            f.attrs['visit'] = result['visit']

        return output_path

    def run(self):
        """运行完整处理流程"""
        print_section("ABC数据集预处理流程")

        # 创建输出目录
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)

        # 1. 查找文件配对
        print("📁 查找EDF/XML文件配对...")
        pairs = self.find_edf_xml_pairs()

        if not pairs:
            print("❌ 未找到任何文件配对!")
            print(f"请检查目录: {self.config['abc_root']}")
            return

        print(f"✅ 找到 {len(pairs)} 对文件")

        # 显示示例
        if pairs:
            print("\n示例:")
            for p in pairs[:3]:
                print(f"  - {p['full_id']}: {Path(p['edf_path']).name}")

        # 限制处理数量
        if self.config['max_files_to_process']:
            pairs = pairs[:self.config['max_files_to_process']]
            print(f"\n⚠️ 测试模式: 只处理前 {len(pairs)} 个文件")

        # 2. 处理
        print_section("开始预处理")
        print(f"📊 待处理文件: {len(pairs)}")
        print(f"📁 输出目录: {output_dir}")

        import time
        start_time = time.time()

        success_count = 0
        failed_count = 0

        for pair_info in tqdm(pairs, desc="预处理进度"):
            result, error = self.process_single_file(pair_info)

            if result is not None:
                # 保存HDF5
                h5_path = self.save_hdf5(result, output_dir)

                self.results.append({
                    'subject_id': result['subject_id'],
                    'success': True,
                    'h5_path': str(h5_path),
                    'n_valid_epochs': result['n_valid_epochs'],
                    'label_distribution': result['label_distribution'],
                    'visit': result['visit']
                })

                success_count += 1
            else:
                self.results.append({
                    'subject_id': pair_info['full_id'],
                    'success': False,
                    'error': error
                })

                failed_count += 1

                if self.config['verbose']:
                    print(f"\n❌ {pair_info['full_id']}: {error[:100]}")

        elapsed = time.time() - start_time

        # 3. 生成报告
        self.generate_report(elapsed, success_count, failed_count)

        # 4. 保存结果文件
        self.save_results(output_dir)

    def generate_report(self, elapsed_time, success_count, failed_count):
        """生成报告"""
        print_section("预处理完成")

        print(f"⏱️ 总耗时: {format_time(elapsed_time)}")
        print(f"⚡ 平均速度: {elapsed_time / (success_count + failed_count):.2f} 秒/文件")
        print()

        print(f"📊 结果统计:")
        print(f"   成功: {success_count}")
        print(f"   失败: {failed_count}")
        print(f"   成功率: {success_count / (success_count + failed_count) * 100:.1f}%")

        # 成功文件的统计
        if success_count > 0:
            success_results = [r for r in self.results if r['success']]

            total_epochs = sum(r['n_valid_epochs'] for r in success_results)

            # 标签分布统计
            total_label_counts = np.zeros(4, dtype=np.int64)
            for r in success_results:
                total_label_counts += np.array(r['label_distribution'])

            print(f"\n📈 数据统计:")
            print(f"   总有效epoch数: {total_epochs:,}")

            # 按visit统计
            visits = {}
            for r in success_results:
                v = r['visit']
                if v not in visits:
                    visits[v] = 0
                visits[v] += 1

            print(f"\n📅 访问分布:")
            for v, count in sorted(visits.items()):
                print(f"   {v}: {count} 文件")

            print(f"\n📊 标签分布:")
            label_names = ['Wake', 'Light', 'Deep', 'REM']
            for i, (name, count) in enumerate(zip(label_names, total_label_counts)):
                percentage = count / total_epochs * 100 if total_epochs > 0 else 0
                print(f"   {name:10s}: {count:8,} ({percentage:5.2f}%)")

    def save_results(self, output_dir):
        """保存结果文件"""
        print_section("保存结果文件")

        output_dir = Path(output_dir)
        success_results = [r for r in self.results if r['success']]

        if success_results:
            # 1. 被试ID列表
            ids_file = output_dir / "processed_subject_ids.txt"
            with open(ids_file, 'w') as f:
                for r in success_results:
                    f.write(r['subject_id'] + '\n')
            print(f"✅ 被试ID列表: {ids_file}")

            # 2. HDF5文件列表
            h5_list_file = output_dir / "h5_file_list.txt"
            with open(h5_list_file, 'w') as f:
                for r in success_results:
                    f.write(r['h5_path'] + '\n')
            print(f"✅ HDF5文件列表: {h5_list_file}")

            # 3. 详细统计CSV
            import csv
            stats_file = output_dir / "processing_statistics.csv"
            with open(stats_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'subject_id', 'visit', 'n_valid_epochs',
                    'wake', 'light', 'deep', 'rem'
                ])
                writer.writeheader()
                for r in success_results:
                    writer.writerow({
                        'subject_id': r['subject_id'],
                        'visit': r['visit'],
                        'n_valid_epochs': r['n_valid_epochs'],
                        'wake': r['label_distribution'][0],
                        'light': r['label_distribution'][1],
                        'deep': r['label_distribution'][2],
                        'rem': r['label_distribution'][3],
                    })
            print(f"✅ 详细统计: {stats_file}")

        print()
        print_section("全部完成")
        print(f"📁 所有HDF5文件保存在: {output_dir}")
        print(f"📊 成功处理: {len(success_results)} 个文件")
        print(f"🎯 可直接用于fine-tuning!")


# ============================================================================
# 合并HDF5文件（用于训练）
# ============================================================================

def merge_abc_to_single_hdf5(processed_dir, output_file):
    """
    将所有处理后的ABC HDF5文件合并为单个文件（与MESA格式兼容）
    只保留PPG信号和标签
    同时生成subject_index文件

    输出格式:
        abc_ppg_with_labels.h5:
            ppg: [total_windows, 1024]
            labels: [total_windows]

        abc_subject_index.h5:
            subjects/{subject_id}/window_indices: 索引数组
            subjects/{subject_id}.attrs: n_windows等
    """
    processed_dir = Path(processed_dir)
    h5_files = list(processed_dir.glob("*.h5"))

    # 排除已存在的合并文件和索引文件
    h5_files = [f for f in h5_files if 'abc_ppg_with_labels' not in f.name
                and 'abc_subject_index' not in f.name]

    print(f"📁 找到 {len(h5_files)} 个HDF5文件")

    all_ppg = []
    all_labels = []
    subject_info = []

    current_idx = 0

    for h5_file in tqdm(h5_files, desc="合并文件"):
        with h5py.File(h5_file, 'r') as f:
            ppg = f['ppg'][:]  # [1200, 1024]
            labels = f['labels'][:]  # [1200]

            # 只保留有效的epochs (labels >= 0)
            valid_mask = labels >= 0
            valid_ppg = ppg[valid_mask]
            valid_labels = labels[valid_mask]

            n_windows = len(valid_ppg)

            if n_windows == 0:
                print(f"⚠️ 跳过无有效数据: {h5_file.name}")
                continue

            all_ppg.append(valid_ppg)
            all_labels.append(valid_labels)

            subject_info.append({
                'subject_id': f.attrs['subject_id'],
                'start_idx': current_idx,
                'n_windows': n_windows,
                'visit': f.attrs.get('visit', 'unknown')
            })

            current_idx += n_windows

    # 合并
    all_ppg = np.concatenate(all_ppg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print(f"\n📊 合并结果:")
    print(f"   PPG形状: {all_ppg.shape}")
    print(f"   标签形状: {all_labels.shape}")
    print(f"   被试数: {len(subject_info)}")

    # 标签分布
    print(f"\n📊 标签分布:")
    label_names = ['Wake', 'Light', 'Deep', 'REM']
    for i in range(4):
        count = np.sum(all_labels == i)
        pct = count / len(all_labels) * 100
        print(f"   {label_names[i]}: {count:,} ({pct:.1f}%)")

    # 保存主数据文件
    print(f"\n💾 保存主数据文件...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('ppg', data=all_ppg, compression='gzip')
        f.create_dataset('labels', data=all_labels)
    print(f"✅ 保存到: {output_file}")

    # 保存subject索引文件（与MESA格式兼容）
    index_file = output_file.parent / "abc_subject_index.h5"
    print(f"\n💾 保存被试索引文件...")
    with h5py.File(index_file, 'w') as f:
        subjects_grp = f.create_group('subjects')
        for info in subject_info:
            subj_grp = subjects_grp.create_group(info['subject_id'])
            subj_grp.attrs['n_windows'] = info['n_windows']
            subj_grp.attrs['visit'] = info['visit']
            subj_grp.create_dataset(
                'window_indices',
                data=np.arange(info['start_idx'], info['start_idx'] + info['n_windows'])
            )
    print(f"✅ 保存到: {index_file}")

    # 打印被试统计
    print(f"\n📊 被试统计:")
    visits_count = {}
    for info in subject_info:
        v = info['visit']
        visits_count[v] = visits_count.get(v, 0) + 1
    for v, c in sorted(visits_count.items()):
        print(f"   {v}: {c} 被试")

    # 检查是否所有被试都有1200个windows（完整10小时）
    complete_subjects = sum(1 for info in subject_info if info['n_windows'] == 1200)
    print(f"\n   完整记录 (1200 windows): {complete_subjects}/{len(subject_info)}")

    return output_file, index_file


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 显示配置
    print("配置信息:")
    print(f"  ABC根目录: {CONFIG['abc_root']}")
    print(f"  输出目录: {CONFIG['output_dir']}")
    print(f"  目标采样率: {CONFIG['target_fs']} Hz")
    print(f"  Epoch长度: {CONFIG['epoch_length_sec']}s")
    print(f"  目标epochs: {CONFIG['target_epochs']}")

    if CONFIG['max_files_to_process']:
        print(f"  ⚠️ 测试模式: 只处理 {CONFIG['max_files_to_process']} 个文件")

    print("\n" + "=" * 80)
    print("确认开始处理? (y/n): ", end='')
    response = input().strip().lower()

    if response != 'y':
        print("\n已取消")
        return

    # 运行预处理
    processor = ABCDatasetProcessor(CONFIG)
    processor.run()

    # 询问是否合并
    print("\n是否合并为单个HDF5文件? (y/n): ", end='')
    response = input().strip().lower()

    if response == 'y':
        output_file = Path(CONFIG['output_dir']) / "abc_ppg_with_labels.h5"
        merge_abc_to_single_hdf5(CONFIG['output_dir'], output_file)

    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)