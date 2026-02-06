
import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from tqdm import tqdm
from datetime import datetime
import xml.etree.ElementTree as ET
from scipy.signal import resample

warnings.filterwarnings('ignore')

# 检查依赖
try:
    import mne
except ImportError:
    print("❌ 缺少依赖: mne")
    print("请运行: pip install mne")
    sys.exit(1)

print("=" * 80)
print("CFS数据集单通道EEG提取工具")
print("配置: C3-M2 导联 (对齐MESA标准)")
print("=" * 80)


# ============================================================================
# 配置部分
# ============================================================================

CONFIG = {
    # 输入文件
    'passed_pairs_file': "./cfs_qc_results_simple/passed_pairs.txt",
    
    # 输出目录
    'output_dir': "./cfs_eeg_c3m2_data",
    
    # EEG通道配置
    'eeg_channel': 'C3',        # 主通道
    'reference_channel': 'M2',  # 参考电极
    'montage_name': 'C3-M2',    # 导联名称
    
    # MESA对齐的预处理参数
    'filter_lowcut': 0.3,       # 高通滤波 (Hz) - MESA标准
    'filter_highcut': 35,       # 低通滤波 (Hz) - MESA标准
    'epoch_length_sec': 30,     # Epoch长度(秒)
    'target_samples': 3000,     # 目标采样点数 (与MESA对齐)
    
    # 处理选项
    'max_files_to_process': None,
    'verbose': False,
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
# 标注解析器 (与CFS PPG保持一致)
# ============================================================================

class AnnotationParser:
    """解析CFS XML标注文件 - 与PPG提取保持一致"""
    
    def __init__(self):
        # CFS标注映射 - 与PPG提取完全相同
        self.label_mapping = {
            'Wake|0': 0,           # Wake
            'Stage 1 sleep|1': 1,  # NREM1
            'Stage 2 sleep|2': 2,  # NREM2
            'Stage 3 sleep|3': 3,  # NREM3 (Deep)
            'Stage 4 sleep|4': 3,  # NREM4 也算Deep
            'REM sleep|5': 4,      # REM
            'Movement|6': -1,      # 忽略
            'Unscored': -1,        # 忽略
        }
    
    def parse_nsrr_xml(self, xml_path):
        """解析NSRR格式的XML标注"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            events = []
            for event in root.findall('.//ScoredEvent'):
                event_type = event.find('EventType')
                event_concept = event.find('EventConcept')
                start = event.find('Start')
                duration = event.find('Duration')
                
                if all(x is not None for x in [event_type, event_concept, start, duration]):
                    if event_type.text == 'Stages|Stages':
                        concept = event_concept.text
                        events.append({
                            'concept': concept,
                            'start': float(start.text),
                            'duration': float(duration.text)
                        })
            
            return events, None
            
        except Exception as e:
            return None, f"解析失败: {str(e)}"
    
    def create_epoch_labels(self, events, total_duration_sec, epoch_length=30):
        """创建epoch级别的标签数组"""
        n_epochs = int(total_duration_sec // epoch_length)
        labels = np.full(n_epochs, -1, dtype=np.int8)
        
        for event in events:
            concept = event['concept']
            start_sec = event['start']
            duration_sec = event['duration']
            
            if concept in self.label_mapping:
                label = self.label_mapping[concept]
                
                start_epoch = int(start_sec // epoch_length)
                end_epoch = int((start_sec + duration_sec) // epoch_length)
                
                for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
                    labels[epoch_idx] = label
        
        return labels


# ============================================================================
# EEG提取器 (对齐MESA流程)
# ============================================================================

class EEGExtractor:
    """EEG信号提取和预处理 - 完全对齐MESA"""
    
    def __init__(self, config):
        self.config = config
    
    def load_eeg_bipolar(self, edf_path):
        """
        加载双极导联EEG: C3-M2
        对齐MESA的处理流程
        """
        try:
            # 加载EDF文件
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            
            # 检查通道是否存在
            if self.config['eeg_channel'] not in raw.ch_names:
                return None, None, f"未找到{self.config['eeg_channel']}通道"
            
            if self.config['reference_channel'] not in raw.ch_names:
                return None, None, f"未找到{self.config['reference_channel']}参考通道"
            
            # 提取C3和M2通道
            raw_picked = raw.copy().pick_channels([
                self.config['eeg_channel'],
                self.config['reference_channel']
            ])
            
            # ===== 对齐MESA: 带通滤波 0.3-35Hz =====
            raw_picked.filter(
                self.config['filter_lowcut'], 
                self.config['filter_highcut'],
                fir_design='firwin',
                skip_by_annotation='edge',
                verbose=False
            )
            
            # 获取数据
            data = raw_picked.get_data()
            c3_data = data[0]
            m2_data = data[1]
            
            # 计算双极导联: C3 - M2
            eeg_bipolar = c3_data - m2_data
            
            fs = raw.info['sfreq']
            
            raw.close()
            
            return eeg_bipolar, fs, None
            
        except Exception as e:
            return None, None, f"加载失败: {str(e)}"
    
    def segment_into_epochs(self, eeg_data, fs):
        """
        分割成30秒epochs
        对齐MESA的分割逻辑
        """
        epoch_length = self.config['epoch_length_sec']
        samples_per_epoch = int(epoch_length * fs)
        
        # 计算完整的epoch数量
        n_epochs = int(len(eeg_data) // samples_per_epoch)
        
        # 分割数据
        epochs = []
        for epoch_id in range(n_epochs):
            start_idx = int(epoch_id * samples_per_epoch)
            end_idx = int((epoch_id + 1) * samples_per_epoch)
            epoch_data = eeg_data[start_idx:end_idx]
            epochs.append(epoch_data)
        
        epochs = np.array(epochs)  # [n_epochs, samples_per_epoch]
        
        return epochs
    
    def resample_to_target(self, epochs):
        """
        重采样到目标采样点数 (3000点)
        对齐MESA的重采样逻辑
        """
        target_samples = self.config['target_samples']
        
        # 使用scipy.signal.resample (与MESA相同)
        resampled_epochs = resample(epochs, target_samples, axis=1)
        
        return resampled_epochs


# ============================================================================
# 完整处理流程
# ============================================================================

class CFSEEGProcessor:
    """CFS EEG处理主类"""
    
    def __init__(self, config):
        self.config = config
        self.results = []
    
    def load_passed_pairs(self, pairs_file):
        """加载passed_pairs.txt"""
        print_section("加载文件配对信息")
        
        if not Path(pairs_file).exists():
            print(f"❌ 错误: 文件不存在: {pairs_file}")
            sys.exit(1)
        
        df = pd.read_csv(pairs_file, sep='\t')
        
        print(f"✅ 加载 {len(df)} 对文件")
        return df
    
    def process_single_file(self, row, extractor, parser):
        """处理单个文件"""
        subject_id = row['subject_id']
        edf_file = row['edf_file']
        annotation_file = row['annotation_file']
        
        try:
            # 1. 加载EEG数据 (C3-M2)
            eeg_data, fs, error = extractor.load_eeg_bipolar(edf_file)
            if error:
                return None, error
            
            # 2. 分割成epochs
            eeg_epochs = extractor.segment_into_epochs(eeg_data, fs)
            
            # 3. 重采样到3000点 (对齐MESA)
            eeg_epochs = extractor.resample_to_target(eeg_epochs)
            
            # 4. 解析标注
            events, error = parser.parse_nsrr_xml(annotation_file)
            if error:
                return None, f"标注解析失败: {error}"
            
            # 5. 创建epoch标签
            duration_sec = len(eeg_data) / fs
            epoch_labels = parser.create_epoch_labels(
                events,
                duration_sec,
                self.config['epoch_length_sec']
            )
            
            # 6. 对齐数据和标签长度
            min_len = min(len(eeg_epochs), len(epoch_labels))
            eeg_epochs = eeg_epochs[:min_len]
            epoch_labels = epoch_labels[:min_len]
            
            assert len(eeg_epochs) == len(epoch_labels), "数据和标签长度不匹配"
            
            # 7. 过滤掉忽略的标签 (Movement和Unscored)
            valid_mask = epoch_labels >= 0
            eeg_epochs = eeg_epochs[valid_mask]
            epoch_labels = epoch_labels[valid_mask]
            
            # 8. 统计信息
            label_counts = np.bincount(epoch_labels, minlength=5)
            
            # 9. 创建结果字典
            result = {
                'subject_id': subject_id,
                'x': eeg_epochs.astype(np.float32),    # [n_epochs, 3000]
                'y': epoch_labels.astype(np.int64),    # [n_epochs]
                'channel': self.config['montage_name'],
                'original_fs': fs,
                'n_epochs': len(eeg_epochs),
                'n_epochs_removed': np.sum(~valid_mask),
                'label_distribution': label_counts.tolist()
            }
            
            return result, None
            
        except Exception as e:
            import traceback
            error_msg = f"处理失败: {str(e)}\n{traceback.format_exc()}"
            return None, error_msg
    
    def save_npz(self, result, output_dir):
        """保存为NPZ格式 (对齐MESA格式)"""
        output_path = Path(output_dir) / f"{result['subject_id']}.npz"
        
        # 与MESA EEG格式完全相同
        np.savez(
            output_path,
            x=result['x'],              # [n_epochs, 3000] - EEG信号
            y=result['y'],              # [n_epochs] - 标签 (0-4)
            # 可选: 添加额外信息
            # channel=result['channel'],
            # original_fs=result['original_fs']
        )
        
        return output_path
    
    def run(self):
        """运行完整流程"""
        print_section("CFS EEG数据提取流程")
        
        # 创建输出目录
        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 加载文件对
        df = self.load_passed_pairs(self.config['passed_pairs_file'])
        
        # 限制处理数量
        if self.config['max_files_to_process'] is not None:
            df = df.head(self.config['max_files_to_process'])
            print(f"\n⚠️  测试模式: 只处理前 {len(df)} 个文件")
        
        # 初始化
        extractor = EEGExtractor(self.config)
        parser = AnnotationParser()
        
        print_section("开始提取EEG数据")
        print(f"📊 待处理文件: {len(df)}")
        print(f"🧠 EEG导联: {self.config['montage_name']}")
        print(f"📁 输出目录: {output_dir}")
        print(f"🔧 滤波范围: {self.config['filter_lowcut']}-{self.config['filter_highcut']} Hz")
        print(f"📏 目标采样点: {self.config['target_samples']}")
        print()
        
        import time
        start_time = time.time()
        
        success_count = 0
        failed_count = 0
        
        # 逐个处理
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="EEG提取进度"):
            result, error = self.process_single_file(row, extractor, parser)
            
            if result is not None:
                # 保存NPZ
                npz_path = self.save_npz(result, output_dir)
                
                self.results.append({
                    'subject_id': result['subject_id'],
                    'success': True,
                    'npz_path': str(npz_path),
                    'n_epochs': result['n_epochs'],
                    'n_epochs_removed': result['n_epochs_removed'],
                    'label_distribution': result['label_distribution']
                })
                
                success_count += 1
            else:
                self.results.append({
                    'subject_id': row['subject_id'],
                    'success': False,
                    'error': error
                })
                
                failed_count += 1
                
                if self.config['verbose']:
                    print(f"\n❌ {row['subject_id']}: {error}")
        
        elapsed = time.time() - start_time
        
        # 统计和报告
        self.generate_report(elapsed, success_count, failed_count)
        
        # 保存结果
        self.save_results()
    
    def generate_report(self, elapsed_time, success_count, failed_count):
        """生成报告"""
        print_section("EEG提取完成")
        
        print(f"⏱️  总耗时: {format_time(elapsed_time)}")
        print(f"⚡ 平均速度: {elapsed_time/(success_count+failed_count):.2f} 秒/文件")
        print()
        
        print(f"📊 结果统计:")
        print(f"   成功: {success_count}")
        print(f"   失败: {failed_count}")
        print(f"   成功率: {success_count/(success_count+failed_count)*100:.1f}%")
        
        # 成功文件的统计
        if success_count > 0:
            success_results = [r for r in self.results if r['success']]
            
            total_epochs = sum(r['n_epochs'] for r in success_results)
            total_removed = sum(r['n_epochs_removed'] for r in success_results)
            
            # 标签分布统计
            total_label_counts = np.zeros(5, dtype=np.int64)
            for r in success_results:
                total_label_counts += np.array(r['label_distribution'])
            
            print(f"\n📈 数据统计:")
            print(f"   总epoch数: {total_epochs:,}")
            print(f"   移除epoch数: {total_removed:,} (Movement/Unscored)")
            print(f"   有效epoch数: {total_epochs:,}")
            
            print(f"\n📊 标签分布:")
            label_names = ['Wake', 'NREM1', 'NREM2', 'NREM3/Deep', 'REM']
            for i, (name, count) in enumerate(zip(label_names, total_label_counts)):
                percentage = count / total_epochs * 100 if total_epochs > 0 else 0
                print(f"   {name:15s}: {count:8,} ({percentage:5.2f}%)")
        
        # 失败原因
        if failed_count > 0:
            print(f"\n❌ 失败文件 (前5个):")
            failed_results = [r for r in self.results if not r['success']]
            for i, r in enumerate(failed_results[:5], 1):
                error_short = r['error'].split('\n')[0][:60]
                print(f"   {i}. {r['subject_id']}: {error_short}")
        
        print()
    
    def save_results(self):
        """保存结果"""
        print_section("保存结果")
        
        output_dir = Path(self.config['output_dir'])
        
        # 保存成功的文件列表
        success_results = [r for r in self.results if r['success']]
        
        if success_results:
            # 被试ID列表
            ids_file = output_dir / "processed_subject_ids.txt"
            with open(ids_file, 'w') as f:
                for r in success_results:
                    f.write(r['subject_id'] + '\n')
            print(f"✅ 被试ID列表: {ids_file}")
            
            # NPZ文件列表
            npz_list_file = output_dir / "npz_file_list.txt"
            with open(npz_list_file, 'w') as f:
                for r in success_results:
                    f.write(r['npz_path'] + '\n')
            print(f"✅ NPZ文件列表: {npz_list_file}")
            
            # 详细统计CSV
            stats_file = output_dir / "eeg_statistics.csv"
            stats_data = []
            for r in success_results:
                stats_data.append({
                    'subject_id': r['subject_id'],
                    'n_epochs': r['n_epochs'],
                    'n_epochs_removed': r['n_epochs_removed'],
                    'wake': r['label_distribution'][0],
                    'nrem1': r['label_distribution'][1],
                    'nrem2': r['label_distribution'][2],
                    'deep': r['label_distribution'][3],
                    'rem': r['label_distribution'][4],
                })
            pd.DataFrame(stats_data).to_csv(stats_file, index=False)
            print(f"✅ 详细统计: {stats_file}")
        
        # 保存完整报告
        summary_file = output_dir / "eeg_extraction_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("CFS数据集EEG提取报告\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"提取时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("EEG配置:\n")
            f.write(f"  导联: {self.config['montage_name']}\n")
            f.write(f"  滤波: {self.config['filter_lowcut']}-{self.config['filter_highcut']} Hz\n")
            f.write(f"  Epoch长度: {self.config['epoch_length_sec']} 秒\n")
            f.write(f"  目标采样点: {self.config['target_samples']}\n\n")
            
            f.write("处理结果:\n")
            f.write(f"  成功: {len(success_results)}\n")
            f.write(f"  失败: {len(self.results) - len(success_results)}\n\n")
            
            if success_results:
                total_epochs = sum(r['n_epochs'] for r in success_results)
                f.write(f"数据统计:\n")
                f.write(f"  总Epoch数: {total_epochs:,}\n")
                f.write(f"  数据形状: [{total_epochs}, 3000]\n")
                f.write(f"  标签范围: 0-4 (5类)\n\n")
                
                f.write("标签说明:\n")
                f.write("  0 - Wake (清醒)\n")
                f.write("  1 - NREM1 (浅睡眠1期)\n")
                f.write("  2 - NREM2 (浅睡眠2期)\n")
                f.write("  3 - NREM3/Deep (深睡眠)\n")
                f.write("  4 - REM (快速眼动期)\n\n")
                
                f.write("数据格式: 与MESA EEG完全兼容\n")
        
        print(f"✅ 提取摘要: {summary_file}")
        print()
        
        print_section("全部完成")
        print(f"📁 所有NPZ文件保存在: {output_dir}")
        print(f"📊 成功提取: {len(success_results)} 个文件")
        print(f"🧠 EEG导联: {self.config['montage_name']}")
        print(f"💾 数据格式: 与MESA EEG完全兼容")
        print(f"🎯 可直接用于训练!")
        print()


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 显示配置
    print("配置信息:")
    print(f"  输入文件: {CONFIG['passed_pairs_file']}")
    print(f"  输出目录: {CONFIG['output_dir']}")
    print(f"  EEG导联: {CONFIG['montage_name']}")
    print(f"  滤波范围: {CONFIG['filter_lowcut']}-{CONFIG['filter_highcut']} Hz (MESA标准)")
    print(f"  Epoch长度: {CONFIG['epoch_length_sec']}s")
    print(f"  目标采样点: {CONFIG['target_samples']} (与MESA对齐)")
    
    if CONFIG['max_files_to_process']:
        print(f"  ⚠️  测试模式: 只处理 {CONFIG['max_files_to_process']} 个文件")
    
    print("\n" + "=" * 80)
    print("确认开始处理? (y/n): ", end='')
    response = input().strip().lower()
    
    if response != 'y':
        print("\n已取消")
        return
    
    # 运行提取
    processor = CFSEEGProcessor(CONFIG)
    processor.run()
    
    print("=" * 80)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
