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
    print(f"❌ Missing dependency: {e}")
    print("Please run: pip install mne scipy")
    sys.exit(1)

print("=" * 80)
print("ABC Dataset Preprocessing Utility")
print("EDF/XML → HDF5 (MESA-compatible format)")
print("=" * 80)

# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    'abc_root': "H:/sleepdata/abc",

    'output_dir': "./abc_processed",

    'target_fs': 34.13,  # Target sampling rate used by SleepPPG-Net
    'epoch_length_sec': 30,  # Epoch length (seconds)
    'samples_per_epoch': 1024,  # Samples per epoch (34.13 * 30 ≈ 1024)
    'target_epochs': 1200,  # Target number of epochs (10 hours)
    'total_samples': 1228800,  # Total number of samples

    # Filtering parameters
    'lowpass_cutoff': 8,  # Low-pass cutoff frequency (Hz) — SleepPPG-Net uses 8 Hz
    'filter_order': 8,  # Filter order

    # Signal clipping
    'clip_std': 3,  # Clip to ±N standard deviations

    # Processing options
    'max_files_to_process': None,  # Limit number of files to process (None = all)
    'verbose': True,  # Verbose output

    # Variants of channel names
    'ppg_channel_variants': ['Pleth', 'PLETH', 'PPG', 'SpO2', 'Pulse'],
}

# ============================================================================
# Sleep stage mapping
# ============================================================================

# ABC uses the AASM standard; map to 4 classes

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
    'Stage 4 sleep|4': 2,  # old R&K stage
    'NREM4': 2,

    # REM
    'REM sleep|5': 3,
    'REM': 3,
    'R': 3,

    # Labels to ignore
    'Movement|6': -1,
    'Movement': -1,
    'Unscored': -1,
    'Unknown': -1,
}


# ============================================================================
# Utility functions
# ============================================================================

def print_section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def format_time(seconds):

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


# ============================================================================
# Annotation parser
# ============================================================================

class ABCAnnotationParser:
    """Parser for ABC XML annotation files (NSRR format)"""

    def __init__(self):
        self.label_mapping = STAGE_MAPPING

    def parse_nsrr_xml(self, xml_path):
        """
        Parse NSRR-format XML annotations

        Returns:
            events: list of sleep stage events
            error: error message (if any)
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            events = []

            # Find all ScoredEvent entries
            for event in root.findall('.//ScoredEvent'):
                event_type = event.find('EventType')
                event_concept = event.find('EventConcept')
                start = event.find('Start')
                duration = event.find('Duration')

                # Check required fields exist
                if event_type is None or event_concept is None or start is None or duration is None:
                    continue

                # Check text content exists
                if event_type.text is None or event_concept.text is None:
                    continue
                if start.text is None or duration.text is None:
                    continue

                # Only keep sleep stage events
                if 'Stages' in event_type.text:
                    events.append({
                        'concept': event_concept.text,
                        'start': float(start.text),
                        'duration': float(duration.text)
                    })

            if len(events) == 0:
                return None, "No sleep stage annotations found"

            return events, None

        except ET.ParseError as e:
            return None, f"XML parsing error: {str(e)}"
        except Exception as e:
            return None, f"Parsing failed: {str(e)}"

    def create_epoch_labels(self, events, total_duration_sec, epoch_length=30):
        """
        Generate epoch-level labels from sleep events

        Returns:
            labels: label array [n_epochs]
        """
        n_epochs = int(total_duration_sec // epoch_length)
        labels = np.full(n_epochs, -1, dtype=np.int8)

        for event in events:
            concept = event['concept']
            start_sec = event['start']
            duration_sec = event['duration']

            # Map label
            label = self.label_mapping.get(concept, -1)

            # Compute affected epoch range
            start_epoch = int(start_sec // epoch_length)
            end_epoch = int((start_sec + duration_sec) // epoch_length)

            # Assign labels
            for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
                labels[epoch_idx] = label

        return labels

# ============================================================================
# PPG preprocessor
# ============================================================================

class ABCPPGPreprocessor:
    """ABC PPG signal preprocessing (following the SleepPPG-Net pipeline)"""

    def __init__(self, config):
        self.config = config

    def find_channel(self, channel_names, variants):
        """Find the matching signal channel"""
        for variant in variants:
            for ch in channel_names:
                if variant.upper() in ch.upper():
                    return ch
        return None

    def load_signal(self, edf_path, channel_variants):
        """Load signal from the specified channel"""
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

            # Find channel
            channel = self.find_channel(raw.ch_names, channel_variants)

            if channel is None:
                return None, None, f"Channel not found: {channel_variants}"

            # Load data
            raw.pick_channels([channel])
            raw.load_data()

            data = raw.get_data()[0]
            fs = raw.info['sfreq']

            raw.close()

            return data, fs, None

        except Exception as e:
            return None, None, f"Signal loading failed: {str(e)}"

    def preprocess_ppg(self, ppg_signal, original_fs):
        """
        Preprocess PPG signal (SleepPPG-Net pipeline)

        Steps:
            1. Low-pass filtering (8 Hz)
            2. Downsample to 34.13 Hz
            3. Clip to ±3σ
            4. Z-score normalization
            5. Pad/truncate to 10 hours

        Returns:
            processed_ppg: processed PPG signal [1,228,800]
        """
        target_fs = self.config['target_fs']

        # Step 1: Low-pass filtering
        nyq = 0.5 * original_fs
        cutoff = self.config['lowpass_cutoff'] / nyq

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

        # Step 2: Downsample to 34.13 Hz
        downsample_factor = original_fs / target_fs
        n_samples_new = int(len(filtered_ppg) / downsample_factor)
        original_indices = np.arange(len(filtered_ppg))
        new_indices = np.linspace(0, len(filtered_ppg) - 1, n_samples_new)
        downsampled_ppg = np.interp(new_indices, original_indices, filtered_ppg)

        # Step 3: Clip to ±3σ
        mean = np.mean(downsampled_ppg)
        std = np.std(downsampled_ppg)
        clipped_ppg = np.clip(
            downsampled_ppg,
            mean - self.config['clip_std'] * std,
            mean + self.config['clip_std'] * std
        )

        # Step 4: Z-score normalization
        standardized_ppg = (clipped_ppg - mean) / (std + 1e-8)

        # Step 5: Pad/truncate to 10 hours
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
        Segment continuous signal into windows

        Returns:
            windows: [1200, 1024]
        """
        samples_per_window = self.config['samples_per_epoch']
        n_windows = self.config['target_epochs']

        # Ensure correct signal length
        expected_length = n_windows * samples_per_window
        if len(ppg_signal) != expected_length:
            if len(ppg_signal) < expected_length:
                ppg_signal = np.pad(
                    ppg_signal,
                    (0, expected_length - len(ppg_signal)),
                    mode='constant'
                )
            else:
                ppg_signal = ppg_signal[:expected_length]

        windows = ppg_signal.reshape(n_windows, samples_per_window)

        return windows

# ============================================================================
# ABC dataset processor
# ============================================================================

class ABCDatasetProcessor:

    def __init__(self, config):
        self.config = config
        self.preprocessor = ABCPPGPreprocessor(config)
        self.parser = ABCAnnotationParser()
        self.results = []

    def find_edf_xml_pairs(self):
        abc_root = Path(self.config['abc_root'])
        polysomnography_dir = abc_root / "polysomnography"

        pairs = []

        visits = ['baseline']

        for visit in visits:
            edf_dir = polysomnography_dir / "edfs" / visit
            xml_dir = polysomnography_dir / "annotations-events-nsrr" / visit

            if not edf_dir.exists():
                if self.config['verbose']:
                    print(f"⚠️ EDF directory not found: {edf_dir}")
                continue

            if not xml_dir.exists():
                if self.config['verbose']:
                    print(f"⚠️ XML directory not found: {xml_dir}")
                continue

            # Iterate over EDF files
            for edf_file in edf_dir.glob("*.edf"):
                # Filename format: abc-baseline-900001.edf
                filename = edf_file.stem  # abc-baseline-900001
                parts = filename.split('-')

                if len(parts) >= 3:
                    subject_id = parts[-1]  # 900001
                else:
                    subject_id = filename

                # Corresponding XML file: abc-baseline-900001-nsrr.xml
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
                        print(f"⚠️ XML not found: {filename}-nsrr.xml")

        return pairs

    def process_single_file(self, pair_info):
        """Process a single EDF/XML pair"""
        subject_id = pair_info['full_id']
        edf_path = pair_info['edf_path']
        xml_path = pair_info['xml_path']

        try:
            # 1. Load PPG signal
            ppg_signal, fs, error = self.preprocessor.load_signal(
                edf_path,
                self.config['ppg_channel_variants']
            )

            if error:
                return None, f"PPG loading failed: {error}"

            duration_sec = len(ppg_signal) / fs

            # 2. Parse annotations
            events, error = self.parser.parse_nsrr_xml(xml_path)
            if error:
                return None, f"Annotation parsing failed: {error}"

            # 3. Create epoch labels
            epoch_labels = self.parser.create_epoch_labels(
                events,
                duration_sec,
                self.config['epoch_length_sec']
            )

            # 4. Preprocess PPG signal
            processed_ppg = self.preprocessor.preprocess_ppg(ppg_signal, fs)

            # 5. Segment into windows
            ppg_windows = self.preprocessor.segment_into_windows(processed_ppg)

            # 6. Process labels
            target_epochs = self.config['target_epochs']
            if len(epoch_labels) < target_epochs:
                final_labels = np.full(target_epochs, -1, dtype=np.int64)
                final_labels[:len(epoch_labels)] = epoch_labels
            else:
                final_labels = epoch_labels[:target_epochs].astype(np.int64)

            # 7. Statistics
            valid_mask = final_labels >= 0
            n_valid = np.sum(valid_mask)

            label_counts = np.zeros(4, dtype=np.int64)
            for i in range(4):
                label_counts[i] = np.sum(final_labels == i)

            # 8. Build result (PPG-only, MESA-compatible)
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
            return None, f"Processing failed: {str(e)}\n{traceback.format_exc()}"

    def save_hdf5(self, result, output_dir):
        """Save as HDF5 (PPG-only, MESA-compatible)"""
        output_path = Path(output_dir) / f"{result['subject_id']}.h5"

        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'ppg',
                data=result['ppg'],
                compression='gzip',
                compression_opts=4
            )

            f.create_dataset('labels', data=result['labels'])

            f.attrs['subject_id'] = result['subject_id']
            f.attrs['fs'] = result['fs']
            f.attrs['n_valid_epochs'] = result['n_valid_epochs']
            f.attrs['visit'] = result['visit']

        return output_path

    def run(self):
        """Run the full preprocessing pipeline"""
        print_section("ABC dataset preprocessing pipeline")

        output_dir = Path(self.config['output_dir'])
        output_dir.mkdir(exist_ok=True, parents=True)

        # 1. Find EDF/XML pairs
        print("📁 Searching for EDF/XML pairs...")
        pairs = self.find_edf_xml_pairs()

        if not pairs:
            print("❌ No EDF/XML pairs found!")
            print(f"Please check the directory: {self.config['abc_root']}")
            return

        print(f"✅ Found {len(pairs)} file pairs")

        # Show examples
        if pairs:
            print("\nExamples:")
            for p in pairs[:3]:
                print(f"  - {p['full_id']}: {Path(p['edf_path']).name}")

        # Limit processing
        if self.config['max_files_to_process']:
            pairs = pairs[:self.config['max_files_to_process']]
            print(f"\n⚠️ Test mode: only processing the first {len(pairs)} files")

        # 2. Process
        print_section("Start preprocessing")
        print(f"📊 Files to process: {len(pairs)}")
        print(f"📁 Output directory: {output_dir}")

        import time
        start_time = time.time()

        success_count = 0
        failed_count = 0

        for pair_info in tqdm(pairs, desc="Preprocessing"):
            result, error = self.process_single_file(pair_info)

            if result is not None:
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

        # 3. Report
        self.generate_report(elapsed, success_count, failed_count)

        # 4. Save outputs
        self.save_results(output_dir)

    def generate_report(self, elapsed_time, success_count, failed_count):
        """Generate a summary report"""
        print_section("Preprocessing completed")

        print(f"⏱️ Total time: {format_time(elapsed_time)}")
        print(f"⚡ Average speed: {elapsed_time / (success_count + failed_count):.2f} sec/file")
        print()

        print("📊 Summary:")
        print(f"   Successful: {success_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Success rate: {success_count / (success_count + failed_count) * 100:.1f}%")

        if success_count > 0:
            success_results = [r for r in self.results if r['success']]
            total_epochs = sum(r['n_valid_epochs'] for r in success_results)

            total_label_counts = np.zeros(4, dtype=np.int64)
            for r in success_results:
                total_label_counts += np.array(r['label_distribution'])

            print("\n📈 Data statistics:")
            print(f"   Total valid epochs: {total_epochs:,}")

            visits = {}
            for r in success_results:
                v = r['visit']
                visits[v] = visits.get(v, 0) + 1

            print("\n📅 Visit distribution:")
            for v, count in sorted(visits.items()):
                print(f"   {v}: {count} files")

            print("\n📊 Label distribution:")
            label_names = ['Wake', 'Light', 'Deep', 'REM']
            for name, count in zip(label_names, total_label_counts):
                percentage = count / total_epochs * 100 if total_epochs > 0 else 0
                print(f"   {name:10s}: {count:8,} ({percentage:5.2f}%)")

    def save_results(self, output_dir):
        """Save output files"""
        print_section("Saving outputs")

        output_dir = Path(output_dir)
        success_results = [r for r in self.results if r['success']]

        if success_results:
            ids_file = output_dir / "processed_subject_ids.txt"
            with open(ids_file, 'w') as f:
                for r in success_results:
                    f.write(r['subject_id'] + '\n')
            print(f"✅ Subject ID list: {ids_file}")

            h5_list_file = output_dir / "h5_file_list.txt"
            with open(h5_list_file, 'w') as f:
                for r in success_results:
                    f.write(r['h5_path'] + '\n')
            print(f"✅ HDF5 file list: {h5_list_file}")

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
            print(f"✅ Detailed statistics: {stats_file}")

        print()
        print_section("All done")
        print(f"📁 All HDF5 files are saved to: {output_dir}")
        print(f"📊 Successfully processed: {len(success_results)} files")
        print("🎯 Ready for fine-tuning!")


# ============================================================================
# Merge HDF5 files (for training)
# ============================================================================

def merge_abc_to_single_hdf5(processed_dir, output_file):

    processed_dir = Path(processed_dir)
    h5_files = list(processed_dir.glob("*.h5"))

    # Exclude previously merged files and index files
    h5_files = [f for f in h5_files if 'abc_ppg_with_labels' not in f.name
                and 'abc_subject_index' not in f.name]

    print(f"📁 Found {len(h5_files)} HDF5 files")

    all_ppg = []
    all_labels = []
    subject_info = []

    current_idx = 0

    for h5_file in tqdm(h5_files, desc="Merging files"):
        with h5py.File(h5_file, 'r') as f:
            ppg = f['ppg'][:]      # [1200, 1024]
            labels = f['labels'][:]  # [1200]

            # Keep only valid epochs
            valid_mask = labels >= 0
            valid_ppg = ppg[valid_mask]
            valid_labels = labels[valid_mask]

            n_windows = len(valid_ppg)

            if n_windows == 0:
                print(f"⚠️ Skipping file with no valid data: {h5_file.name}")
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

    all_ppg = np.concatenate(all_ppg, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    print("\n📊 Merge results:")
    print(f"   PPG shape: {all_ppg.shape}")
    print(f"   Label shape: {all_labels.shape}")
    print(f"   Subjects: {len(subject_info)}")

    print("\n📊 Label distribution:")
    label_names = ['Wake', 'Light', 'Deep', 'REM']
    for i in range(4):
        count = np.sum(all_labels == i)
        pct = count / len(all_labels) * 100
        print(f"   {label_names[i]}: {count:,} ({pct:.1f}%)")

    print("\n💾 Saving main dataset file...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('ppg', data=all_ppg, compression='gzip')
        f.create_dataset('labels', data=all_labels)
    print(f"✅ Saved to: {output_file}")

    index_file = output_file.parent / "abc_subject_index.h5"

    print("\n💾 Saving subject index file...")
    with h5py.File(index_file, 'w') as f:
        subjects_grp = f.create_group('subjects')

        for info in subject_info:
            subj_grp = subjects_grp.create_group(info['subject_id'])
            subj_grp.attrs['n_windows'] = info['n_windows']
            subj_grp.attrs['visit'] = info['visit']
            subj_grp.create_dataset(
                'window_indices',
                data=np.arange(info['start_idx'],
                               info['start_idx'] + info['n_windows'])
            )

    print(f"✅ Saved to: {index_file}")

    print("\n📊 Subject statistics:")
    visits_count = {}

    for info in subject_info:
        v = info['visit']
        visits_count[v] = visits_count.get(v, 0) + 1

    for v, c in sorted(visits_count.items()):
        print(f"   {v}: {c} subjects")

    complete_subjects = sum(
        1 for info in subject_info if info['n_windows'] == 1200
    )

    print(f"\n   Complete recordings (1200 windows): "
          f"{complete_subjects}/{len(subject_info)}")

    return output_file, index_file


# ============================================================================
# Main function
# ============================================================================

def main():
    """Main entry point"""

    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    print("Configuration:")
    print(f"  ABC root: {CONFIG['abc_root']}")
    print(f"  Output directory: {CONFIG['output_dir']}")
    print(f"  Target sampling rate: {CONFIG['target_fs']} Hz")
    print(f"  Epoch length: {CONFIG['epoch_length_sec']} s")
    print(f"  Target epochs: {CONFIG['target_epochs']}")

    if CONFIG['max_files_to_process']:
        print(f"  ⚠️ Test mode: processing only "
              f"{CONFIG['max_files_to_process']} files")

    print("\n" + "=" * 80)
    print("Proceed with processing? (y/n): ", end='')
    response = input().strip().lower()

    if response != 'y':
        print("\nCancelled")
        return

    processor = ABCDatasetProcessor(CONFIG)
    processor.run()

    print("\nMerge into a single HDF5 file? (y/n): ", end='')
    response = input().strip().lower()

    if response == 'y':
        output_file = Path(CONFIG['output_dir']) / "abc_ppg_with_labels.h5"
        merge_abc_to_single_hdf5(CONFIG['output_dir'], output_file)

    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        sys.exit(0)

    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
