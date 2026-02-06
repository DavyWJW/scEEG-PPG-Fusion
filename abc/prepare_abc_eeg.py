
import os
import glob
import numpy as np
from scipy.signal import resample
import mne
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm



CONFIG = {

    'data_dir': r"H:\sleepdata\abc\polysomnography\edfs\baseline",
    'annotations_dir': r"H:\sleepdata\abc\polysomnography\annotations-events-nsrr\baseline",


    'save_dir': r"H:\sleepdata\abc-eeg",


    'eeg_bipolar_priority': [
        ('C3', 'M2'),
        ('C4', 'M1'),
        ('F3', 'M2'),
        ('F4', 'M1'),
    ],


    'eeg_single_priority': ['C3-M2', 'C4-M1', 'EEG', 'EEG1', 'EEG2'],


    'lowcut': 0.3,
    'highcut': 35,


    'target_samples_per_epoch': 3000,  # 100Hz * 30s = 3000
    'epoch_length_sec': 30,
}


STAGE_MAPPING = {
    'Wake|0': 0,
    'Stage 1 sleep|1': 1,  # N1 -> Light
    'Stage 2 sleep|2': 1,  # N2 -> Light
    'Stage 3 sleep|3': 2,  # N3 -> Deep
    'Stage 4 sleep|4': 2,  # N4 -> Deep
    'REM sleep|5': 3,  # REM
    'Movement|6': -1,
    'Unscored': -1,
}


def find_eeg_channel(raw, priority_list):

    channel_names = raw.info['ch_names']
    channel_names_upper = [ch.upper() for ch in channel_names]

    for priority_ch in priority_list:
        for idx, ch in enumerate(channel_names_upper):
            if priority_ch.upper() == ch or priority_ch.upper() in ch:
                return channel_names[idx]

    return None


def find_bipolar_channels(raw, bipolar_priority):

    channel_names = raw.info['ch_names']
    channel_names_upper = [ch.upper() for ch in channel_names]

    for signal_name, ref_name in bipolar_priority:
        signal_ch = None
        ref_ch = None


        for idx, ch in enumerate(channel_names_upper):
            if signal_name.upper() == ch:
                signal_ch = channel_names[idx]
                break


        for idx, ch in enumerate(channel_names_upper):
            if ref_name.upper() == ch:
                ref_ch = channel_names[idx]
                break

        if signal_ch and ref_ch:
            return signal_ch, ref_ch

    return None, None


def parse_nsrr_xml_stages(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()


    epoch_length_elem = root.find('EpochLength')
    epoch_length = int(epoch_length_elem.text) if epoch_length_elem is not None else 30


    stage_events = []

    for event in root.findall('.//ScoredEvent'):
        event_type = event.find('EventType')
        event_concept = event.find('EventConcept')
        start = event.find('Start')
        duration = event.find('Duration')


        if event_type is None or event_concept is None or start is None or duration is None:
            continue
        if event_type.text is None or event_concept.text is None:
            continue


        if 'Stages' in event_type.text:
            stage_events.append({
                'concept': event_concept.text,
                'start': float(start.text),
                'duration': float(duration.text)
            })

    if not stage_events:
        return None


    max_time = max(e['start'] + e['duration'] for e in stage_events)
    n_epochs = int(np.ceil(max_time / epoch_length))


    labels = np.full(n_epochs, -1, dtype=np.int64)


    for event in stage_events:
        concept = event['concept']
        start_sec = event['start']
        duration_sec = event['duration']


        mapped_label = STAGE_MAPPING.get(concept, -1)


        start_epoch = int(start_sec // epoch_length)
        end_epoch = int((start_sec + duration_sec) // epoch_length)

        for epoch_idx in range(start_epoch, min(end_epoch + 1, n_epochs)):
            labels[epoch_idx] = mapped_label

    return labels


def load_one_subject_data(data_file_path, label_file_path, config):

    raw = mne.io.read_raw_edf(data_file_path, preload=True, verbose=False)


    eeg_channel = find_eeg_channel(raw, config['eeg_single_priority'])

    if eeg_channel:

        print(f"  EEG channel: {eeg_channel}")
        raw_picked = raw.copy().pick_channels([eeg_channel])
        channel_name = eeg_channel
    else:

        signal_ch, ref_ch = find_bipolar_channels(raw, config['eeg_bipolar_priority'])

        if signal_ch is None or ref_ch is None:
            raise ValueError(f" {raw.info['ch_names']}")

        print(f"  {signal_ch} - {ref_ch}")
        channel_name = f"{signal_ch}-{ref_ch}"


        raw_picked = raw.copy().pick_channels([signal_ch, ref_ch])


        data_array = raw_picked.get_data()  # [2, samples]


        picked_names = raw_picked.info['ch_names']
        signal_idx = picked_names.index(signal_ch)
        ref_idx = picked_names.index(ref_ch)


        bipolar_data = data_array[signal_idx] - data_array[ref_idx]  # [samples]


        info = mne.create_info([channel_name], raw.info['sfreq'], ch_types=['eeg'])
        raw_picked = mne.io.RawArray(bipolar_data.reshape(1, -1), info, verbose=False)


    raw_picked.filter(config['lowcut'], config['highcut'],
                      fir_design='firwin', skip_by_annotation='edge', verbose=False)


    raw_data = raw_picked.get_data()
    sampling_rate = raw_picked.info['sfreq']

    print(f"  sampling_rate: {sampling_rate} Hz")
    print(f"  data_shape: {raw_data.shape}")


    epoch_length_samples = int(config['epoch_length_sec'] * sampling_rate)
    epoch_num = int(raw_data.shape[1] // epoch_length_samples)

    data = []
    for epoch_id in range(epoch_num):
        start_sample = int(epoch_id * epoch_length_samples)
        end_sample = int((epoch_id + 1) * epoch_length_samples)
        data.append(raw_data[:, start_sample:end_sample])
    data = np.array(data)  # [epochs, 1, samples]

    print(f"  epochs: {data.shape[0]}")

    # 加载标签
    labels = parse_nsrr_xml_stages(label_file_path)
    if labels is None:
        raise ValueError("no label")

    print(f"  label: {len(labels)}")


    min_len = min(len(labels), data.shape[0])
    labels = labels[:min_len]
    data = data[:min_len, :, :]

    print(f"  len: {min_len}")


    data = resample(data, config['target_samples_per_epoch'], axis=2)

    print(f"  resample: {data.shape}")

    return data, labels, eeg_channel


def preprocess_abc_eeg():

    config = CONFIG

    print("=" * 80)
    print("=" * 80)


    save_dir = Path(config['save_dir'])
    eeg_save_dir = save_dir / 'eeg'
    eeg_save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {config['data_dir']}")
    print(f"Annotation directory: {config['annotations_dir']}")
    print(f"Output directory: {eeg_save_dir}")
    print()


    data_dir = Path(config['data_dir'])
    annotations_dir = Path(config['annotations_dir'])

    edf_files = list(data_dir.glob("*.edf"))
    edf_files.sort()

    print(f"find {len(edf_files)} EDF")
    print("-" * 80)


    preprocessed_list = [f.stem for f in eeg_save_dir.glob("*.npz")]

    success_count = 0
    failed_count = 0
    failed_files = []

    for edf_file in tqdm(edf_files, desc="preprocessed"):
        file_name = edf_file.stem  # abc-baseline-900001


        if file_name in preprocessed_list:
            print(f"✓ {file_name} preprocessed")
            success_count += 1
            continue

        print(f"\npreprocessed: {file_name}")


        xml_file = annotations_dir / f"{file_name}-nsrr.xml"

        if not xml_file.exists():
            print(f"  ✗ Annotation file not found: {xml_file.name}")
            failed_count += 1
            failed_files.append((file_name, "Annotation file missing"))
            continue

        try:

            data, labels, channel_name = load_one_subject_data(
                str(edf_file), str(xml_file), config
            )


            unique, counts = np.unique(labels, return_counts=True)
            label_names = {-1: 'Invalid', 0: 'Wake', 1: 'Light', 2: 'Deep', 3: 'REM'}
            dist_str = ', '.join([f"{label_names.get(u, u)}:{c}" for u, c in zip(unique, counts)])
            print(f"  Label distribution: {dist_str}")

            save_path = eeg_save_dir / f"{file_name}.npz"
            np.savez(save_path, x=data[:, 0, :], y=labels)

            print(f"  ✓ Saved successfully: {save_path.name}")
            print(f"    Data shape: {data[:, 0, :].shape}, Label shape: {labels.shape}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ failed: {str(e)}")
            failed_count += 1
            failed_files.append((file_name, str(e)))

    # 打印统计
    print("\n" + "=" * 80)
    print("Processing completed")
    print("=" * 80)
    print(f"success: {success_count}")
    print(f"failed: {failed_count}")

    if failed_files:
        print("\nfailed:")
        for name, error in failed_files[:10]:
            print(f"  - {name}: {error[:50]}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} ")

    print(f"\nOutput directory: {eeg_save_dir}")


def process_all_visits():

    base_data_dir = r"H:\sleepdata\abc\polysomnography\edfs"
    base_annotations_dir = r"H:\sleepdata\abc\polysomnography\annotations-events-nsrr"
    base_save_dir = r"H:\sleepdata\abc-eeg"

    visits = ['baseline', '9-month', '18-month']

    for visit in visits:
        print("\n" + "=" * 80)
        print(f"Processing {visit}")
        print("=" * 80)

        data_dir = os.path.join(base_data_dir, visit)
        annotations_dir = os.path.join(base_annotations_dir, visit)

        if not os.path.exists(data_dir):
            print(f"Directory not found, skipping: {data_dir}")

            continue


        CONFIG['data_dir'] = data_dir
        CONFIG['annotations_dir'] = annotations_dir
        CONFIG['save_dir'] = os.path.join(base_save_dir, visit)

        preprocess_abc_eeg()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--all':

        process_all_visits()
    else:

        preprocess_abc_eeg()