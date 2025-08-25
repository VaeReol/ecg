
# Copyright 2024 ST-MEM paper authors. <https://github.com/bakqui/ST-MEM>

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm


_LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def get_parser():
    description = "Process WFDB ECG database."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i',
                        '--input_dir',
                        type=str,
                        required=False,
                        default='/home/data/chapman/WFDBRecords',
                        help="Path to the WFDB ECG database directory.")
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        required=False,
                        default='/ssd1/stmem/chapman+ningbo/data',
                        help="Path to the directory where the preprocessed signals will be saved.")
    parser.add_argument('--index_path',
                        type=str,
                        default="/ssd1/stmem/chapman+ningbo/index.csv",
                        help="Path to the index file.")
    args = parser.parse_args()
    return args


def find_records(root_dir):
    """Find all the .hea files in the root directory and its subdirectories.
    Args:
        root_dir (str): The directory to search for .hea files.
    Returns:
        records (set): A set of record names.
                       (e.g., ['database/1/ecg001', 'database/1/ecg001', ..., 'database/9/ecg991'])
    """
    records = set()
    for root, _, files in os.walk(root_dir):
        for file in files:
            extension = os.path.splitext(file)[1]
            if extension == '.hea':
                record = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.add(record)
    records = sorted(records)
    return records

def read_header_with_fallback(file_path):
    try:
        record = wfdb.rdheader(file_path)
        return record
    except ValueError as e:
        print(f"Error reading header for {file_path}: {e}")
        # 返回一个默认的 record 或者 None
        return None
    
def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window.
    """
    if crop_length > x.shape[1]:
        raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    return [x[:, i:i + crop_length] for i in start_idx]
import scipy.io

def read_mat_file(file_path):
    """Read signal and metadata from a .mat file."""
    mat_data = scipy.io.loadmat(file_path)
    # 假设 mat_data 中包含 'signal' 和 'fs' 键
    signal = mat_data['val']
    
    return signal

def run_with_mat(args):
    # Identify the mat files
    record_rel_paths = find_records(args.input_dir)
    print(f"Found {len(record_rel_paths)} records.")

    # Prepare an index dataframe
    index_df = pd.DataFrame(columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE"])

    # Save all the cropped signals
    num_saved = 0
    for record_rel_path in tqdm(record_rel_paths):
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        source_name = record_rel_dir.split("/")[0]
        
        # 使用自定义的读取方法
        signal = read_mat_file(os.path.join(args.input_dir, record_rel_path + '.mat'))
        record = read_header_with_fallback(os.path.join(args.input_dir, record_rel_path))
        if record is None:
            print('record是none')
            continue
        fs = record.fs
        lead_idx = [record.sig_name.index(lead_name) for lead_name in _LEAD_NAMES if lead_name in record.sig_name]
        if len(lead_idx) < len(_LEAD_NAMES):
            missing_leads = set(_LEAD_NAMES) - set(record.sig_name)
            print(f"Warning: Missing leads {missing_leads} for record {record_name}.")
            continue
        signal = signal[lead_idx, :]
        signal_length = signal.shape[1] # 假设信号是 (channels, length) 形状
        if signal_length < 10 * fs:  # 除去小于10s的信号
            print('小于10s')
            continue
        cropped_signals = moving_window_crop(signal, crop_length=10 * fs, crop_stride=10 * fs)
        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                print('裁剪后小于10s')
                continue
            pd.to_pickle(cropped_signal.astype(np.float32),
                         os.path.join(save_dir, f"{record_name}_{idx}.pkl"))
            index_df.loc[num_saved] = [f"{record_name}_{idx}.pkl",
                                       f"{record_name}_{idx}.pkl",
                                       fs,
                                       source_name]
            num_saved += 1

    print(f"Saved {num_saved} cropped signals.")
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    index_df.to_csv(args.index_path, index=False)

if __name__ == "__main__":
    run_with_mat(get_parser())