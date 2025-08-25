import argparse
import os
import re
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split

_LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def get_parser():
    description = "Process WFDB ECG database."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_dir', type=str,
                        default='/ssd1/chenbo/Dataset/SCDRawdata',
                        help="Path to the WFDB ECG database directory.")
    parser.add_argument('-o', '--output_dir', type=str,
                        default='/ssd1/qinzehao/cusi/test',
                        help="Path to the directory where the preprocessed signals will be saved.")
    parser.add_argument('--index_dir', type=str,
                        default="/ssd1/qinzehao/cusi/test",
                        help="Path to the directory where the index files will be saved.")
    args = parser.parse_args()
    return args


def find_records(root_dir):
    """Find all the .hea files in the root directory and its subdirectories."""
    records = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.hea'):
                record_path = os.path.relpath(os.path.join(root, file), root_dir)[:-4]
                records.append(record_path)
    return sorted(records)


def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window."""
    if crop_length > x.shape[1]:
        raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    return [x[:, i:i + crop_length] for i in start_idx]


def run(args):
    # 找到所有 .hea 文件
    record_rel_paths = find_records(args.input_dir)
    print(f"Found {len(record_rel_paths)} records.")

    # 存储 index 信息
    index_list = []

    num_saved = 0
    print('开始处理')

    for record_rel_path in tqdm(record_rel_paths):
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)

        # 确定 LABEL
        if "live" in record_rel_dir:
            label = 0
        elif "death" in record_rel_dir:
            label = 1
        else:
            continue  # 其他情况跳过

        try:
            signal, record_info = wfdb.rdsamp(os.path.join(args.input_dir, record_rel_path))
        except:
            print(f"跳过 {record_rel_path}, 读取失败")
            continue

        # 选取 12 导联
        try:
            lead_idx = np.array([record_info["sig_name"].index(lead_name) for lead_name in _LEAD_NAMES])
            signal = signal[:, lead_idx]
        except ValueError:
            print(f"跳过 {record_rel_path}, 导联信息不匹配")
            continue

        fs = record_info["fs"]
        subject_id_str = record_info['comments'][0]
        subject_id = re.findall(r'\d+', subject_id_str)[0]
        signal_length = record_info["sig_len"]

        # 长度限制（排除小于10秒的ECG信号）
        if signal_length < 10 * fs:
            print('小于10s')
            continue

        # 进行滑动窗口切割
        cropped_signals = moving_window_crop(signal.T, crop_length=10 * fs, crop_stride=10 * fs)

        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs:
                print('切割出现问题：长度不对')
                continue

            if np.isnan(cropped_signal).any():
                print('切割出现问题：含有 NaN')
                cropped_signal = np.nan_to_num(cropped_signal, nan=0.0)

            # 保存切割的信号
            save_path = os.path.join(save_dir, f"{record_name}_{idx}.pkl")
            pd.to_pickle(cropped_signal.astype(np.float32), save_path)

            # 存入索引信息
            index_list.append([f"{record_rel_path}_{idx}.pkl",
                               f"{record_name}_{idx}.pkl",
                               fs,
                               subject_id,
                               label])

            num_saved += 1

    print(f"Saved {num_saved} cropped signals.")

    # 转换为 DataFrame 并打乱数据
    index_df = pd.DataFrame(index_list, columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SUBJECT_ID", "LABEL"])
    index_df = index_df.sample(frac=1).reset_index(drop=True)  # 随机打乱数据

    # 划分 train / val / test（80% / 10% / 10%）
    train_df, test_df = train_test_split(index_df, test_size=0.2, random_state=42, stratify=index_df["LABEL"])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df["LABEL"])

    # 创建目录并保存
    os.makedirs(args.index_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.index_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.index_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.index_dir, "test.csv"), index=False)

    print("Index files saved: train_index.csv, val_index.csv, test_index.csv")


if __name__ == "__main__":
    run(get_parser())
