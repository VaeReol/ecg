import argparse
import os
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import scipy.io as sio
_LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# 秦余臻的cpsc lead['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']



def get_parser():
    description = "Process WFDB ECG database."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-i', '--input_dir', type=str,
                        default='/home/yuzhenqin/dataset/TRAIN',
                        help="Path to the WFDB ECG database directory.")
    parser.add_argument('-o', '--output_dir', type=str,
                        default='/ssd1/stmem/processed_cpsc/ecg',
                        help="Path to the directory where the preprocessed signals will be saved.")
    parser.add_argument('--index_dir', type=str,
                        default="/ssd1/stmem/processed_cpsc",
                        help="Path to the directory where the index files will be saved.")
    args = parser.parse_args()
    return args


def find_records(root_dir):
    """Find all the .hea files in the root directory and its subdirectories."""
    records = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mat'):
                record_path = os.path.relpath(os.path.join(root, file), root_dir)
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
    csv_path = '/home/yuzhenqin/dataset/TRAIN/REFERENCE.csv'
    df = pd.read_csv(csv_path)
    num_saved = 0
    print('开始处理')

    for record_rel_path in tqdm(record_rel_paths):
        record_rel_dir, record_name = os.path.split(record_rel_path)
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        record_name = record_name[:-4]
        label = df.loc[df['Recording'] == record_name, 'First_label'].values.item() - 1
        label2 = df.loc[df['Recording'] == record_name, 'Second_label'].values.item()
        print(label2)
        if pd.isna(label2):  # 检查 label2 是否为空
            pass  # label2 为空时执行 pass
        else:
            print('出现多标签')
            continue
        # print(label)
        try:
            signal = sio.loadmat(os.path.join(args.input_dir, record_rel_path))['ECG'][0][0][2]
        except:
            print(f"跳过 {record_rel_path}, 读取失败")
            continue

        # 选取 12 导联
        try:
            lead_idx = np.arange(12)
            signal = signal[lead_idx, :]
            print(signal.shape)
        except ValueError:
            print(f"跳过 {record_rel_path}, 导联信息不匹配")
            continue

        fs = 500
        signal_length = signal.shape[1]

        # 长度限制（排除小于10秒的ECG信号）
        if signal_length < 10 * fs:
            print('排除小于10s')
            continue

        # 进行滑动窗口切割
        cropped_signals = moving_window_crop(signal, crop_length=10 * fs, crop_stride=10 * fs)

        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs or np.isnan(cropped_signal).any():
                print('分割后发现异常')
                continue

            # 保存切割的信号
            save_path = os.path.join(save_dir, f"{record_name}_{idx}.pkl")
            pd.to_pickle(cropped_signal.astype(np.float32), save_path)

            # 存入索引信息
            index_list.append([f"{record_rel_path}_{idx}.pkl",
                               f"{record_name}_{idx}.pkl",
                               fs,
                               label])

            num_saved += 1

    print(f"Saved {num_saved} cropped signals.")

    # 转换为 DataFrame 并打乱数据
    index_df = pd.DataFrame(index_list, columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "LABEL"])
    index_df = index_df.sample(frac=1).reset_index(drop=True)  # 随机打乱数据

    # 划分 train / val / test（80% / 10% / 10%）
    train_df, test_df = train_test_split(index_df, test_size=0.2, random_state=42, stratify=index_df["LABEL"])
    val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df["LABEL"])

    # 创建目录并保存
    os.makedirs(args.index_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.index_dir, "train_index.csv"), index=False)
    val_df.to_csv(os.path.join(args.index_dir, "val_index.csv"), index=False)
    test_df.to_csv(os.path.join(args.index_dir, "test_index.csv"), index=False)

    print("Index files saved: train.csv, val.csv, test.csv")


if __name__ == "__main__":
    run(get_parser())
