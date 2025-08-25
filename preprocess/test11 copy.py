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
    parser.add_argument('-i', '--input_txt', type=str,
                        default='/ssd1/chenbo/Project/AIECG-master/data/New_Hosp_Death_28days/test2.txt',
                        help="Path to the txt file listing dat file paths.")
    parser.add_argument('-o', '--output_dir', type=str,
                        default='/ssd1/qinzehao/cusi/6.10ecg/train',
                        help="Path to the directory where the preprocessed signals will be saved.")
    parser.add_argument('--index_dir', type=str,
                        default="/ssd1/qinzehao/cusi/6.10ecg/train",
                        help="Path to the directory where the index files will be saved.")
    args = parser.parse_args()
    return args

def moving_window_crop(x: np.ndarray, crop_length: int, crop_stride: int) -> np.ndarray:
    """Crop the input sequence with a moving window."""
    if crop_length > x.shape[1]:
        raise ValueError(f"crop_length must be smaller than the length of x ({x.shape[1]}).")
    start_idx = np.arange(0, x.shape[1] - crop_length + 1, crop_stride)
    return [x[:, i:i + crop_length] for i in start_idx]

def run(args):
    # 读取txt文件中的dat路径
    with open(args.input_txt, 'r') as f:
        dat_paths = [line.strip() for line in f if line.strip()]

    print(f"Total {len(dat_paths)} .dat files listed in txt.")

    index_list = []
    num_saved = 0
    print('开始处理')

    for dat_path in tqdm(dat_paths):
        if not os.path.isfile(dat_path):
            print(f"跳过：文件不存在 {dat_path}")
            continue

        record_path = dat_path[:-4]  # 去掉 .dat

        # 确定label
        if "live" in dat_path:
            label = 1
        elif "death" in dat_path:
            label = 0
        else:
            print(f"跳过：未知标签 {dat_path}")
            continue

        try:
            signal, record_info = wfdb.rdsamp(record_path)
        except:
            print(f"跳过 {dat_path}, 读取失败")
            continue

        # 选取12导联
        try:
            lead_idx = np.array([record_info["sig_name"].index(lead_name) for lead_name in _LEAD_NAMES])
            signal = signal[:, lead_idx]
        except ValueError:
            print(f"跳过 {dat_path}, 导联信息不匹配")
            continue

        fs = record_info["fs"]
        subject_id_str = record_info['comments'][0]
        subject_id = re.findall(r'\d+', subject_id_str)[0]
        signal_length = record_info["sig_len"]

        # 长度限制（排除小于10秒的ECG信号）
        if signal_length < 10 * fs:
            print(f"跳过：小于10秒 {dat_path}")
            continue

        # 滑动窗口切割
        cropped_signals = moving_window_crop(signal.T, crop_length=10 * fs, crop_stride=10 * fs)

        record_name = os.path.basename(record_path)
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)

        for idx, cropped_signal in enumerate(cropped_signals):
            if cropped_signal.shape[1] != 10 * fs:
                print(f"切割问题：长度不对 {dat_path}")
                continue

            if np.isnan(cropped_signal).any():
                print(f"切割问题：含NaN {dat_path}")
                cropped_signal = np.nan_to_num(cropped_signal, nan=0.0)

            save_path = os.path.join(save_dir, f"{record_name}_{idx}.pkl")
            pd.to_pickle(cropped_signal.astype(np.float32), save_path)

            index_list.append([save_path,  # 绝对路径
                               f"{record_name}_{idx}.pkl",  # 文件名
                               fs,
                               subject_id,
                               label])
            num_saved += 1

    print(f"Saved {num_saved} cropped signals.")

    # 保存index为train.csv
    index_df = pd.DataFrame(index_list, columns=["FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SUBJECT_ID", "LABEL"])
    os.makedirs(args.index_dir, exist_ok=True)
    index_csv_path = os.path.join(args.index_dir, "test_index.csv")
    index_df.to_csv(index_csv_path, index=False)

    print(f"Train index saved to {index_csv_path}")

if __name__ == "__main__":
    run(get_parser())