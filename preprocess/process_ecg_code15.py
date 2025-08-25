import h5py
import numpy as np
import pandas as pd
from pathlib import Path
import os

def padCrop(signal, target_length):
    ori_length = signal.shape[1]
    if ori_length >= target_length:
        target_loc = int((ori_length - target_length) // 2)
        signal = signal[:, target_loc: target_loc + target_length]
    else:
        pad_signal = signal
        while (target_length - pad_signal.shape[1]) >= ori_length:
            pad_signal = np.concatenate((pad_signal, signal), axis=1)
        if target_length - pad_signal.shape[1] > 0:
            pad_num = target_length - pad_signal.shape[1]
            signal = np.concatenate((pad_signal, signal[:, :pad_num]), axis=1)
        else:
            signal = pad_signal
    return signal

# 设置数据文件夹路径
data_folder = Path('/home/data/CODE-15')
dump_folder = Path('/ssd1/stmem/CODE-15')  # 替换为实际路径
os.makedirs(dump_folder, exist_ok=True)  # 创建保存文件夹

# 初始化索引数据的 DataFrame
index_df = pd.DataFrame(columns=["RELATIVE_FILE_PATH", "FILE_NAME", "SAMPLE_RATE", "SOURCE"])

# 初始化.pkl文件计数器
pkl_count = 0

# 获取所有 HDF5 文件
hdf5_files = data_folder.glob('*.hdf5')

for hdf5_File in hdf5_files:
    file_CODE = h5py.File(hdf5_File, "r")
    idx = np.array(file_CODE["exam_id"])
    
    for i in range(len(idx) - 1):
        exam_id = idx[i]
        print(f'processing {exam_id}')
        
        # 读取并转置数据
        raw = file_CODE["tracings"][i].transpose(1, 0)
        
        # 检查是否全零
        if not np.any(raw):
            print("发现全零数据,舍弃之:", exam_id)
            continue
        
        # 检查第二个维度的前100个数据点是否全为零
        if np.all(raw[:, :100] == 0):
            print("前100个数据点全为零, 舍弃之:", exam_id)
            continue
        
        # 假设这里有一些处理后的有效数据
        # 例如：将 raw 数据进行某种处理，得到有效数据
        data = padCrop(raw, 4000)
        valid_data = data  # 这里可以替换为实际的有效数据处理逻辑

        # 生成 .pkl 文件的保存路径
        dump_path = os.path.join(dump_folder, f"{os.path.basename(hdf5_File)}_{i}.pkl")
        
        # 保存数据为 .pkl 文件
        pd.to_pickle(valid_data.astype(np.float32), dump_path)
        
        # 更新索引数据
        index_df.loc[len(index_df)] = [
            f"{os.path.basename(hdf5_File)}_{i}.pkl",  # 相对文件路径
            f"{os.path.basename(hdf5_File)}_{i}.pkl",  # 文件名
            400,  # 采样率
            os.path.dirname(hdf5_File),  # 来源
        ]
        
        # 增加.pkl文件计数器
        pkl_count += 1

# 将索引数据写入 CSV 文件
index_csv_path = os.path.join(dump_folder, 'index_data.csv')
index_df.to_csv(index_csv_path, index=False)

# 关闭 HDF5 文件
file_CODE.close()

# 打印最终保存的.pkl文件个数
print(f"最终保存的.pkl文件个数为: {pkl_count}")