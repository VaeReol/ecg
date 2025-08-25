import os
import wfdb
from collections import Counter

def collect_wfdb_labels(data_dir):
    """
    收集指定目录下所有WFDB文件的标签（comments[2]中的LOINC码组合）
    
    Args:
        data_dir: WFDB数据所在目录
    
    Returns:
        label_counter: 每个标签组合及其出现次数的计数器
        code_set: 所有唯一LOINC码的集合
    """
    label_counter = Counter()  # 统计标签组合出现的次数
    all_codes = set()          # 所有唯一的LOINC码
    
    # 遍历目录下所有文件
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hea'):  # 只处理头文件，因为rdheader只需要头文件
                # 获取不含扩展名的文件名作为记录名
                record_name = os.path.splitext(file)[0]
                record_path = os.path.join(root, record_name)
                
                try:
                    # 读取记录信息（不读取信号以提高效率）
                    record_info = wfdb.rdheader(record_path)
                    
                    # 检查基本记录格式
                    if not hasattr(record_info, 'comments'):
                        print(f"警告: {record_path} 缺少comments属性")
                        continue
                    
                    # 提取标签（假设格式为: Dx: 123456,789012,...）
                    if len(record_info.comments) > 2:
                        comment = record_info.comments[2]
                        if comment.startswith('Dx: '):
                            # 提取LOINC码并去重、排序
                            codes = comment[3:].split(',')
                            codes = [code.strip() for code in codes if code.strip()]
                            codes_sorted = sorted(codes)  # 排序以确保相同组合顺序一致
                            
                            # 组合成字符串作为标签
                            label = ','.join(codes_sorted)
                            
                            # 更新计数器和LOINC码集合
                            label_counter[label] += 1
                            all_codes.update(codes)
                except Exception as e:
                    print(f"Error reading {record_path}: {e}")
                    # 尝试读取头文件内容以获取更多信息
                    try:
                        with open(f"{record_path}.hea", 'r') as f:
                            print(f"文件 {record_path}.hea 的前200个字符:")
                            print(f.read(200))
                    except Exception:
                        pass
                    continue
    
    return label_counter, all_codes

# 使用示例
data_dir = '/ssd1/qinzehao/chapman/physionet.org/content/challenge-2021/1.0.3/training/chapman_shaoxing/g1'
label_counter, all_codes = collect_wfdb_labels(data_dir)

# 输出统计结果
print(f"共发现 {len(label_counter)} 种不同的标签组合")
print(f"共涉及 {len(all_codes)} 个不同的LOINC码")