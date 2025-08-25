import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt


# =============== 滤波器定义 ===============
class SOSFilter:
    """Apply SOS filter to the input sequence."""
    def __init__(self,
                 fs: int,
                 cutoff: float,
                 order: int = 5,
                 btype: str = 'highpass') -> None:
        self.sos = butter(order, cutoff, btype=btype, fs=fs, output='sos')

    def __call__(self, x):
        return sosfiltfilt(self.sos, x)


class HighpassFilter(SOSFilter):
    """Highpass filter."""
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(HighpassFilter, self).__init__(fs, cutoff, order, btype='highpass')


class LowpassFilter(SOSFilter):
    """Lowpass filter."""
    def __init__(self, fs: int, cutoff: float, order: int = 5) -> None:
        super(LowpassFilter, self).__init__(fs, cutoff, order, btype='lowpass')


# =============== FFT增强函数 ===============
def fft_aug(x, rate=0.5, dim=1):
    # Perform FFT on the input data
    x_f = np.fft.fft(x, axis=dim)

    # Create the mask (uniform random values)
    m = np.random.rand(*x_f.shape) < rate

    # Compute the amplitude and find the dominant frequencies
    amp = np.abs(x_f)
    sorted_indices = np.argsort(amp, axis=dim)[..., ::-1]  # Sort indices in descending order
    dominant_mask = np.zeros_like(m)
    dominant_mask[np.arange(m.shape[0])[:, None], sorted_indices[:, 2:]] = 1  # Keep top 2 dominant frequencies

    # Combine the original mask and dominant mask
    m = np.logical_and(m, dominant_mask)

    # Mask the real and imaginary parts of x_f
    freal = np.real(x_f)
    fimag = np.imag(x_f)
    freal[m] = 0
    fimag[m] = 0

    # Shuffle the data indices and apply to another batch (x2)
    b_idx = np.arange(x.shape[0])
    np.random.shuffle(b_idx)
    x2 = x[b_idx]

    # Perform FFT on the shuffled batch (x2)
    x2_f = np.fft.fft(x2, axis=dim)

    # Invert the mask for the second batch (x2)
    m2 = np.logical_not(m)
    freal2 = np.real(x2_f)
    fimag2 = np.imag(x2_f)
    freal2[m2] = 0
    fimag2[m2] = 0

    # Combine the real and imaginary parts from both batches
    freal += freal2
    fimag += fimag2

    # Reconstruct the complex FFT result and apply inverse FFT
    x_f = freal + 1j * fimag
    x = np.abs(np.fft.ifft(x_f, axis=dim))

    return x


# =============== 主流程 ===============
# 读取数据
data = pd.read_pickle(r"/home/zehaoqin/ST-MEM/dataset/ptbxldata/00001_hr_0.pkl")
data = np.array(data)   # 转换为 numpy 数组，假设 shape = (leads, length)

# 设置采样率（PTB-XL 是 500Hz，如果你的是别的要改）
fs = 500  

# 定义滤波器
hp_filter = HighpassFilter(fs=fs, cutoff=0.65, order=5)
lp_filter = LowpassFilter(fs=fs, cutoff=40, order=5)

# 先对每个导联做滤波
filted_data = np.zeros_like(data)
for i in range(data.shape[0]):
    x = data[i]
    x = hp_filter(x)
    x = lp_filter(x)
    filted_data[i] = x

# 再做数据增强
aug_data = fft_aug(filted_data, dim=-1)

# =============== 绘图 ===============
num_leads = data.shape[0]
plt.figure(figsize=(15, num_leads * 2.5))

for i in range(num_leads):
    plt.subplot(num_leads, 2, 2*i+1)
    plt.plot(filted_data[i])
    plt.title(f"Lead {i+1} - Original (Filtered)")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(num_leads, 2, 2*i+2)
    plt.plot(aug_data[i])
    plt.title(f"Lead {i+1} - Augmented")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

plt.tight_layout()
plt.savefig("ecg_augmented_all_leads.png")
plt.show()

print("Original data shape:", data.shape)
print("Filtered data shape:", filted_data.shape)
print("Augmented data shape:", aug_data.shape)
