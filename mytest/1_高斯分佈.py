import torch
from scipy.stats import norm

# 定義高斯分布的參數
mean = 1000  # 均值
std = 8      # 標準差
# 生成標準高斯分布的隨機數
data = torch.randn(10000) * std + mean

# 使用 matplotlib 繪製直方圖
import matplotlib.pyplot as plt
plt.hist(data.numpy(), bins=50, density=True, alpha=0.6, color='g')

# 把 data 中的數據超出範圍(975, 1025)的數據列出來
outliers = data[(data < 975) | (data > 1025)]
print("Outliers:", outliers.numpy())

# 計算範圍 [975, 1025] 內的概率
lower_bound = 975
upper_bound = 1025
probability = norm.cdf(upper_bound, loc=mean, scale=std) - norm.cdf(lower_bound, loc=mean, scale=std)
print(f"Probability of data in range [{lower_bound}, {upper_bound}]: {probability:.4f}")

# 繪製理論上的標準高斯分布曲線
import numpy as np
x = np.linspace(960, 1040, 1000)
pdf = (1 / (8 * np.sqrt(2 * np.pi))) * np.exp(-((x - 1000)**2) / (2 * 8**2))
plt.plot(x, pdf, label="Standard Gaussian PDF", color='red')
plt.legend()
plt.title("Histogram of Random Data vs Gaussian PDF")
plt.ylabel("Probability Density")
plt.show()