"""
根據教學章節 pytorch/chapter_deep-learning-computation/custom-layer.ipynb 所作的練習

本章節核心概念：
1. 自定義層的設計與實作
2. 不帶參數的層：繼承 nn.Module 並實現 forward 方法
3. 帶參數的層：使用 nn.Parameter 創建可訓練參數
4. 層的組合與重用：自定義層可以像內置層一樣使用

主要技術點：
- 繼承 nn.Module 基礎類
- 實現 __init__ 和 forward 方法
- 使用 nn.Parameter 定義可訓練參數
- 理解參數的自動梯度計算機制
"""

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


print("=== 自定義層練習 ===")

# ===== 1. 不帶參數的自定義層 =====
print("\n1. 不帶參數的自定義層示例")

class CenteredLayer(nn.Module):
    """
    自定義的中心化層：從輸入中減去均值
    這是一個不帶參數的層的典型例子
    """
    def __init__(self):
        super().__init__()

    def forward(self, X):
        # 計算輸入張量的均值並從每個元素中減去
        return X - X.mean()


# 測試中心化層
layer = CenteredLayer()
test_data = torch.FloatTensor([1, 2, 3, 4, 5])
centered_data = layer(test_data)
print(f"原始數據: {test_data}")
print(f"中心化後: {centered_data}")
print(f"中心化後的均值: {centered_data.mean():.6f}")

# 將自定義層嵌入到更複雜的網路中
net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
Y = net(torch.rand(4, 8))
print(f"網路輸出的均值: {Y.mean():.6f}")


# ===== 2. 帶參數的自定義層 =====
print("\n2. 帶參數的自定義層示例")

class MyLinear(nn.Module):
    """
    自定義的線性層：實現全連接層功能
    使用 ReLU 作為激活函數
    """
    def __init__(self, in_units, units):
        super().__init__()
        # 使用 nn.Parameter 創建可訓練的權重和偏置
        # nn.Parameter 用於將張量標記為模型的可訓練參數(即model.parameters()中)，這些參數會自動參與梯度計算
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    
    def forward(self, X):
        # 執行線性變換：X @ W + b，然後應用 ReLU
        # torch.matmul() 是執行矩陣乘法的函數
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)


# 測試自定義線性層
linear = MyLinear(5, 3)
print(f"權重形狀: {linear.weight.shape}")
print(f"偏置形狀: {linear.bias.shape}")

# 執行前向傳播
test_input = torch.rand(2, 5)
output = linear(test_input)
print(f"輸入形狀: {test_input.shape}")
print(f"輸出形狀: {output.shape}")

# 構建包含自定義層的網路
net = nn.Sequential(MyLinear(64, 8), MyLinear(8, 1))
result = net(torch.rand(2, 64))
print(f"網路最終輸出形狀: {result.shape}")


# ===== 3. 練習題解答 =====
print("\n3. 練習題解答")

# 練習1: 設計一個接受輸入並計算張量降維的層，返回 y_k = Σ(i,j) W_ijk * x_i * x_j
class TensorReductionLayer(nn.Module):
    """
    張量降維層：計算 y_k = Σ(i,j) W_ijk * x_i * x_j
    輸入: (batch_size, input_dim)
    輸出: (batch_size, output_dim)
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # W_ijk 是一個三維張量：(input_dim, input_dim, output_dim)
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, X):
        batch_size = X.shape[0]
        # X 形狀: (batch_size, input_dim)
        
        # 計算 x_i * x_j 的外積，得到 (batch_size, input_dim, input_dim)
        X_outer = torch.einsum('bi,bj->bij', X, X)
        
        # 與權重張量相乘並求和，得到 (batch_size, output_dim)
        result = torch.einsum('bij,ijk->bk', X_outer, self.weight)
        
        return result


print("練習1: 張量降維層")
tensor_layer = TensorReductionLayer(4, 2)
test_input = torch.randn(3, 4)  # batch_size=3, input_dim=4
output = tensor_layer(test_input)
print(f"輸入形狀: {test_input.shape}")
print(f"輸出形狀: {output.shape}")
print(f"權重形狀: {tensor_layer.weight.shape}")


# 練習2: 設計一個返回輸入數據的傅立葉系數前半部分的層
class FFTLayer(nn.Module):
    """
    傅立葉變換層：返回輸入的傅立葉系數前半部分
    使用 PyTorch 的 FFT 函數進行實數到複數的變換
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, X):
        # 對最後一個維度進行 FFT
        fft_result = torch.fft.fft(X, dim=-1)
        
        # 取前半部分的係數
        half_length = X.shape[-1] // 2
        fft_half = fft_result[..., :half_length]
        
        # 返回複數的模（幅度）
        return torch.abs(fft_half)


print("\n練習2: 傅立葉變換層")
fft_layer = FFTLayer()

# 創建一個簡單的正弦波信號進行測試
t = torch.linspace(0, 2*np.pi, 64)
signal = torch.sin(5*t) + 0.5*torch.sin(10*t)  # 5Hz 和 10Hz 的混合信號
signal = signal.unsqueeze(0)  # 添加 batch 維度

fft_result = fft_layer(signal)
print(f"原始信號形狀: {signal.shape}")
print(f"FFT 結果形狀: {fft_result.shape}")
print(f"前10個傅立葉係數: {fft_result[0, :10]}")


# ===== 4. 綜合示例：組合多個自定義層 =====
print("\n4. 綜合示例：組合多個自定義層")

class CustomNet(nn.Module):
    """
    組合多個自定義層的網路
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.center = CenteredLayer()
        self.linear1 = MyLinear(input_dim, hidden_dim)
        self.linear2 = MyLinear(hidden_dim, output_dim)
        self.tensor_reduce = TensorReductionLayer(output_dim, 1)
    
    def forward(self, X):
        # 先中心化輸入
        X = self.center(X)
        # 通過兩個自定義線性層
        X = self.linear1(X)
        X = self.linear2(X)
        # 最後進行張量降維
        X = self.tensor_reduce(X)
        return X


# 測試組合網路
custom_net = CustomNet(10, 16, 8)
test_input = torch.randn(5, 10)
final_output = custom_net(test_input)

print(f"組合網路輸入形狀: {test_input.shape}")
print(f"組合網路輸出形狀: {final_output.shape}")

# 檢查網路參數
total_params = sum(p.numel() for p in custom_net.parameters())
print(f"網路總參數數量: {total_params}")

print("\n=== 自定義層練習完成 ===")
print("核心要點總結：")
print("1. 自定義層需要繼承 nn.Module")
print("2. 必須實現 __init__ 和 forward 方法")
print("3. 使用 nn.Parameter 定義可訓練參數")
print("4. 自定義層可以像內置層一樣組合使用")
print("5. 參數管理和梯度計算由框架自動處理")
