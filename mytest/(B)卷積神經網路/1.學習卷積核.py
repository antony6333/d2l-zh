# 學習卷積核 - 基於 conv-layer.ipynb 教學章節的練習
"""
本程式基於 pytorch/chapter_convolutional-neural-networks/conv-layer.ipynb 教學章節
主要學習內容：
1. 互相關運算的實現與理解
2. 卷積層的基本構建
3. 邊緣檢測的應用
4. 通過梯度下降學習卷積核參數
5. 感受野的概念

核心概念：
- 卷積神經網路實際使用的是互相關運算，而非嚴格的卷積運算
- 卷積核可以用來檢測圖像中的特定特徵（如邊緣）
- 卷積核的參數可以通過反向傳播和梯度下降來學習
"""

import torch
from torch import nn
import numpy as np

def corr2d(X, K):
    """
    計算二維互相關運算
    
    Args:
        X: 輸入張量 (h_in, w_in)
        K: 卷積核張量 (k_h, k_w)
    
    Returns:
        Y: 輸出張量 (h_out, w_out)，其中 h_out = h_in - k_h + 1, w_out = w_in - k_w + 1
    """
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 對應位置元素相乘後求和
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    """
    自定義的二維卷積層
    """
    def __init__(self, kernel_size):
        super().__init__()
        # 卷積核權重作為可學習參數
        self.weight = nn.Parameter(torch.rand(kernel_size))
        # 偏置項
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

def demonstrate_cross_correlation():
    """
    示範基本的互相關運算
    """
    print("=== 基本互相關運算示範 ===")
    
    # 創建輸入張量（3x3）
    X = torch.tensor([[0.0, 1.0, 2.0], 
                      [3.0, 4.0, 5.0], 
                      [6.0, 7.0, 8.0]])
    print(f"輸入張量 X:\n{X}")
    
    # 創建卷積核（2x2）
    K = torch.tensor([[0.0, 1.0], 
                      [2.0, 3.0]])
    print(f"卷積核 K:\n{K}")
    
    # 執行互相關運算
    Y = corr2d(X, K)
    print(f"輸出張量 Y:\n{Y}")
    
    # 驗證計算結果
    print("\n手動計算驗證：")
    print(f"Y[0,0] = 0*0 + 1*1 + 3*2 + 4*3 = {0*0 + 1*1 + 3*2 + 4*3}")
    print(f"Y[0,1] = 1*0 + 2*1 + 4*2 + 5*3 = {1*0 + 2*1 + 4*2 + 5*3}")
    print(f"輸出大小：{Y.shape} = 輸入大小{X.shape} - 卷積核大小{K.shape} + (1,1)")

def edge_detection_example():
    """
    邊緣檢測範例
    """
    print("\n=== 邊緣檢測範例 ===")
    
    # 創建一個6x8的黑白圖像，中間4列為黑色(0)，其餘為白色(1)
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    print(f"原始圖像 X:\n{X}")
    
    # 水平邊緣檢測卷積核
    K = torch.tensor([[1.0, -1.0]])
    print(f"邊緣檢測卷積核 K:\n{K}")
    
    # 執行邊緣檢測
    Y = corr2d(X, K)
    print(f"邊緣檢測結果 Y:\n{Y}")
    print("說明：1代表從白色到黑色的邊緣，-1代表從黑色到白色的邊緣")
    
    # 測試轉置後的效果
    print(f"\n轉置圖像後的邊緣檢測結果:\n{corr2d(X.t(), K)}")
    print("說明：垂直邊緣檢測器無法檢測水平邊緣")

def learn_convolution_kernel():
    """
    通過梯度下降學習卷積核參數
    """
    print("\n=== 學習卷積核參數 ===")
    
    # 準備訓練數據
    X = torch.ones((6, 8))
    X[:, 2:6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    
    print(f"目標卷積核:\n{K}")
    print(f"期望輸出:\n{Y}")
    
    # 使用PyTorch內建的卷積層
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)
    
    # 調整張量形狀以符合PyTorch卷積層的要求 (batch_size, channels, height, width)
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    
    # 設定學習率
    lr = 3e-2
    
    print(f"\n初始卷積核:\n{conv2d.weight.data.reshape((1, 2))}")
    
    # 訓練過程
    for i in range(10):
        # 前向傳播
        Y_hat = conv2d(X)
        # 計算損失（均方誤差）
        l = (Y_hat - Y) ** 2
        # 清零梯度
        conv2d.zero_grad()
        # 反向傳播
        l.sum().backward()
        # 更新參數
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {l.sum():.6f}')
    
    # 顯示學習到的卷積核
    learned_kernel = conv2d.weight.data.reshape((1, 2))
    print(f"\n學習到的卷積核:\n{learned_kernel}")
    print(f"與目標卷積核的差異:\n{learned_kernel - K}")

'''
感受野擴大的概念
1.單層卷積的感受野：
  在單層卷積中，感受野的大小等於卷積核的大小。例如，使用一個 3x3 的卷積核，輸出張量的每個元素對應於輸入張量中的一個 3x3 區域。
  這意味著輸出張量的每個元素「感知」到的輸入範圍是固定的。
2.多層卷積的感受野：
  當多層卷積疊加時，感受野會隨著層數的增加而擴大。例如：
      第一層卷積核大小為 3x3，輸出張量的每個元素感受野為 3x3。
      第二層再使用 3x3 的卷積核，輸出張量的每個元素感受野會擴大到 5x5，因為第二層的每個輸出元素實際上是第一層輸出的 3x3 區域的結果。
3.感受野擴大的意義：
  感受野的擴大使得網路能夠捕捉更大範圍的特徵，這對於處理高層次的全局特徵（如物體形狀或整體結構）非常重要。
  在深層網路中，感受野的擴大能夠讓網路學習到更抽象的特徵，從而提升模型的表現能力。
'''
def demonstrate_receptive_field():
    """
    示範感受野的概念
    """
    print("\n=== 感受野概念示範 ===")
    
    # 創建一個簡單的多層卷積網路
    print("構建兩層卷積網路來展示感受野擴大的效果")
    
    # 第一層：3x3輸入 -> 2x2卷積核 -> 2x2輸出
    X = torch.tensor([[1.0, 2.0, 3.0], 
                      [4.0, 5.0, 6.0], 
                      [7.0, 8.0, 9.0]])
    K1 = torch.tensor([[1.0, 0.0], 
                       [0.0, 1.0]])
    
    print(f"輸入張量 (3x3):\n{X}")
    print(f"第一層卷積核 (2x2):\n{K1}")
    
    Y1 = corr2d(X, K1)
    print(f"第一層輸出 (2x2):\n{Y1}")
    
    # 第二層：2x2輸入 -> 2x2卷積核 -> 1x1輸出
    K2 = torch.tensor([[0.5, 0.5], 
                       [0.5, 0.5]])
    print(f"第二層卷積核 (2x2):\n{K2}")
    
    Y2 = corr2d(Y1, K2)
    print(f"第二層輸出 (1x1):\n{Y2}")
    
    print("\n感受野分析：")
    print("- 第一層輸出的每個元素感受野：2x2")
    print("- 第二層輸出的元素感受野：3x3（覆蓋整個原始輸入）")
    print("- 隨著網路深度增加，感受野逐漸擴大，能夠捕捉更大範圍的特徵")

def practice_exercises():
    """
    完成教學章節中的練習題
    """
    print("\n=== 練習題解答 ===")
    
    # 練習1：對角線邊緣檢測
    print("練習1：構建具有對角線邊緣的圖像")
    
    # 創建對角線圖像
    X_diag = torch.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            if i == j:  # 主對角線為1
                X_diag[i, j] = 1.0
    
    print(f"對角線圖像:\n{X_diag}")
    
    # 使用水平邊緣檢測器
    K_horizontal = torch.tensor([[1.0, -1.0]])
    result_diag = corr2d(X_diag, K_horizontal)
    print(f"水平邊緣檢測器在對角線圖像上的結果:\n{result_diag}")
    
    # 練習4：設計特殊卷積核
    print("\n練習4：設計特殊卷積核")
    
    # 二階導數核（拉普拉斯算子的簡化版本）
    laplacian_kernel = torch.tensor([[-1.0, 2.0, -1.0]])
    print(f"一維二階導數核:\n{laplacian_kernel}")
    
    # 測試在簡單信號上
    signal = torch.tensor([[1.0, 1.0, 2.0, 2.0, 1.0, 1.0]])
    second_derivative = corr2d(signal, laplacian_kernel)
    print(f"測試信號:\n{signal}")
    print(f"二階導數結果:\n{second_derivative}")
    print("說明：二階導數可以檢測信號的曲率變化")

def main():
    """
    主函數：執行所有示範和練習
    """
    print("卷積核學習與應用示範")
    print("=" * 50)
    
    # 基本互相關運算
    demonstrate_cross_correlation()
    
    # 邊緣檢測應用
    edge_detection_example()
    
    # 學習卷積核參數
    learn_convolution_kernel()
    
    # 感受野概念
    demonstrate_receptive_field()
    
    # 練習題
    practice_exercises()
    
    print("\n" + "=" * 50)
    print("程式執行完成！")
    print("\n重要概念總結：")
    print("1. 卷積神經網路使用互相關運算，不是嚴格的卷積運算")
    print("2. 卷積核可以檢測特定的圖像特徵（邊緣、紋理等）")
    print("3. 卷積核參數可以通過反向傳播自動學習")
    print("4. 感受野隨網路深度增加而擴大，能捕捉更大範圍的特徵")
    print("5. 輸出大小 = 輸入大小 - 卷積核大小 + 1")

if __name__ == "__main__":
    main()
