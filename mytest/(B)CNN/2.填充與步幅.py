"""
填充與步幅 - 根據 pytorch/chapter_convolutional-neural-networks/padding-and-strides.ipynb 的練習

本章節主要內容：
1. 填充(Padding)的概念與作用
2. 步幅(Stride)的概念與應用
3. 填充與步幅對輸出尺寸的影響
4. 卷積層參數設定的實作

核心概念：
- 填充：在輸入圖像邊界填充元素（通常為0），解決卷積後尺寸縮小的問題，並保持輸出特徵圖的空間尺寸，避免邊界資訊的丟失。
- 步幅：卷積窗口每次移動的距離，可用於降採樣，減少輸出特徵圖的尺寸，從而降低計算量和記憶體使用，並增大感受野。
- 輸出尺寸計算公式：(n_h-k_h+p_h+1) × (n_w-k_w+p_w+1)
- 步幅輸出尺寸：⌊(n_h-k_h+p_h+s_h)/s_h⌋ × ⌊(n_w-k_w+p_w+s_w)/s_w⌋
"""

import torch
from torch import nn
import numpy as np


def comp_conv2d(conv2d, X):
    """
    計算卷積層的輔助函數
    將2D輸入轉換為4D張量(batch_size, channels, height, width)
    然後提取輸出的空間維度
    --------------------------
    在 PyTorch 中，卷積層（如 nn.Conv2d）要求輸入的張量必須是 4 維的，
    因為它需要知道批次大小和通道數。這裡的 reshape 操作是為了滿足卷積層的輸入要求。
    """
    # 增加批次維度和通道維度：(1, 1, height, width)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 移除批次和通道維度，只返回空間維度
    return Y.reshape(Y.shape[2:])


def demonstrate_padding():
    """
    示範填充的效果
    """
    print("=== 填充(Padding)示範 ===")
    
    # 建立8x8的隨機輸入
    X = torch.rand(size=(8, 8))
    print(f"原始輸入尺寸: {X.shape}")
    
    # 案例1: 3x3卷積核，填充1
    conv2d_1 = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    output_1 = comp_conv2d(conv2d_1, X)
    print(f"3x3卷積核，填充1的輸出尺寸: {output_1.shape}")
    
    # 案例2: 5x3卷積核，填充(2,1)
    conv2d_2 = nn.Conv2d(1, 1, kernel_size=(5, 3), padding=(2, 1))
    output_2 = comp_conv2d(conv2d_2, X)
    print(f"5x3卷積核，填充(2,1)的輸出尺寸: {output_2.shape}")
    
    # 計算驗證
    n_h, n_w = 8, 8
    k_h, k_w = 5, 3
    p_h, p_w = 2, 1
    expected_h = n_h - k_h + p_h + 1
    expected_w = n_w - k_w + p_w + 1
    print(f"理論計算輸出尺寸: ({expected_h}, {expected_w})")
    print()


def demonstrate_stride():
    """
    示範步幅的效果
    """
    print("=== 步幅(Stride)示範 ===")
    
    X = torch.rand(size=(8, 8))
    print(f"原始輸入尺寸: {X.shape}")
    
    # 案例1: 步幅為2
    conv2d_1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
    output_1 = comp_conv2d(conv2d_1, X)
    print(f"3x3卷積核，填充1，步幅2的輸出尺寸: {output_1.shape}")
    
    # 案例2: 複雜的步幅設定
    conv2d_2 = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    output_2 = comp_conv2d(conv2d_2, X)
    print(f"3x5卷積核，填充(0,1)，步幅(3,4)的輸出尺寸: {output_2.shape}")
    
    # 計算驗證（練習題1）
    n_h, n_w = 8, 8
    k_h, k_w = 3, 5
    p_h, p_w = 0, 1
    s_h, s_w = 3, 4
    expected_h = (n_h - k_h + p_h + s_h) // s_h
    expected_w = (n_w - k_w + p_w + s_w) // s_w
    print(f"理論計算輸出尺寸: ({expected_h}, {expected_w})")
    print()


def exercise_1():
    """
    練習題1：計算最後一個示例的輸出形狀
    """
    print("=== 練習題1：計算輸出形狀 ===")
    
    # 輸入參數
    n_h, n_w = 8, 8  # 輸入尺寸
    k_h, k_w = 3, 5  # 卷積核尺寸
    p_h, p_w = 0, 1  # 填充
    s_h, s_w = 3, 4  # 步幅
    
    print(f"輸入尺寸: {n_h} × {n_w}")
    print(f"卷積核尺寸: {k_h} × {k_w}")
    print(f"填充: ({p_h}, {p_w})")
    print(f"步幅: ({s_h}, {s_w})")
    
    # 使用公式計算
    output_h = (n_h - k_h + p_h + s_h) // s_h
    output_w = (n_w - k_w + p_w + s_w) // s_w
    
    print(f"計算過程：")
    print(f"  高度: ⌊({n_h} - {k_h} + {p_h} + {s_h}) / {s_h}⌋ = ⌊{n_h - k_h + p_h + s_h} / {s_h}⌋ = {output_h}")
    print(f"  寬度: ⌊({n_w} - {k_w} + {p_w} + {s_w}) / {s_w}⌋ = ⌊{n_w - k_w + p_w + s_w} / {s_w}⌋ = {output_w}")
    print(f"理論輸出尺寸: ({output_h}, {output_w})")
    
    # 實際驗證
    X = torch.rand(size=(8, 8))
    conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
    actual_output = comp_conv2d(conv2d, X)
    print(f"實際輸出尺寸: {actual_output.shape}")
    print(f"計算結果{'正確' if (output_h, output_w) == tuple(actual_output.shape) else '錯誤'}")
    print()


def exercise_2():
    """
    練習題2：嘗試其他填充和步幅組合
    """
    print("=== 練習題2：不同填充和步幅組合 ===")
    
    X = torch.rand(size=(10, 10))  # 使用10x10的輸入
    print(f"輸入尺寸: {X.shape}")
    
    # 測試不同的組合
    test_cases = [
        {"kernel_size": 3, "padding": 0, "stride": 1, "name": "無填充，步幅1"},
        {"kernel_size": 3, "padding": 1, "stride": 1, "name": "填充1，步幅1(保持尺寸)"},
        {"kernel_size": 5, "padding": 2, "stride": 1, "name": "填充2，步幅1(保持尺寸)"},
        {"kernel_size": 3, "padding": 1, "stride": 2, "name": "填充1，步幅2(減半)"},
        {"kernel_size": 7, "padding": 3, "stride": 3, "name": "填充3，步幅3"},
        {"kernel_size": (3, 5), "padding": (1, 2), "stride": (1, 2), "name": "不對稱設定"},
    ]
    
    for case in test_cases:
        conv2d = nn.Conv2d(1, 1, **{k: v for k, v in case.items() if k != "name"})
        output = comp_conv2d(conv2d, X)
        print(f"{case['name']}: {output.shape}")
    print()


def exercise_3():
    """
    練習題3：音訊信號中步幅2的意義
    """
    print("=== 練習題3：音訊信號中的步幅 ===")
    
    print("對於音訊信號，步幅2的意義：")
    print("1. 降採樣：將採樣率減半，例如從44.1kHz降到22.05kHz")
    print("2. 減少計算量：處理的樣本點數量減半")
    print("3. 時間解析度降低：每個輸出點代表更長的時間窗口")
    print("4. 頻率範圍限制：根據奈奎斯特定理，最高可表示頻率也會減半")
    
    # 模擬音訊信號處理
    print("\n模擬1D卷積（音訊處理）：")
    audio_signal = torch.randn(1, 1, 1000)  # 模擬1000個樣本的音訊
    
    # 1D卷積層，用於音訊處理
    conv1d_stride1 = nn.Conv1d(1, 1, kernel_size=3, stride=1, padding=1)
    conv1d_stride2 = nn.Conv1d(1, 1, kernel_size=3, stride=2, padding=1)
    
    output_stride1 = conv1d_stride1(audio_signal)
    output_stride2 = conv1d_stride2(audio_signal)
    
    print(f"原始音訊長度: {audio_signal.shape[2]}")
    print(f"步幅1輸出長度: {output_stride1.shape[2]}")
    print(f"步幅2輸出長度: {output_stride2.shape[2]}")
    print(f"步幅2的壓縮比: {audio_signal.shape[2] / output_stride2.shape[2]:.2f}")
    print()


def exercise_4():
    """
    練習題4：步幅大於1的計算優勢
    """
    print("=== 練習題4：大步幅的計算優勢 ===")
    
    print("步幅大於1的計算優勢：")
    print("1. 降低計算複雜度：減少輸出特徵圖的尺寸")
    print("2. 減少記憶體使用：更小的特徵圖需要更少記憶體")
    print("3. 增大感受野：每層能夠「看到」更大範圍的輸入")
    print("4. 避免過擬合：相當於正則化效果")
    print("5. 快速降採樣：替代池化層的作用")
    
    # 計算複雜度比較
    print("\n計算複雜度比較：")
    input_size = 224  # 常見的輸入圖像尺寸
    kernel_size = 3
    
    # 步幅1的輸出尺寸（假設適當填充）
    output_stride1 = input_size
    # 步幅2的輸出尺寸
    output_stride2 = input_size // 2
    
    # 計算所需的操作數（簡化計算）
    ops_stride1 = output_stride1 ** 2 * kernel_size ** 2
    ops_stride2 = output_stride2 ** 2 * kernel_size ** 2
    
    print(f"輸入尺寸: {input_size} × {input_size}")
    print(f"步幅1輸出尺寸: {output_stride1} × {output_stride1}")
    print(f"步幅2輸出尺寸: {output_stride2} × {output_stride2}")
    print(f"步幅1計算量: {ops_stride1:,} 次乘法")
    print(f"步幅2計算量: {ops_stride2:,} 次乘法")
    print(f"計算量減少比例: {ops_stride1 / ops_stride2:.2f}x")
    
    # 記憶體使用比較
    memory_stride1 = output_stride1 ** 2 * 4  # 假設每個元素4字節
    memory_stride2 = output_stride2 ** 2 * 4
    print(f"步幅1記憶體使用: {memory_stride1 / 1024:.1f} KB")
    print(f"步幅2記憶體使用: {memory_stride2 / 1024:.1f} KB")
    print(f"記憶體節省: {memory_stride1 / memory_stride2:.2f}x")
    print()


def demonstrate_receptive_field():
    """
    額外示範：感受野的概念
    """
    print("=== 額外內容：感受野分析 ===")
    
    print("感受野：輸出特徵圖中一個元素對應的輸入區域大小")
    
    def calculate_receptive_field(layers_info):
        """計算多層卷積的感受野"""
        rf = 1  # 初始感受野
        stride_product = 1  # 累積步幅
        
        for i, (kernel_size, stride) in enumerate(layers_info):
            rf = rf + (kernel_size - 1) * stride_product
            stride_product *= stride
            print(f"第{i+1}層後感受野: {rf}, 累積步幅: {stride_product}")
        
        return rf
    
    # 案例1：全部步幅1
    print("\n案例1：三層3x3卷積，步幅都為1")
    layers_1 = [(3, 1), (3, 1), (3, 1)]
    rf_1 = calculate_receptive_field(layers_1)
    
    # 案例2：第二層步幅2
    print("\n案例2：三層3x3卷積，第二層步幅為2")
    layers_2 = [(3, 1), (3, 2), (3, 1)]
    rf_2 = calculate_receptive_field(layers_2)
    
    print(f"\n比較：步幅2的設計可以用更少的層達到更大的感受野")
    print()


if __name__ == "__main__":
    print("深度學習 - 填充與步幅 練習")
    print("=" * 50)
    
    # 基本概念示範
    demonstrate_padding()
    demonstrate_stride()
    
    # 練習題
    exercise_1()
    exercise_2()
    exercise_3()
    exercise_4()
    
    # 額外內容
    demonstrate_receptive_field()
    
    print("練習完成！")
    print("\n重要概念總結：")
    print("1. 填充用於保持特徵圖尺寸，避免邊界資訊丟失")
    print("2. 步幅用於降採樣，減少計算量和記憶體使用")
    print("3. 適當的填充和步幅設定是CNN設計的關鍵")
    print("4. 理解輸出尺寸計算公式對網路設計很重要")
