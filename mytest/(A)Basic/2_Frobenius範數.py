# 根據教學章節 pytorch/chapter_preliminaries/norms.ipynb 所作的練習
# 
# 核心概念：
# 1. Frobenius 範數：矩陣中所有元素平方和的平方根，表示矩陣的整體大小。
# 2. 計算公式：$$ \|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n a_{ij}^2} $$。
# 3. 幾何意義：範數反映了矩陣元素的分佈特性和整體大小。

import torch
import matplotlib.pyplot as plt

'''
從繪出的圖形中，我們可以通過以下方式理解矩陣的範圍（Frobenius 範數）為 6：

1. 矩陣的元素值
圖形中每個圓點代表矩陣的一個元素，torch.ones((4, 9)) 的所有元素值都是 1.0。
圓點的大小與元素的絕對值成正比，因此所有圓點的大小相同，並且在圖形中標註的數值也都是 1.0。
2. Frobenius 範數的計算
Frobenius 範數的公式為： $$ |\mathbf{A}|F = \sqrt{\sum{i=1}^m \sum_{j=1}^n a_{ij}^2} $$

矩陣 A 的形狀為 (4, 9)，共有 $4 \times 9 = 36$ 個元素。
每個元素的值為 1.0，平方後仍為 1.0。
所有元素的平方和為 $36$，取平方根後得到： $$ \sqrt{36} = 6 $$
3. 圖形與範數的關係
圖形的標題中顯示了 Frobenius 範數為 6.0，這是矩陣所有元素的「整體大小」。
圓點的大小和數值顯示矩陣的元素值，這些值的平方和決定了範數的大小。
4. 範數的意義
Frobenius 範數表示矩陣中所有元素的「整體大小」或「範圍」。
在這個例子中，因為所有元素的值相同，範數的大小直接反映了矩陣的元素數量（平方和的平方根）。

總結
從圖形中可以看到：

矩陣的所有元素值為 1.0，且分佈均勻。
Frobenius 範數為 6.0，表示矩陣的整體大小，這是由所有元素的平方和（36）取平方根得到的結果。
'''

# 定義矩陣
A = torch.ones((4, 9))

# 計算 Frobenius 範數
frobenius_norm = torch.norm(A).item()

# 繪製矩陣元素的範圍
plt.figure(figsize=(9, 4))
for i in range(A.shape[0]):  # 遍歷行
    for j in range(A.shape[1]):  # 遍歷列
        # 使用圓點表示矩陣元素，大小與元素絕對值成正比
        plt.scatter(j, i, s=abs(A[i, j].item()) * 100, c='blue', alpha=0.6)
        # 在圓點上顯示元素的數值
        plt.text(j, i, f"{A[i, j].item():.1f}", ha='center', va='center', color='white')

# 添加標題和範圍說明
plt.title(f"Matrix Visualization (Frobenius Norm: {frobenius_norm:.2f})")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.xticks(range(A.shape[1]))
plt.yticks(range(A.shape[0]))
plt.grid(True)
plt.gca().invert_yaxis()  # 翻轉 y 軸以符合矩陣的視覺化習慣
plt.show()