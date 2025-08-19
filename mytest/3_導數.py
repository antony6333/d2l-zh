import numpy as np
import matplotlib.pyplot as plt

# 定義函數 f(x) 和切線
def f(x):
    return x**3 - 1/x

def tangent_line(x):
    # 切線的斜率為 f'(x) 在 x=1 的值，f'(x) = 3x^2 + 1/x^2
    slope = 3 * (1**2) + 1 / (1**2)  # f'(x) = 3x^2 + 1/x^2    <==== 即為f(x)的導數
    return slope * (x - 1) + f(1)  # 切線方程 y = slope * (x - 1) + f(1)

# 定義 x 範圍
x = np.linspace(0.1, 2, 500)  # 避免 x=0 以防除以零
y = f(x)

# 繪製函數和切線
plt.figure(figsize=(6, 4))
plt.plot(x, y, label=r"$f(x) = x^3 - \frac{1}{x}$", color="blue")
plt.plot(x, tangent_line(x), label="Tangent line at x=1", color="orange", linestyle="--")

# 標記 x=1 的點
plt.scatter([1], [f(1)], color="red", label="Point (1, f(1))")

# 添加標籤和圖例
plt.xlabel("x")
plt.ylabel("y")
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
plt.legend()
plt.title("Function and Tangent Line at x=1")
plt.grid()
plt.show()