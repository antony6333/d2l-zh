import numpy as np

# 定義函數及其梯度
def f(x):
    return x**2

def grad_f(x):
    return 2 * x

# 初始化參數
x = 10  # 初始值
eta = 0.1  # 學習率
num_iterations = 50  # 迭代次數

'''
梯度下降
梯度下降的目的是最小化損失函數，即找到函數的最小值。
在機器學習和深度學習中，損失函數通常用來衡量模型預測值與真實值之間的差距。
通過梯度下降，我們可以調整模型的參數，使損失函數的值逐漸減小，從而提升模型的性能。
'''
for i in range(num_iterations):
    gradient = grad_f(x)
    x = x - eta * gradient
    print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")