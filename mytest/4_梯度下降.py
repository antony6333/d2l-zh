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

# 梯度下降
for i in range(num_iterations):
    gradient = grad_f(x)
    x = x - eta * gradient
    print(f"Iteration {i+1}: x = {x:.4f}, f(x) = {f(x):.4f}")