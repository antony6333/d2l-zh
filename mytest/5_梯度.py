import torch

# 創建一個張量並啟用梯度計算
x = torch.tensor([2.0, 3.0], requires_grad=True)

# 定義一個簡單的函數 y = x1^2 + x2^3
y = x[0]**2 + x[1]**3

# 計算 y 對 x 的梯度
y.backward()

# 輸出梯度 (x.grad：存儲梯度，對應於 y 對 x 的偏導數)
print("x 的梯度:", x.grad)
