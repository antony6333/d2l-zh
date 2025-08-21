# 根據教學章節 pytorch/chapter_linear-networks/linear-regression-scratch.ipynb 所作的練習
# 
# 核心概念：
# 1. 線性迴歸的手動實現：不使用高階框架，直接操作張量進行計算。
# 2. 梯度計算：使用 PyTorch 的自動微分功能計算參數的梯度。
# 3. 模型訓練流程：包括前向傳播、損失計算、反向傳播和參數更新。

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 訓練資料(輸入): 100個-1到1之間的等距數字
x_train = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 輸入特徵 (100, 1)
# 訓練資料(標籤，期望目標值)，線性公式 y = 2x + 1，加上一些雜訊
y_train = 2 * x_train + 1 + 0.2 * torch.rand(x_train.size())  # 標籤 (100, 1)

# 定義線性迴歸模型 (y = wx + b)
class LinearRegressionModel:
    def __init__(self):
        # 以需要梯度的 tensor 當參數
        self.w = torch.randn(1, 1, requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, x):
        # 使用矩陣乘法與偏差
        return x @ self.w + self.b

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        # 回傳可被 optimizer 使用的參數列表
        return [self.w, self.b]

    # 保留 train()/eval() 介面以免現有訓練程式碼需要修改
    def train(self):
        pass

    def eval(self):
        pass

model = LinearRegressionModel()

# 定義損失函數（均方誤差 MSE）
criterion = nn.MSELoss()

# 定義優化器（SGD，學習率 0.1）
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 訓練模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()

    # 推論
    outputs = model(x_train)
    # 計算損失
    loss = criterion(outputs, y_train)

    # 反向傳播與優化
    optimizer.zero_grad()
    # 計算梯度
    loss.backward()
    # 更新參數
    optimizer.step()

    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 訓練完後繪製結果
model.eval()
predicted = model(x_train).detach()

# 查看 predicted 的內容，搭配對應的 x_train 值
###for x, y_pred in zip(x_train, predicted):
###    print(f"x: {x.item():.4f}, predicted y: {y_pred.item():.4f}")

plt.scatter(x_train.numpy(), y_train.numpy(), label='原始資料')
plt.plot(x_train.numpy(), predicted.numpy(), 'r-', label='線性迴歸預測')
plt.legend()
plt.show()

# 使用訓練好的模型進行其它輸入值的預測
x2_input = torch.unsqueeze(torch.linspace(-2, 3, 20), dim=1)  # 輸入特徵 (20, 1)
y2_expected = 2 * x2_input + 1  # 標籤 (100, 1)
predicted = model(x2_input).detach()

plt.scatter(x2_input.numpy(), y2_expected.numpy(), label='測試資料')
plt.plot(x2_input.numpy(), predicted.numpy(), 'r-', label='線性迴歸預測')
plt.legend()
plt.show()