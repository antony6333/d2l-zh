import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = 'Microsoft JhengHei'  # 或 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 產生線性資料 y = 2x + 1，加上一些雜訊
x_train = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 輸入特徵 (100, 1)
y_train = 2 * x_train + 1 + 0.2 * torch.rand(x_train.size())  # 標籤 (100, 1)

# 定義線性迴歸模型 (y = wx + b)
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # 輸入維度1，輸出維度1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 定義損失函數（均方誤差 MSE）
criterion = nn.MSELoss()

# 定義優化器（SGD，學習率 0.1）
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 訓練模型
num_epochs = 20
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

    if (epoch + 1) % 10 == 0:
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