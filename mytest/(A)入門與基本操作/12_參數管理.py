# 12_參數管理.py
# 根據教學章節 parameters.ipynb 所作的練習
# 
# 核心概念：
# 1. 參數訪問 - 通過不同方式訪問神經網路的權重和偏置
# 2. 參數初始化 - 使用內建和自定義方法初始化網路參數  
# 3. 參數綁定 - 在多個層之間共享參數

import torch
from torch import nn
import numpy as np

print("=== 參數管理練習 ===\n")

# ============ 1. 基本參數訪問 ============
print("1. 基本參數訪問")
print("-" * 30)

# 創建一個簡單的多層感知機
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))

print("網路結構:")
print(net)

# 執行前向傳播
output = net(X)
print(f"輸入形狀: {X.shape}")
print(f"輸出形狀: {output.shape}")
print(f"輸出值: {output.data}")

# 訪問第二個全連接層（索引為2）的參數
print("\n第二個全連接層的參數:")
print(f"state_dict(): {net[2].state_dict()}")

# 訪問具體的權重和偏置
print(f"\n權重類型: {type(net[2].weight)}")
print(f"權重形狀: {net[2].weight.shape}")
print(f"權重值: {net[2].weight.data}")

print(f"\n偏置類型: {type(net[2].bias)}")
print(f"偏置形狀: {net[2].bias.shape}")
print(f"偏置值: {net[2].bias.data}")

# 檢查梯度（初始狀態應該是None）
print(f"\n權重梯度是否為None: {net[2].weight.grad is None}")

print("\n" + "="*50 + "\n")

# ============ 2. 一次性訪問所有參數 ============
print("2. 一次性訪問所有參數")
print("-" * 30)

# 訪問第一層的所有參數
print("第一層的參數:")
for name, param in net[0].named_parameters():
    print(f"  {name}: {param.shape}")

print("\n整個網路的參數:")
for name, param in net.named_parameters():
    print(f"  {name}: {param.shape}")

# 通過state_dict訪問特定參數
print(f"\n通過state_dict訪問偏置: {net.state_dict()['2.bias'].data}")

print("\n" + "="*50 + "\n")

# ============ 3. 嵌套塊的參數訪問 ============
print("3. 嵌套塊的參數訪問")
print("-" * 30)

# 定義塊工廠函數
def block1():
    """創建一個基本的塊，包含兩個線性層和ReLU激活"""
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())

def block2():
    """創建一個包含多個block1的複合塊"""
    net = nn.Sequential()
    for i in range(4):
        # 在這裡嵌套多個block1
        net.add_module(f'block {i}', block1())
    return net

# 創建複雜的嵌套網路
rgnet = nn.Sequential(block2(), nn.Linear(4, 1))
output = rgnet(X)

print("嵌套網路結構:")
print(rgnet)

print(f"\n嵌套網路輸出: {output.data}")

# 訪問嵌套結構中的特定參數
# rgnet[0][1][0] = 第一個主要塊 -> 第二個子塊 -> 第一層
bias_value = rgnet[0][1][0].bias.data
print(f"\n訪問嵌套參數 rgnet[0][1][0].bias: {bias_value}")

print("\n" + "="*50 + "\n")

# ============ 4. 參數初始化 ============
print("4. 參數初始化")
print("-" * 30)

# 重新創建網路用於初始化實驗
net_init = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))

print("4.1 正態分佈初始化")
def init_normal(m):
    """將線性層的權重初始化為正態分佈，偏置初始化為0"""
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)

net_init.apply(init_normal)
print(f"第一層權重前幾個值: {net_init[0].weight.data[0]}")
print(f"第一層偏置前幾個值: {net_init[0].bias.data[0]}")

print("\n4.2 常數初始化")
def init_constant(m):
    """將線性層的權重初始化為常數1，偏置初始化為0"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

net_init.apply(init_constant)
print(f"常數初始化後權重: {net_init[0].weight.data[0]}")
print(f"常數初始化後偏置: {net_init[0].bias.data[0]}")

print("\n4.3 不同層使用不同初始化方法")
def init_xavier(m):
    """Xavier均勻初始化"""
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def init_42(m):
    """初始化為常數42"""
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)

# 對不同層應用不同的初始化
net_init[0].apply(init_xavier)  # 第一層使用Xavier初始化
net_init[2].apply(init_42)      # 第三層使用常數42初始化

print(f"Xavier初始化的第一層權重: {net_init[0].weight.data[0]}")
print(f"常數42初始化的第三層權重: {net_init[2].weight.data}")

print("\n" + "="*50 + "\n")

# ============ 5. 自定義初始化 ============
print("5. 自定義初始化")
print("-" * 30)

def my_init(m):
    """
    自定義初始化方法：
    - 1/4 機率: U(5, 10)
    - 1/2 機率: 0
    - 1/4 機率: U(-10, -5)
    """
    if type(m) == nn.Linear:
        print(f"初始化層: {[name for name, param in m.named_parameters()][0]}, "
              f"形狀: {m.weight.shape}")
        
        # 先用均勻分佈 U(-10, 10) 初始化
        nn.init.uniform_(m.weight, -10, 10)
        
        # 保留絕對值 >= 5 的權重，其他設為0
        # 這樣可以得到 U(-10, -5) 和 U(5, 10) 的分佈，中間為0
        m.weight.data *= m.weight.data.abs() >= 5

# 創建新網路並應用自定義初始化
net_custom = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
net_custom.apply(my_init)

print(f"\n自定義初始化結果（前兩行）:")
print(net_custom[0].weight[:2])

print("\n5.1 直接設置參數")
# 直接修改參數值
net_custom[0].weight.data[:] += 1  # 所有權重加1
net_custom[0].weight.data[0, 0] = 42  # 設置特定位置為42
print(f"直接修改後的第一行權重: {net_custom[0].weight.data[0]}")

print("\n" + "="*50 + "\n")

# ============ 6. 參數綁定（共享參數） ============
print("6. 參數綁定（共享參數）")
print("-" * 30)

# 創建共享層
shared = nn.Linear(8, 8)

# 構建網路，其中第3層和第5層共享參數
net_shared = nn.Sequential(
    nn.Linear(4, 8), nn.ReLU(),    # 層 0, 1
    shared, nn.ReLU(),             # 層 2, 3 (shared layer)
    shared, nn.ReLU(),             # 層 4, 5 (same shared layer)
    nn.Linear(8, 1)                # 層 6
)

output_shared = net_shared(X)
print(f"共享參數網路輸出: {output_shared.data}")

# 檢查參數是否真的相同
print(f"\n第2層和第4層權重是否相同:")
print(f"前幾個權重值相等: {torch.equal(net_shared[2].weight.data[0], net_shared[4].weight.data[0])}")

# 修改一個共享層的參數，驗證另一個也會改變
original_value = net_shared[2].weight.data[0, 0].clone()
net_shared[2].weight.data[0, 0] = 100

print(f"\n修改第2層的權重為100後:")
print(f"第2層權重[0,0]: {net_shared[2].weight.data[0, 0]}")
print(f"第4層權重[0,0]: {net_shared[4].weight.data[0, 0]}")
print(f"兩層權重是否仍然相等: {torch.equal(net_shared[2].weight.data[0], net_shared[4].weight.data[0])}")

# 恢復原值
net_shared[2].weight.data[0, 0] = original_value

print("\n" + "="*50 + "\n")

# ============ 7. 練習題實作 ============
print("7. 練習題實作")
print("-" * 30)

print("7.1 練習1: 使用FancyMLP模型訪問各層參數")

# 實作類似FancyMLP的模型（從第5章模型構建部分）
class FancyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 隨機權重參數，訓練時不更新（常數參數）
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        # 使用創建的常數參數，以及nn.functional.relu函數
        X = nn.functional.relu(torch.mm(X, self.rand_weight) + 1)
        # 複用全連接層。這相當於兩個全連接層共享參數
        X = self.linear(X)
        # 控制流
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

fancy_net = FancyMLP()
X_fancy = torch.rand(2, 20)
output_fancy = fancy_net(X_fancy)

print(f"FancyMLP輸出: {output_fancy}")
print("\nFancyMLP的參數:")
for name, param in fancy_net.named_parameters():
    print(f"  {name}: {param.shape}")

print(f"\n常數權重矩陣形狀: {fancy_net.rand_weight.shape}")
print(f"線性層權重形狀: {fancy_net.linear.weight.shape}")

print("\n7.2 練習3: 訓練過程中觀察共享參數的梯度")

# 創建一個簡單的共享參數網路用於訓練
shared_layer = nn.Linear(4, 4)
train_net = nn.Sequential(
    nn.Linear(2, 4),
    shared_layer,
    nn.ReLU(),
    shared_layer,  # 共享參數
    nn.Linear(4, 1)
)

# 創建訓練數據
X_train = torch.randn(10, 2)
y_train = torch.randn(10, 1)

# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(train_net.parameters(), lr=0.01)

print("訓練前共享層參數:")
print(f"第1個共享層權重[0,0]: {train_net[1].weight.data[0, 0]}")
print(f"第2個共享層權重[0,0]: {train_net[3].weight.data[0, 0]}")

# 進行一次訓練步驟
optimizer.zero_grad()
output_train = train_net(X_train)
loss = criterion(output_train, y_train)
loss.backward()

print(f"\n反向傳播後的梯度:")
print(f"第1個共享層梯度[0,0]: {train_net[1].weight.grad[0, 0]}")
print(f"第2個共享層梯度[0,0]: {train_net[3].weight.grad[0, 0]}")
print(f"梯度是否相同: {torch.equal(train_net[1].weight.grad, train_net[3].weight.grad)}")

optimizer.step()

print(f"\n更新後參數:")
print(f"第1個共享層權重[0,0]: {train_net[1].weight.data[0, 0]}")
print(f"第2個共享層權重[0,0]: {train_net[3].weight.data[0, 0]}")
print(f"參數是否仍然相同: {torch.equal(train_net[1].weight.data, train_net[3].weight.data)}")

print("\n7.3 為什麼共享參數是個好主意？")
print("""
共享參數的優點：
1. 參數數量減少：減少模型的總參數量，降低過擬合風險
2. 記憶體效率：節省記憶體空間
3. 歸納偏置：強制模型在不同位置學習相同的特徵表示
4. 正則化效果：共享參數本身就是一種正則化技術
5. 平移不變性：在CNN中實現平移不變性
6. 計算效率：減少需要更新的參數數量，加快訓練速度

常見應用：
- CNN中的卷積核共享
- RNN中的時間步權重共享  
- Transformer中的多頭注意力
- 詞嵌入層和輸出層權重綁定
""")

print("\n=== 參數管理練習完成 ===")
