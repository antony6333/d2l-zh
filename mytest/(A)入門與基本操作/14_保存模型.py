"""
根據pytorch/chapter_deep-learning-computation/read-write.ipynb章節所作的練習

本章節核心概念：
1. 張量的保存與載入 - 使用torch.save()和torch.load()函數
2. 模型參數的保存與載入 - 使用state_dict()方法
3. 模型架構與參數的分離 - 架構需要在程式碼中定義，參數可以從檔案載入

主要技術點：
- torch.save()和torch.load()用於檔案讀寫
- 可以保存單個張量、張量列表或字典
- 模型參數透過state_dict()保存和load_state_dict()載入
- 模型恢復需要先定義架構再載入參數
"""

import torch
from torch import nn
from torch.nn import functional as F
import os

print("=== 第一部分：張量的保存與載入 ===")

# 1. 保存和載入單個張量
print("\n1. 單個張量的保存與載入")
x = torch.arange(4)
print(f"原始張量 x: {x}")

# 保存張量到檔案
torch.save(x, 'x-file')
print("已保存張量到 'x-file'")

# 從檔案載入張量
x2 = torch.load('x-file')
print(f"載入的張量 x2: {x2}")
print(f"x 和 x2 是否相等: {torch.equal(x, x2)}")

# 2. 保存和載入張量列表
print("\n2. 張量列表的保存與載入")
y = torch.zeros(4)
print(f"張量 x: {x}")
print(f"張量 y: {y}")

# 保存張量列表
torch.save([x, y], 'x-files')
print("已保存張量列表到 'x-files'")

# 載入張量列表
x2, y2 = torch.load('x-files')
print(f"載入的張量 x2: {x2}")
print(f"載入的張量 y2: {y2}")

# 3. 保存和載入張量字典
print("\n3. 張量字典的保存與載入")
mydict = {'x': x, 'y': y}
print(f"原始字典: {mydict}")

# 保存字典
torch.save(mydict, 'mydict')
print("已保存字典到 'mydict'")

# 載入字典
mydict2 = torch.load('mydict')
print(f"載入的字典: {mydict2}")

print("\n=== 第二部分：模型參數的保存與載入 ===")

# 定義多層感知機模型
class MLP(nn.Module):
    """
    簡單的多層感知機模型
    包含一個隱藏層和一個輸出層
    """
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # 隱藏層：20輸入，256輸出
        self.output = nn.Linear(256, 10)  # 輸出層：256輸入，10輸出

    def forward(self, x):
        """前向傳播"""
        return self.output(F.relu(self.hidden(x)))

# 創建模型實例並進行前向傳播
print("\n4. 創建並測試模型")
net = MLP()
X = torch.randn(size=(2, 20))  # 創建隨機輸入
Y = net(X)  # 前向傳播
print(f"輸入形狀: {X.shape}")
print(f"輸出形狀: {Y.shape}")
print(f"模型參數數量: {sum(p.numel() for p in net.parameters())}")

# 5. 保存模型參數
print("\n5. 保存模型參數")
torch.save(net.state_dict(), 'mlp.params')
print("已保存模型參數到 'mlp.params'")

# 顯示模型的狀態字典結構
print("模型參數結構:")
for name, param in net.state_dict().items():
    print(f"  {name}: {param.shape}")

# 6. 載入模型參數
print("\n6. 載入模型參數")
# 創建新的模型實例
clone = MLP()
# 載入保存的參數
clone.load_state_dict(torch.load('mlp.params'))
# 設置為評估模式
clone.eval()
print("已創建模型副本並載入參數")

# 7. 驗證模型一致性
print("\n7. 驗證模型一致性")
Y_clone = clone(X)
print(f"原始模型輸出: {Y[0][:5]}...")  # 只顯示前5個值
print(f"載入模型輸出: {Y_clone[0][:5]}...")  # 只顯示前5個值
print(f"輸出是否完全相等: {torch.equal(Y, Y_clone)}")

# 計算差異
diff = torch.abs(Y - Y_clone).max()
print(f"最大差異: {diff.item()}")

print("\n=== 第三部分：練習題解答 ===")

print("\n練習1：存儲模型參數的實際好處")
print("""
1. 訓練過程中的檢查點保存 - 防止意外中斷導致的訓練進度丟失
2. 模型版本管理 - 可以保存不同訓練階段的模型進行比較
3. 分佈式訓練 - 可以在不同設備間共享訓練好的參數
4. 增量學習 - 可以基於已訓練的模型繼續訓練
5. 實驗重現 - 確保實驗結果的可重現性
""")

print("\n練習2：複用網絡的部分層")
print("示範如何複用前兩層：")

# 創建新的網絡架構，只使用前兩層
class PartialMLP(nn.Module):
    def __init__(self, pretrained_net):
        super().__init__()
        # 複用前兩層
        self.hidden = pretrained_net.hidden
        # 添加新的輸出層
        self.new_output = nn.Linear(256, 5)  # 假設新任務只需要5個輸出
        
    def forward(self, x):
        return self.new_output(F.relu(self.hidden(x)))

# 創建部分複用的模型
partial_net = PartialMLP(net)
print(f"新模型輸出形狀: {partial_net(X).shape}")

# 也可以通過參數字典選擇性載入
print("\n通過參數字典選擇性載入：")
new_net = MLP()
state_dict = torch.load('mlp.params')
# 只載入hidden層的參數
new_net.hidden.load_state_dict({
    'weight': state_dict['hidden.weight'],
    'bias': state_dict['hidden.bias']
})
print("已選擇性載入hidden層參數")

print("\n練習3：同時保存架構和參數")
print("""
方法1：保存整個模型（包含架構）
- 使用torch.save(model, 'model.pth')
- 限制：模型類必須在載入環境中可用

方法2：使用torchscript
- 將模型轉換為可序列化的格式
- 不依賴Python環境和類定義
""")

# 示範保存整個模型
torch.save(net, 'complete_model.pth')
print("已保存完整模型到 'complete_model.pth'")

# 載入完整模型
try:
    # PyTorch 2.6+ 將 torch.load 的默認 weights_only 參數設為 True，
    # 若要載入整個模型物件，需顯式指定 weights_only=False。
    loaded_net = torch.load('complete_model.pth', weights_only=False)
except TypeError:
    # 若使用的 PyTorch 版本較舊，不支援 weights_only 參數，回退至不帶參數的呼叫
    loaded_net = torch.load('complete_model.pth')
loaded_net.eval()
print("已載入完整模型")

# 驗證載入的完整模型
Y_complete = loaded_net(X)
print(f"完整模型載入是否成功: {torch.equal(Y, Y_complete)}")

print("\n=== 清理生成的檔案 ===")
# 清理生成的檔案
files_to_remove = ['x-file', 'x-files', 'mydict', 'mlp.params', 'complete_model.pth']
for file in files_to_remove:
    if os.path.exists(file):
        os.remove(file)
        print(f"已刪除 {file}")

print("\n程式執行完成！")