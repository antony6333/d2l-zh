"""
深度學習計算 - 層和塊 (Layers and Blocks)
============================================

本章節學習如何構建自定義的神經網络塊，包括：
1. 理解塊的概念和基本功能
2. 自定義MLP塊
3. 順序塊的實現
4. 在前向傳播中執行代碼
5. 混合搭配各種組合塊

作者：深度學習課程實作練習
日期：2025年8月

"""

import torch
from torch import nn
from torch.nn import functional as F

print("=" * 80)
print("深度學習計算 - 層和塊 (Layers and Blocks)")
print("=" * 80)

# ============================================================================
# 1. 回顧多層感知機的實現
# ============================================================================
print("\n1. 回顧多層感知機的實現")
print("-" * 50)

# 使用nn.Sequential構建簡單的多層感知機
# 包含一個256單元的隱藏層（ReLU激活）和一個10單元的輸出層
net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

# 創建隨機輸入數據：2個樣本，每個樣本20個特徵
X = torch.rand(2, 20)
print(f"輸入張量形狀: {X.shape}")

# 進行前向傳播
output = net(X)
print(f"輸出張量形狀: {output.shape}")
print(f"輸出結果:\n{output}")

# ============================================================================
# 2. 自定義MLP塊
# ============================================================================
print("\n\n2. 自定義MLP塊")
print("-" * 50)

class MLP(nn.Module):
    """
    自定義多層感知機(MLP)塊
    
    這個類展示了如何從零開始實現一個神經網絡塊，包括：
    - 構造函數中定義層
    - 前向傳播函數中定義計算流程
    """
    
    def __init__(self):
        """
        構造函數：初始化MLP的各個層
        
        必須調用父類的__init__()來確保正確的初始化
        """
        # 調用父類Module的構造函數
        super().__init__()
        
        # 定義隱藏層：20個輸入特徵 -> 256個隱藏單元
        self.hidden = nn.Linear(20, 256)
        
        # 定義輸出層：256個隱藏單元 -> 10個輸出
        self.out = nn.Linear(256, 10)
    
    def forward(self, X):
        """
        前向傳播函數：定義數據如何在網絡中流動
        
        Args:
            X: 輸入張量，形狀為 (batch_size, input_features)
            
        Returns:
            輸出張量，形狀為 (batch_size, output_features)
        """
        # 通過隱藏層並應用ReLU激活函數
        hidden_output = F.relu(self.hidden(X))
        
        # 通過輸出層（不使用激活函數）
        return self.out(hidden_output)

# 測試自定義MLP
print("測試自定義MLP:")
custom_net = MLP()
custom_output = custom_net(X)
print(f"自定義MLP輸出形狀: {custom_output.shape}")
print(f"自定義MLP輸出:\n{custom_output}")

# ============================================================================
# 3. 順序塊的自定義實現
# ============================================================================
print("\n\n3. 順序塊的自定義實現")
print("-" * 50)

class MySequential(nn.Module):
    """
    自定義順序塊，模仿nn.Sequential的功能
    
    這個類展示了如何實現一個容器塊，可以按順序組合多個子塊
    """
    
    def __init__(self, *args):
        """
        構造函數：接受可變數量的模塊參數
        
        Args:
            *args: 可變數量的nn.Module實例
        """
        super().__init__()
        
        # 將每個模塊添加到_modules有序字典中
        # _modules是nn.Module的特殊屬性，用於自動管理參數
        for idx, module in enumerate(args):
            # 使用字符串索引作為鍵名
            self._modules[str(idx)] = module
    
    def forward(self, X):
        """
        前向傳播：按添加順序依次執行每個模塊
        
        Args:
            X: 輸入張量
            
        Returns:
            最終輸出張量
        """
        # OrderedDict保證按照添加順序遍歷
        for block in self._modules.values():
            X = block(X)  # 每個塊的輸出作為下一個塊的輸入
        return X

# 測試自定義順序塊
print("測試自定義順序塊:")
my_net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
my_output = my_net(X)
print(f"自定義順序塊輸出形狀: {my_output.shape}")
print(f"自定義順序塊輸出:\n{my_output}")

# ============================================================================
# 4. 在前向傳播函數中執行代碼
# ============================================================================
print("\n\n4. 在前向傳播函數中執行代碼")
print("-" * 50)

class FixedHiddenMLP(nn.Module):
    """
    具有固定參數和控制流的MLP
    
    這個類展示了如何在神經網絡中：
    - 使用常數參數（不參與梯度計算）
    - 在前向傳播中使用控制流
    - 重用層
    """
    
    def __init__(self):
        """構造函數：初始化固定權重和可訓練層"""
        super().__init__()
        
        # 創建固定的隨機權重，不參與梯度計算
        # requires_grad=False 表示這個參數不會被優化器更新
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        
        # 創建可訓練的線性層
        self.linear = nn.Linear(20, 20)
    
    def forward(self, X):
        """
        前向傳播：包含固定參數運算和控制流
        
        Args:
            X: 輸入張量
            
        Returns:
            標量輸出（所有元素的和）
        """
        # 第一次通過線性層
        X = self.linear(X)
        
        # 使用固定的隨機權重進行矩陣乘法，並加上常數1
        # torch.mm: 矩陣乘法
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        
        # 重用同一個線性層（參數共享）
        X = self.linear(X)
        
        # 控制流：當L1範數大於1時，將X除以2
        while X.abs().sum() > 1:
            X /= 2
        
        # 返回所有元素的和（標量）
        return X.sum()

# 測試固定隱藏層MLP
print("測試固定隱藏層MLP:")
fixed_net = FixedHiddenMLP()
fixed_output = fixed_net(X)
print(f"固定隱藏層MLP輸出: {fixed_output}")
print(f"輸出類型: {type(fixed_output)}")

# ============================================================================
# 5. 嵌套塊
# ============================================================================
print("\n\n5. 嵌套塊")
print("-" * 50)

class NestMLP(nn.Module):
    """
    嵌套MLP：在一個塊中包含另一個順序塊
    
    這展示了塊的組合性：塊可以包含其他塊
    """
    
    def __init__(self):
        """構造函數：創建嵌套的網絡結構"""
        super().__init__()
        
        # 創建一個順序塊作為子網絡
        self.net = nn.Sequential(
            nn.Linear(20, 64), nn.ReLU(),  # 第一層：20->64
            nn.Linear(64, 32), nn.ReLU()   # 第二層：64->32
        )
        
        # 創建最終的線性層
        self.linear = nn.Linear(32, 16)
    
    def forward(self, X):
        """
        前向傳播：先通過子網絡，再通過最終線性層
        
        Args:
            X: 輸入張量
            
        Returns:
            輸出張量
        """
        return self.linear(self.net(X))

# 創建複雜的組合網絡（嵌套多個不同的塊）
print("創建複雜的組合網絡:")
chimera = nn.Sequential(
    NestMLP(),          # 嵌套MLP：20->16
    nn.Linear(16, 20),  # 線性層：16->20
    FixedHiddenMLP()    # 固定隱藏層MLP：20->標量
)

chimera_output = chimera(X)
print(f"組合網絡輸出: {chimera_output}")

# ============================================================================
# 6. 練習題實現
# ============================================================================
print("\n\n6. 練習題實現")
print("-" * 50)

# 練習1：探討將MySequential中的_modules改為Python列表會有什麼問題
print("練習1：探討MySequential使用列表 vs 字典的差異")

class MySequentialWithList(nn.Module):
    """
    使用Python列表而非_modules字典的順序塊
    
    這會導致參數管理問題！
    """
    
    def __init__(self, *args):
        super().__init__()
        # 使用普通Python列表存儲模塊
        self.layers = list(args)  # 問題：PyTorch無法自動追蹤這些參數！
    
    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

# 比較參數數量
print("使用_modules的網絡參數數量:")
correct_net = MySequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
print(f"參數數量: {sum(p.numel() for p in correct_net.parameters())}")

print("\n使用列表的網絡參數數量:")
broken_net = MySequentialWithList(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 10))
print(f"參數數量: {sum(p.numel() for p in broken_net.parameters())}")
print("問題：使用列表時，PyTorch無法自動註冊和管理子模塊的參數！")

# 練習2：實現平行塊
print("\n練習2：實現平行塊")

class ParallelBlock(nn.Module):
    """
    平行塊：接受兩個網絡，返回它們輸出的串聯
    
    這展示了如何組合多個並行的網絡分支
    """
    
    def __init__(self, net1, net2):
        """
        構造函數
        
        Args:
            net1: 第一個網絡
            net2: 第二個網絡
        """
        super().__init__()
        self.net1 = net1
        self.net2 = net2
    
    def forward(self, X):
        """
        前向傳播：並行執行兩個網絡，然後串聯輸出
        
        Args:
            X: 輸入張量
            
        Returns:
            串聯後的輸出張量
        """
        # 分別通過兩個網絡
        out1 = self.net1(X)
        out2 = self.net2(X)
        
        # 在最後一個維度上串聯輸出
        return torch.cat((out1, out2), dim=1)

# 測試平行塊
print("測試平行塊:")
net1 = nn.Sequential(nn.Linear(20, 32), nn.ReLU())
net2 = nn.Sequential(nn.Linear(20, 16), nn.ReLU())
parallel_net = ParallelBlock(net1, net2)

parallel_output = parallel_net(X)
print(f"平行塊輸出形狀: {parallel_output.shape}")  # 應該是 [2, 48] (32+16)
print(f"net1輸出形狀: {net1(X).shape}")
print(f"net2輸出形狀: {net2(X).shape}")

# 練習3：生成同一塊的多個實例
print("\n練習3：生成同一塊的多個實例")

def create_multiple_instances(block_class, num_instances, *args, **kwargs):
    """
    創建同一塊的多個實例並組合成更大的網絡
    
    Args:
        block_class: 要實例化的塊類
        num_instances: 實例數量
        *args, **kwargs: 傳遞給塊構造函數的參數
        
    Returns:
        包含多個實例的順序網絡
    """
    blocks = []
    for i in range(num_instances):
        # 創建塊的新實例
        block = block_class(*args, **kwargs)
        blocks.append(block)
    
    # 將所有實例組合成順序網絡
    return nn.Sequential(*blocks)

# 定義一個簡單的塊用於測試
class SimpleBlock(nn.Module):
    """簡單的線性變換塊"""
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, X):
        return self.relu(self.linear(X))

# 創建多個相同塊的實例
print("創建3個相同塊的實例:")
multi_block_net = create_multiple_instances(SimpleBlock, 3, 20, 20)

# 添加最終輸出層
final_net = nn.Sequential(
    multi_block_net,
    nn.Linear(20, 10)
)

multi_output = final_net(X)
print(f"多實例網絡輸出形狀: {multi_output.shape}")
print(f"網絡結構:\n{final_net}")

# ============================================================================
# 7. 總結和效率討論
# ============================================================================
print("\n\n7. 總結")
print("-" * 50)

print("""
本章節學習要點：

1. 塊的概念：
   - 塊是PyTorch中組織神經網絡的基本單位
   - 可以是單個層、多個層的組合，或完整的模型
   - 必須繼承nn.Module並實現forward方法

2. 塊的關鍵功能：
   - 接受輸入並生成輸出
   - 存儲和管理參數
   - 支持自動梯度計算
   - 可以嵌套和組合

3. 實現技巧：
   - 使用_modules字典管理子模塊（不要用普通列表）
   - 在__init__中調用super().__init__()
   - 可以在forward中使用控制流和任意計算
   - 支持參數共享和固定參數

4. 設計模式：
   - 順序塊：按順序執行多個塊
   - 平行塊：並行執行多個分支
   - 嵌套塊：塊中包含其他塊
   - 動態塊：根據輸入動態改變計算流程

5. 效率考量：
   - PyTorch會優化計算圖
   - 避免過度的Python控制流
   - 使用適當的批次大小
""")

print("\n實驗完成！")
print("=" * 80)
