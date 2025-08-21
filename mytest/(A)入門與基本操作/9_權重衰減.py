"""
權重衰減（Weight Decay）範例程式
=====================================

這個程式展示了權重衰減（L2正則化）在防止過拟合中的作用。
我們將比較有無權重衰減的線性回歸模型在高維數據上的表現。

權重衰減的核心概念：
1. 在損失函數中加入權重的L2範數作為懲罰項
2. 懲罰項為：λ/2 * ||w||²，其中λ是正則化參數
3. 這會導致權重在訓練過程中逐漸"衰減"到較小的值
4. 有助於防止模型過拟合，提高泛化能力

若注意到模型出現過擬合或欠擬合的情況時，可以通過調整優化器中的權重衰減參數來進行控制。
1. 過擬合：
   如果模型在訓練集上的表現很好，但在測試集上的表現較差（測試損失較高），這表明模型過擬合。
   此時，可以增加權重衰減參數的值（例如，從 0 增加到 0.01 或更高），以限制權重的大小，從而提高模型的泛化能力。
2. 欠擬合：
   如果模型在訓練集和測試集上的表現都不好（損失都較高），這表明模型欠擬合。此時，權重衰減可能設置得過大，
   導致模型的學習能力不足。可以嘗試減小權重衰減參數的值（例如，從 0.1 減小到 0.001），以允許模型學習更多的特徵。
   
權重衰減參數和學習率是兩個不同的超參數，分別用於控制模型的正則化和學習過程
1. 權重衰減參數（λ）控制模型的正則化強度，防止過擬合。
   - 增大權重衰減參數（λ）會使模型的權重更小，從而減少過擬合，但可能導致欠擬合。
   - 減小權重衰減參數會減弱正則化效果，可能導致過擬合。
   - 權重衰減主要用於控制模型的複雜度，特別是在高維數據或小樣本數據中，幫助提高模型的泛化能力。
2. 學習率(η)控制模型在每次參數更新時的步長大小，影響模型的收斂速度和穩定性
   - 學習率過大可能導致模型在訓練過程中發散，無法收斂。
   - 學習率過小會使模型收斂速度變慢，甚至陷入局部最優。
   - 學習率是優化器的核心參數，用於控制模型的訓練過程，適當的學習率可以加速收斂並提高模型性能。
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.font_manager as fm
import platform

# 設置隨機種子以確保結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 設置matplotlib中文字體
def setup_chinese_font():
    """設置matplotlib的中文字體"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系統的中文字體
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong']
    elif system == "Darwin":  # macOS
        # macOS系統的中文字體
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'STHeiti']
    else:  # Linux
        # Linux系統的中文字體
        chinese_fonts = ['WenQuanYi Micro Hei', 'SimHei', 'DejaVu Sans']
    
    # 設置中文字體
    plt.rcParams['font.sans-serif'] = chinese_fonts + plt.rcParams['font.sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    print(f"已設置中文字體支持")

# 初始化字體設置
setup_chinese_font()

class WeightDecayDemo:
    """權重衰減演示類"""
    
    def __init__(self, n_train=20, n_test=100, num_inputs=200, batch_size=5):
        """
        初始化參數
        
        Args:
            n_train: 訓練樣本數量（故意設置得很小以容易產生過拟合）
            n_test: 測試樣本數量
            num_inputs: 輸入特徵維度（高維度容易導致過拟合）
            batch_size: 批次大小
        """
        self.n_train = n_train
        self.n_test = n_test
        self.num_inputs = num_inputs
        self.batch_size = batch_size
        
        # 真實的權重和偏置（用於生成數據）
        # 所有權重都設為0.01，偏置為0.05
        self.true_w = torch.ones((num_inputs, 1)) * 0.01
        self.true_b = 0.05
        
        # 生成數據
        self.train_data, self.test_data = self._generate_data()
        self.train_loader, self.test_loader = self._create_data_loaders()
    
    def _generate_data(self):
        """
        生成合成數據
        
        數據生成公式：y = 0.05 + Σ(0.01 * x_i) + ε
        其中 ε ~ N(0, 0.01²) 是高斯噪聲
        
        Returns:
            train_data: 訓練數據 (X, y)
            test_data: 測試數據 (X, y)
        """
        print("正在生成合成數據...")
        print(f"數據特徵：維度={self.num_inputs}, 訓練樣本={self.n_train}, 測試樣本={self.n_test}")
        
        # 生成訓練數據
        X_train = torch.normal(0, 1, (self.n_train, self.num_inputs))
        noise_train = torch.normal(0, 0.01, (self.n_train, 1))
        y_train = torch.mm(X_train, self.true_w) + self.true_b + noise_train
        
        # 生成測試數據
        X_test = torch.normal(0, 1, (self.n_test, self.num_inputs))
        noise_test = torch.normal(0, 0.01, (self.n_test, 1))
        y_test = torch.mm(X_test, self.true_w) + self.true_b + noise_test
        
        return (X_train, y_train), (X_test, y_test)
    
    def _create_data_loaders(self):
        """創建數據加載器"""
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader

class LinearRegressionFromScratch:
    """從零開始實現帶權重衰減的線性回歸"""
    
    def __init__(self, num_inputs):
        """
        初始化模型參數
        
        Args:
            num_inputs: 輸入特徵數量
        """
        self.num_inputs = num_inputs
        self.w = torch.normal(0, 0.01, size=(num_inputs, 1), requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
    
    def forward(self, X):
        """前向傳播：計算預測值"""
        return torch.mm(X, self.w) + self.b
    
    def l2_penalty(self):
        """
        計算L2懲罰項（權重的平方和）
        
        Returns:
            L2懲罰值：||w||²/2
        """
        return torch.sum(self.w.pow(2)) / 2
    
    def squared_loss(self, y_hat, y):
        """
        計算平方損失
        
        Args:
            y_hat: 預測值
            y: 真實值
            
        Returns:
            損失值
        """
        return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
    
    def train(self, train_loader, test_loader, lambd=0, lr=0.003, num_epochs=100):
        """
        訓練模型
        
        Args:
            train_loader: 訓練數據加載器
            test_loader: 測試數據加載器
            lambd: 權重衰減係數（λ）
            lr: 學習率
            num_epochs: 訓練輪數
            
        Returns:
            train_losses: 訓練損失記錄
            test_losses: 測試損失記錄
        """
        print(f"\n開始訓練（權重衰減係數 λ = {lambd}）...")
        
        train_losses = []
        test_losses = []
        
        for epoch in range(num_epochs):
            # 訓練階段
            total_train_loss = 0
            num_batches = 0
            
            for X, y in train_loader:
                # 前向傳播
                y_hat = self.forward(X)
                
                # 計算損失：原始損失 + L2懲罰項
                loss = self.squared_loss(y_hat, y).mean()
                if lambd > 0:
                    loss = loss + lambd * self.l2_penalty()
                
                # 反向傳播
                loss.backward()
                
                # 手動實現梯度下降
                with torch.no_grad():
                    # 更新權重：包含權重衰減項
                    if lambd > 0:
                        self.w -= lr * (self.w.grad + lambd * self.w)
                    else:
                        self.w -= lr * self.w.grad
                    
                    # 更新偏置（通常不對偏置應用權重衰減）
                    self.b -= lr * self.b.grad
                    
                    # 清零梯度
                    self.w.grad.zero_()
                    self.b.grad.zero_()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            # 計算測試損失
            test_loss = self._evaluate(test_loader)
            
            # 記錄損失
            avg_train_loss = total_train_loss / num_batches
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            
            # 每20輪輸出一次
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"訓練損失: {avg_train_loss:.6f}, "
                      f"測試損失: {test_loss:.6f}, "
                      f"權重L2範數: {torch.norm(self.w).item():.6f}")
        
        print(f"最終權重L2範數: {torch.norm(self.w).item():.6f}")
        return train_losses, test_losses
    
    def _evaluate(self, data_loader):
        """評估模型在數據集上的損失"""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in data_loader:
                y_hat = self.forward(X)
                loss = self.squared_loss(y_hat, y).mean()
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

class LinearRegressionPyTorch:
    """使用PyTorch內建功能實現帶權重衰減的線性回歸"""
    
    def __init__(self, num_inputs):
        """初始化模型"""
        self.model = nn.Linear(num_inputs, 1)
        # 使用正態分佈初始化參數
        nn.init.normal_(self.model.weight, std=0.01)
        nn.init.zeros_(self.model.bias)
    
    def train(self, train_loader, test_loader, weight_decay=0, lr=0.003, num_epochs=100):
        """
        訓練模型
        
        Args:
            train_loader: 訓練數據加載器
            test_loader: 測試數據加載器
            weight_decay: 權重衰減係數
            lr: 學習率
            num_epochs: 訓練輪數
        """
        print(f"\n使用PyTorch內建功能訓練（權重衰減係數 = {weight_decay}）...")
        
        # 設置損失函數
        criterion = nn.MSELoss()
        
        # 設置優化器，只對權重應用權重衰減，偏置不衰減
        optimizer = torch.optim.SGD([
            {'params': self.model.weight, 'weight_decay': weight_decay},
            {'params': self.model.bias, 'weight_decay': 0}
        ], lr=lr)
        
        train_losses = []
        test_losses = []
        
        for epoch in range(num_epochs):
            # 訓練階段
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            for X, y in train_loader:
                optimizer.zero_grad()
                y_hat = self.model(X)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                num_batches += 1
            
            # 評估階段
            self.model.eval()
            test_loss = self._evaluate(test_loader, criterion)
            
            # 記錄損失
            avg_train_loss = total_train_loss / num_batches
            train_losses.append(avg_train_loss)
            test_losses.append(test_loss)
            
            # 每20輪輸出一次
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"訓練損失: {avg_train_loss:.6f}, "
                      f"測試損失: {test_loss:.6f}, "
                      f"權重L2範數: {self.model.weight.norm().item():.6f}")
        
        print(f"最終權重L2範數: {self.model.weight.norm().item():.6f}")
        return train_losses, test_losses
    
    def _evaluate(self, data_loader, criterion):
        """評估模型"""
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for X, y in data_loader:
                y_hat = self.model(X)
                loss = criterion(y_hat, y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches

def plot_results(results_dict, title="權重衰減效果比較"):
    """
    繪製訓練和測試損失的比較圖
    
    Args:
        results_dict: 包含不同λ值結果的字典
        title: 圖表標題
    """
    plt.figure(figsize=(15, 5))
    
    # 創建子圖
    for i, (lambda_val, (train_losses, test_losses)) in enumerate(results_dict.items()):
        plt.subplot(1, len(results_dict), i + 1)
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='訓練損失', linewidth=2)
        plt.plot(epochs, test_losses, 'r-', label='測試損失', linewidth=2)
        
        plt.xlabel('訓練輪數')
        plt.ylabel('損失')
        plt.title(f'λ = {lambda_val}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用對數尺度更好地顯示差異
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def main():
    """主函數：演示權重衰減的效果"""
    print("=" * 60)
    print("權重衰減（Weight Decay）演示程式")
    print("=" * 60)
    
    # 創建數據
    demo = WeightDecayDemo(n_train=20, n_test=100, num_inputs=200, batch_size=5)
    
    print("\n實驗設置：")
    print(f"- 訓練樣本數: {demo.n_train} (故意設置很小以容易過拟合)")
    print(f"- 測試樣本數: {demo.n_test}")
    print(f"- 特徵維度: {demo.num_inputs} (高維度容易過拟合)")
    print(f"- 真實權重: 全部為 0.01")
    print(f"- 真實偏置: 0.05")
    print(f"- 噪聲: N(0, 0.01²)")
    
    # 實驗1：從零開始實現，比較不同的λ值
    print("\n" + "=" * 40)
    print("實驗1：從零開始實現權重衰減")
    print("=" * 40)
    
    lambda_values = [0, 1, 3, 10]  # 不同的權重衰減係數
    results_scratch = {}
    
    for lambd in lambda_values:
        print(f"\n--- λ = {lambd} ---")
        model = LinearRegressionFromScratch(demo.num_inputs)
        train_losses, test_losses = model.train(
            demo.train_loader, demo.test_loader, 
            lambd=lambd, lr=0.003, num_epochs=100
        )
        results_scratch[lambd] = (train_losses, test_losses)
    
    # 繪製從零實現的結果
    plot_results(results_scratch, "從零實現：權重衰減效果比較")
    
    # 實驗2：使用PyTorch內建功能
    print("\n" + "=" * 40)
    print("實驗2：使用PyTorch內建權重衰減")
    print("=" * 40)
    
    results_pytorch = {}
    
    for wd in [0, 3]:  # 比較無權重衰減和有權重衰減
        print(f"\n--- 權重衰減係數 = {wd} ---")
        model = LinearRegressionPyTorch(demo.num_inputs)
        train_losses, test_losses = model.train(
            demo.train_loader, demo.test_loader,
            weight_decay=wd, lr=0.003, num_epochs=100
        )
        results_pytorch[wd] = (train_losses, test_losses)
    
    # 繪製PyTorch實現的結果
    plot_results(results_pytorch, "PyTorch實現：權重衰減效果比較")
    
    # 分析結果
    print("\n" + "=" * 40)
    print("結果分析")
    print("=" * 40)
    
    print("\n觀察要點：")
    print("1. 無權重衰減 (λ=0):")
    print("   - 訓練損失持續下降，但測試損失可能上升")
    print("   - 這表明模型過拟合")
    print("   - 權重的L2範數會變得很大")
    
    print("\n2. 適當的權重衰減 (λ=1~3):")
    print("   - 訓練損失略微增加，但測試損失明顯減少")
    print("   - 模型的泛化能力提高")
    print("   - 權重的L2範數保持在較小的範圍")
    
    print("\n3. 過大的權重衰減 (λ=10):")
    print("   - 可能導致權重過小，模型容量不足")
    print("   - 訓練和測試損失都可能較高")
    print("   - 這稱為欠拟合")
    
    print("\n權重衰減的數學原理：")
    print("- 原始損失函數: L(w,b) = (1/n)Σ(1/2)(w^T x^(i) + b - y^(i))²")
    print("- 加入L2懲罰: L(w,b) + (λ/2)||w||²")
    print("- 梯度更新: w ← (1-ηλ)w - η∇L")
    print("- 權重在每次更新時都會'衰減'一點")
    
    print("\n實際應用建議：")
    print("- 使用驗證集來選擇最佳的λ值")
    print("- 通常λ在0.001到0.1之間")
    print("- 對於不同的層，可以使用不同的權重衰減係數")
    print("- 偏置項通常不應用權重衰減")

if __name__ == "__main__":
    # 設置中文字體
    setup_chinese_font()
    
    main()
