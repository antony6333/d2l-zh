"""
暫退法（Dropout）範例程式
=========================

這個程式展示了暫退法在深度學習中防止過拟合的作用。
我們將實現從零開始和簡潔版本的暫退法，並比較有無暫退法的模型表現。

暫退法的核心概念：
1. 在訓練過程中，以一定概率p隨機將某些神經元的輸出設為0
2. 剩餘的神經元輸出需要除以(1-p)來保持期望值不變
3. 測試時不使用暫退法，使用完整的網絡
4. 能夠有效防止神經元之間的共適應性，提高泛化能力

暫退法的優點：
- 防止過拟合
- 提高模型的泛化能力
- 破壞神經元之間的依賴關係
- 模擬集成學習的效果

注意事項：
- 不同層可以設置不同的暫退概率
- 靠近輸入層的層通常使用較低的暫退概率
- 測試時必須關閉暫退法
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.font_manager as fm
import platform
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

# 設置隨機種子以確保結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 設置matplotlib中文字體
def setup_chinese_font():
    """設置matplotlib的中文字體"""
    system = platform.system()
    
    if system == "Windows":
        # Windows系統中文字體
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'SimSun']
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                break
            except:
                continue
    elif system == "Darwin":  # macOS
        # macOS系統中文字體
        chinese_fonts = ['Heiti TC', 'Arial Unicode MS', 'PingFang TC', 'STHeiti']
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                break
            except:
                continue
    else:  # Linux
        # Linux系統中文字體
        chinese_fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'SimHei']
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]
                break
            except:
                continue
    
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 初始化中文字體
setup_chinese_font()

def dropout_layer(X, dropout):
    """
    實現暫退法函數
    
    參數:
        X: 輸入張量
        dropout: 暫退概率，範圍[0, 1]
    
    返回:
        應用暫退法後的張量
    """
    assert 0 <= dropout <= 1
    
    # 在本情況中，所有元素都被丟弃
    if dropout == 1:
        return torch.zeros_like(X)
    
    # 在本情況中，所有元素都被保留
    if dropout == 0:
        return X
    
    # 生成隨機遮罩，保留概率為(1-dropout)的元素
    mask = (torch.rand(X.shape) > dropout).float()
    
    # 對保留的元素進行縮放，保持期望值不變
    return mask * X / (1.0 - dropout)

def test_dropout_layer():
    """測試暫退法函數的功能"""
    print("=" * 50)
    print("測試暫退法函數")
    print("=" * 50)
    
    # 創建測試數據
    X = torch.arange(16, dtype=torch.float32).reshape((2, 8))
    print("原始輸入：")
    print(X)
    print()
    
    print("暫退概率為0（保留所有元素）：")
    print(dropout_layer(X, 0.))
    print()
    
    print("暫退概率為0.5（隨機丟棄50%元素）：")
    print(dropout_layer(X, 0.5))
    print()
    
    print("暫退概率為1（丟棄所有元素）：")
    print(dropout_layer(X, 1.))
    print()

class DropoutMLP(nn.Module):
    """
    從零開始實現的帶暫退法的多層感知機
    """
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, 
                 dropout1=0.2, dropout2=0.5, is_training=True):
        super(DropoutMLP, self).__init__()
        self.num_inputs = num_inputs
        self.is_training = is_training
        self.dropout1 = dropout1  # 第一個隱藏層的暫退概率
        self.dropout2 = dropout2  # 第二個隱藏層的暫退概率
        
        # 定義網絡層
        self.lin1 = nn.Linear(num_inputs, num_hiddens1)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs)
        self.relu = nn.ReLU()
        
    def forward(self, X):
        # 第一個隱藏層
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs))))
        
        # 只有在訓練模式下才使用暫退法
        if self.is_training:
            H1 = dropout_layer(H1, self.dropout1)
        
        # 第二個隱藏層
        H2 = self.relu(self.lin2(H1))
        
        # 只有在訓練模式下才使用暫退法
        if self.is_training:
            H2 = dropout_layer(H2, self.dropout2)
        
        # 輸出層（不使用暫退法）
        out = self.lin3(H2)
        return out

class SimpleDropoutMLP(nn.Module):
    """
    使用PyTorch內建Dropout的簡潔實現
    """
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, 
                 dropout1=0.2, dropout2=0.5):
        super(SimpleDropoutMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_hiddens1),
            nn.ReLU(),
            nn.Dropout(dropout1),  # 第一個暫退層
            nn.Linear(num_hiddens1, num_hiddens2),
            nn.ReLU(), 
            nn.Dropout(dropout2),  # 第二個暫退層
            nn.Linear(num_hiddens2, num_outputs)
        )
        
        # 初始化權重
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """初始化網絡權重"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # 使用Xavier初始化
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)

def load_fashion_mnist(batch_size, num_workers=0):
    """
    載入Fashion-MNIST數據集
    
    參數:
        batch_size: 批次大小
        num_workers: 工作線程數
    
    返回:
        train_loader: 訓練數據載入器
        test_loader: 測試數據載入器
    """
    # 數據轉換 - 使用更溫和的標準化
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST的均值和標準差
    ])
    
    # 載入訓練集
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 載入測試集
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 創建數據載入器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def accuracy(y_hat, y):
    """計算預測準確率"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_loader, device):
    """
    評估模型準確率
    
    參數:
        net: 神經網絡模型
        data_loader: 數據載入器
        device: 計算設備
    
    返回:
        準確率
    """
    if isinstance(net, torch.nn.Module):
        net.eval()  # 設置為評估模式
    
    metric = [0.0, 0.0]  # 正確預測數, 總預測數
    
    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            metric[0] += accuracy(net(X), y)
            metric[1] += y.numel()
    
    return metric[0] / metric[1]

def train_epoch(net, train_loader, loss_fn, optimizer, device):
    """
    訓練一個epoch
    
    參數:
        net: 神經網絡模型
        train_loader: 訓練數據載入器
        loss_fn: 損失函數
        optimizer: 優化器
        device: 計算設備
    
    返回:
        平均損失, 訓練準確率
    """
    if isinstance(net, torch.nn.Module):
        net.train()  # 設置為訓練模式
    
    metric = [0.0, 0.0, 0.0]  # 訓練損失總和, 訓練準確率總和, 樣本數
    batch_count = 0
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        
        # 前向傳播
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        
        # 如果使用了reduction='none'，需要計算平均損失
        if loss.dim() > 0:  # 檢查是否有多個損失值
            loss = loss.mean()
        
        # 檢查是否有NaN或無窮大
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"警告：檢測到異常損失值 {loss.item()}，跳過該批次")
            continue
        
        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        
        # 檢查梯度是否異常
        total_norm = 0.0
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 10.0:  # 梯度過大
            print(f"警告：梯度過大 {total_norm:.4f}，將進行裁剪")
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 更新指標
        with torch.no_grad():
            metric[0] += float(loss.item())
            metric[1] += accuracy(y_hat, y)
            metric[2] += y.numel()
        
        batch_count += 1
    
    return metric[0] / metric[2], metric[1] / metric[2]

def train_model(net, train_loader, test_loader, loss_fn, optimizer, 
                num_epochs, device, model_name="模型"):
    """
    完整的模型訓練過程
    
    參數:
        net: 神經網絡模型
        train_loader: 訓練數據載入器
        test_loader: 測試數據載入器
        loss_fn: 損失函數
        optimizer: 優化器
        num_epochs: 訓練輪數
        device: 計算設備
        model_name: 模型名稱
    
    返回:
        訓練歷史記錄
    """
    print(f"開始訓練 {model_name}")
    print("=" * 50)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # 訓練
        train_loss, train_acc = train_epoch(net, train_loader, loss_fn, optimizer, device)
        
        # 評估
        test_acc = evaluate_accuracy(net, test_loader, device)
        
        # 記錄歷史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # 打印進度
        print(f'Epoch {epoch+1:2d}/{num_epochs}: '
              f'訓練損失 {train_loss:.4f}, '
              f'訓練準確率 {train_acc:.4f}, '
              f'測試準確率 {test_acc:.4f}')
    
    print(f"{model_name} 訓練完成！")
    print("=" * 50)
    print()
    
    return history

def plot_training_history(histories, model_names):
    """
    繪製訓練歷史對比圖
    
    參數:
        histories: 各模型的訓練歷史列表
        model_names: 模型名稱列表
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 訓練損失
    axes[0].set_title('訓練損失對比')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 
                    color=colors[i % len(colors)], label=name, linewidth=2)
    axes[0].set_xlabel('訓練輪數')
    axes[0].set_ylabel('損失')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 訓練準確率
    axes[1].set_title('訓練準確率對比')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        epochs = range(1, len(history['train_acc']) + 1)
        axes[1].plot(epochs, history['train_acc'], 
                    color=colors[i % len(colors)], label=name, linewidth=2)
    axes[1].set_xlabel('訓練輪數')
    axes[1].set_ylabel('準確率')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 測試準確率
    axes[2].set_title('測試準確率對比')
    for i, (history, name) in enumerate(zip(histories, model_names)):
        epochs = range(1, len(history['test_acc']) + 1)
        axes[2].plot(epochs, history['test_acc'], 
                    color=colors[i % len(colors)], label=name, linewidth=2)
    axes[2].set_xlabel('訓練輪數')
    axes[2].set_ylabel('準確率')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_dropout_effect(histories, model_names):
    """
    分析暫退法的效果
    
    參數:
        histories: 各模型的訓練歷史列表
        model_names: 模型名稱列表
    """
    print("=" * 50)
    print("暫退法效果分析")
    print("=" * 50)
    
    for i, (history, name) in enumerate(zip(histories, model_names)):
        final_train_acc = history['train_acc'][-1]
        final_test_acc = history['test_acc'][-1]
        overfitting = final_train_acc - final_test_acc
        
        print(f"{name}:")
        print(f"  最終訓練準確率: {final_train_acc:.4f}")
        print(f"  最終測試準確率: {final_test_acc:.4f}")
        print(f"  過拟合程度: {overfitting:.4f}")
        print()
    
    print("分析結論:")
    print("1. 過拟合程度 = 訓練準確率 - 測試準確率")
    print("2. 過拟合程度越小，模型泛化能力越好")
    print("3. 暫退法通常能夠減少過拟合，提高泛化能力")
    print("=" * 50)

def demonstrate_dropout_variance():
    """
    展示暫退法對激活值方差的影響
    """
    print("=" * 50)
    print("暫退法對激活值方差的影響")
    print("=" * 50)
    
    # 創建測試數據
    X = torch.randn(1000, 256)  # 1000個樣本，256個特徵
    
    dropout_probs = [0.0, 0.2, 0.5, 0.8]
    variances = []
    
    for p in dropout_probs:
        # 多次應用暫退法並計算方差
        outputs = []
        for _ in range(100):
            output = dropout_layer(X, p)
            outputs.append(output)
        
        # 計算輸出的方差
        outputs_tensor = torch.stack(outputs)
        variance = torch.var(outputs_tensor, dim=0).mean().item()
        variances.append(variance)
        
        print(f"暫退概率 {p}: 激活值方差 = {variance:.4f}")
    
    # 繪製方差變化圖
    plt.figure(figsize=(10, 6))
    plt.plot(dropout_probs, variances, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('暫退概率')
    plt.ylabel('激活值方差')
    plt.title('暫退概率對激活值方差的影響')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n觀察結果:")
    print("1. 隨著暫退概率增加，激活值方差通常會增加")
    print("2. 這種方差增加有助於增強模型的稳健性")
    print("3. 適度的方差有助於防止過拟合")
    print("=" * 50)

def main():
    """主函數"""
    print("暫退法（Dropout）深度學習範例")
    print("=" * 50)
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    print()
    
    # 測試暫退法函數
    test_dropout_layer()
    
    # 展示暫退法對方差的影響
    demonstrate_dropout_variance()
    
    # 設置超參數
    num_inputs = 784  # Fashion-MNIST圖片大小 28x28
    num_outputs = 10  # 10個類別
    num_hiddens1 = 256  # 第一個隱藏層大小
    num_hiddens2 = 256  # 第二個隱藏層大小
    dropout1 = 0.2  # 第一層暫退概率
    dropout2 = 0.5  # 第二層暫退概率
    
    num_epochs = 10
    lr = 0.5  # 使用與notebook相同的學習率
    batch_size = 256
    
    print(f"超參數設置:")
    print(f"  隱藏層大小: {num_hiddens1}, {num_hiddens2}")
    print(f"  暫退概率: {dropout1}, {dropout2}")
    print(f"  學習率: {lr}")
    print(f"  批次大小: {batch_size}")
    print(f"  訓練輪數: {num_epochs}")
    print()
    
    # 載入數據
    print("載入Fashion-MNIST數據集...")
    train_loader, test_loader = load_fashion_mnist(batch_size)
    print(f"訓練集大小: {len(train_loader.dataset)}")
    print(f"測試集大小: {len(test_loader.dataset)}")
    print()
    
    # 創建模型（僅使用暫退法的模型）
    models = [
        SimpleDropoutMLP(num_inputs, num_outputs, num_hiddens1, num_hiddens2, 
                         dropout1, dropout2)
    ]
    
    model_names = ["使用暫退法的模型"]

    # 將模型移到設備上
    for model in models:
        model.to(device)
    
    # 訓練所有模型
    histories = []
    
    for model, name in zip(models, model_names):
        # 創建損失函數和優化器
        loss_fn = nn.CrossEntropyLoss(reduction='none')  # 與notebook保持一致
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 使用SGD優化器
        
        # 訓練模型
        history = train_model(
            model, train_loader, test_loader, loss_fn, optimizer, 
            num_epochs, device, name
        )
        
        histories.append(history)
    
    # 繪製訓練歷史圖（僅一個模型）
    plot_training_history(histories, model_names)
    
    # 分析暫退法效果（單模型）
    analyze_dropout_effect(histories, model_names)
    
    print("程式執行完成！")
    print("\n重要觀察:")
    print("1. 暫退法能有效減少過拟合")
    print("2. 暫退法提高了模型的泛化能力")
    print("3. 不同層可以使用不同的暫退概率")
    print("4. 測試時必須關閉暫退法")

if __name__ == "__main__":
    main()
