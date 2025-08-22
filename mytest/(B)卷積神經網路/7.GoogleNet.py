# GoogleNet 是 2014 年由 Szegedy 等人提出的深度學習技術。
# 本程式是根據 pytorch/chapter_convolutional-modern/googlenet.ipynb 所做的練習
# GoogleNet (含並行連結的網路) 學習與實作

"""
GoogleNet核心概念與技術點：

1. Inception塊的設計思想：
   - 解決了什麼樣大小的卷積核最合適的問題
   - 使用不同大小的卷積核組合(1×1, 3×3, 5×5)並行處理
   - 通過多路徑設計同時捕獲不同尺度的特徵
   - 使用1×1卷積進行降維，減少計算複雜度

2. Inception塊的四條路徑：
   - 路徑1：單個1×1卷積層
   - 路徑2：1×1卷積層後接3×3卷積層
   - 路徑3：1×1卷積層後接5×5卷積層
   - 路徑4：3×3最大池化層後接1×1卷積層

3. GoogleNet的整體架構：
   - 使用9個Inception塊的堆疊
   - 採用全局平均池化層替代全連接層
   - 避免了過多的全連接層參數
   - 計算複雜度相對較低但效果良好

4. 降維技術的應用：
   - 1×1卷積用於降維和升維
   - 減少參數數量和計算複雜度
   - 增加網路的非線性表達能力

5. 多尺度特徵提取：
   - 同時使用不同大小的感受野
   - 能夠處理不同尺度的物體
   - 提高了模型的表達能力和泛化性能

6. 網路設計的創新點：
   - "Network in Network"的思想應用
   - 並行計算路徑的設計
   - 全局平均池化的使用
   - 計算效率和精度的平衡
"""

import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils import data
from torchvision import transforms
import time
import numpy as np

# ===== 從 d2l 模組複製過來的必要函數 =====

def get_dataloader_workers():
    """使用4个进程来读取数据"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 數學函數
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

class Timer:
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

def train_ch6_simple(net, train_iter, test_iter, num_epochs, lr, device):
    """简化版训练函数"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    timer, num_batches = Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                print(f'epoch {epoch + 1}, step {i + 1}/{num_batches}, '
                      f'train loss {train_l:.3f}, train acc {train_acc:.3f}')
        
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1}, test acc {test_acc:.3f}')
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

# ===== 結束 d2l 模組函數 =====


class Inception(nn.Module):
    """
    Inception塊實現
    
    Inception塊是GoogleNet的核心組件，包含四條並行路徑：
    1. 1×1卷積路徑
    2. 1×1卷積 → 3×3卷積路徑  
    3. 1×1卷積 → 5×5卷積路徑
    4. 3×3最大池化 → 1×1卷積路徑
    """
    
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        """
        參數：
        - in_channels: 輸入通道數
        - c1: 路徑1的輸出通道數
        - c2: 路徑2的通道數配置 (降維通道數, 輸出通道數)
        - c3: 路徑3的通道數配置 (降維通道數, 輸出通道數)  
        - c4: 路徑4的輸出通道數
        """
        super(Inception, self).__init__(**kwargs)
        
        # 路徑1：單1×1卷積層
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        
        # 路徑2：1×1卷積層後接3×3卷積層
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)  # 降維
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)  # 3×3卷積
        
        # 路徑3：1×1卷積層後接5×5卷積層
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)  # 降維
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)  # 5×5卷積
        
        # 路徑4：3×3最大池化層後接1×1卷積層
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 池化
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)  # 1×1卷積
    
    def forward(self, x):
        """前向傳播：並行計算四條路徑並連接輸出"""
        # 路徑1：直接1×1卷積
        p1 = F.relu(self.p1_1(x))
        
        # 路徑2：1×1降維 → 3×3卷積
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        
        # 路徑3：1×1降維 → 5×5卷積  
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        
        # 路徑4：最大池化 → 1×1卷積
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        
        # 在通道維度上連接四條路徑的輸出
        return torch.cat((p1, p2, p3, p4), dim=1)


def create_googlenet():
    """
    構建完整的GoogleNet網路
    
    GoogleNet包含5個模塊：
    - 模塊1：基礎卷積層
    - 模塊2：兩個卷積層
    - 模塊3：兩個Inception塊
    - 模塊4：五個Inception塊  
    - 模塊5：兩個Inception塊 + 全局平均池化
    """
    
    # 模塊1：64通道7×7卷積層，步幅2，後接最大池化
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 模塊2：兩個卷積層
    # 第一個：64通道1×1卷積
    # 第二個：192通道3×3卷積，通道數增加三倍
    b2 = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 模塊3：兩個Inception塊
    # 第一個Inception塊：輸出通道數 64+128+32+32=256
    # 第二個Inception塊：輸出通道數 128+192+96+64=480
    b3 = nn.Sequential(
        Inception(192, 64, (96, 128), (16, 32), 32),
        Inception(256, 128, (128, 192), (32, 96), 64),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 模塊4：五個Inception塊
    # 輸出通道數分別為：512, 512, 512, 528, 832
    b4 = nn.Sequential(
        Inception(480, 192, (96, 208), (16, 48), 64),
        Inception(512, 160, (112, 224), (24, 64), 64),
        Inception(512, 128, (128, 256), (24, 64), 64),
        Inception(512, 112, (144, 288), (32, 64), 64),
        Inception(528, 256, (160, 320), (32, 128), 128),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 模塊5：兩個Inception塊 + 全局平均池化
    # 輸出通道數：832, 1024
    b5 = nn.Sequential(
        Inception(832, 256, (160, 320), (32, 128), 128),
        Inception(832, 384, (192, 384), (48, 128), 128),
        nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
        nn.Flatten()
    )
    
    # 組合所有模塊並添加最終分類層
    net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))
    return net


def analyze_inception_block():
    """分析Inception塊的結構和輸出"""
    print("=== Inception塊結構分析 ===")
    
    # 創建一個Inception塊進行分析
    # 輸入通道192，輸出通道64+128+32+32=256
    inception_block = Inception(192, 64, (96, 128), (16, 32), 32)
    
    # 創建測試輸入
    x = torch.randn(1, 192, 28, 28)
    print(f"輸入形狀: {x.shape}")
    
    # 分析各路徑輸出
    with torch.no_grad():
        # 路徑1：1×1卷積
        p1 = F.relu(inception_block.p1_1(x))
        print(f"路徑1輸出形狀: {p1.shape} (1×1卷積)")
        
        # 路徑2：1×1 → 3×3卷積
        p2_temp = F.relu(inception_block.p2_1(x))
        p2 = F.relu(inception_block.p2_2(p2_temp))
        print(f"路徑2輸出形狀: {p2.shape} (1×1→3×3卷積)")
        
        # 路徑3：1×1 → 5×5卷積
        p3_temp = F.relu(inception_block.p3_1(x))
        p3 = F.relu(inception_block.p3_2(p3_temp))
        print(f"路徑3輸出形狀: {p3.shape} (1×1→5×5卷積)")
        
        # 路徑4：池化 → 1×1卷積
        p4_temp = inception_block.p4_1(x)
        p4 = F.relu(inception_block.p4_2(p4_temp))
        print(f"路徑4輸出形狀: {p4.shape} (池化→1×1卷積)")
        
        # 最終輸出
        output = inception_block(x)
        print(f"Inception塊輸出形狀: {output.shape}")
        print(f"輸出通道數驗證: {p1.shape[1]}+{p2.shape[1]}+{p3.shape[1]}+{p4.shape[1]} = {output.shape[1]}")


def analyze_googlenet_structure():
    """分析GoogleNet的整體結構"""
    print("\n=== GoogleNet網路結構分析 ===")
    
    # 創建網路
    net = create_googlenet()
    
    # 測試輸入（96×96單通道圖像）
    X = torch.randn(size=(1, 1, 96, 96))
    print(f"輸入形狀: {X.shape}")
    
    # 分析各模塊輸出
    print("\n各模塊輸出形狀:")
    for i, layer in enumerate(net):
        X = layer(X)
        if i < 5:  # 前5個是主要模塊
            print(f"模塊{i+1} 輸出形狀: {X.shape}")
        else:  # 最後的全連接層
            print(f"全連接層 輸出形狀: {X.shape}")


def compare_model_complexity():
    """比較不同模型的複雜度"""
    print("\n=== 模型複雜度比較 ===")
    
    # 創建GoogleNet
    googlenet = create_googlenet()
    
    # 計算參數數量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    googlenet_params = count_parameters(googlenet)
    print(f"GoogleNet 參數數量: {googlenet_params:,}")
    
    # 理論比較（基於文獻數據）
    print("\n理論模型比較:")
    print("AlexNet:    約 60M 參數")
    print("VGG-16:     約 138M 參數") 
    print(f"GoogleNet:  約 {googlenet_params/1000000:.1f}M 參數")
    print("ResNet-50:  約 26M 參數")
    
    print("\nGoogleNet優勢:")
    print("1. 相比VGG參數更少，計算效率更高")
    print("2. 使用1×1卷積進行降維，減少參數數量")
    print("3. 全局平均池化替代全連接層，避免過擬合")
    print("4. 多尺度特徵提取，表達能力強")


def exercise_1():
    """
    練習題1：分析GoogleNet後續版本的改進
    """
    print("\n=== 練習題1：GoogleNet後續版本分析 ===")
    
    print("GoogleNet的後續改進版本:")
    print("\n1. Inception v2 (批量歸一化):")
    print("   - 添加Batch Normalization層")
    print("   - 加速訓練並提高穩定性")
    print("   - 減少內部協變量偏移")
    
    print("\n2. Inception v3 (結構優化):")
    print("   - 將5×5卷積分解為兩個3×3卷積")
    print("   - 將n×n卷積分解為1×n和n×1卷積")
    print("   - 減少計算複雜度的同時保持表達能力")
    
    print("\n3. Inception v4 (標籤平滑):")
    print("   - 使用標籤平滑正則化技術")
    print("   - 改善模型的泛化能力")
    print("   - 減少過度自信的預測")
    
    print("\n4. Inception-ResNet (殘差連接):")
    print("   - 結合ResNet的殘差連接思想")
    print("   - 解決深層網路的退化問題")
    print("   - 允許訓練更深的網路")


def exercise_2():
    """
    練習題2：分析GoogleNet的最小圖像大小
    """
    print("\n=== 練習題2：GoogleNet最小圖像大小分析 ===")
    
    print("分析各階段的尺寸變化:")
    
    # 模擬不同輸入尺寸的變化
    def trace_size_changes(input_size):
        size = input_size
        print(f"\n輸入尺寸: {size}×{size}")
        
        # 模塊1: 7×7卷積，步幅2，池化步幅2
        size = ((size + 2*3 - 7) // 2 + 1)  # 卷積
        size = ((size + 2*1 - 3) // 2 + 1)  # 池化
        print(f"模塊1後: {size}×{size}")
        
        # 模塊2: 池化步幅2
        size = ((size + 2*1 - 3) // 2 + 1)
        print(f"模塊2後: {size}×{size}")
        
        # 模塊3: 池化步幅2
        size = ((size + 2*1 - 3) // 2 + 1)
        print(f"模塊3後: {size}×{size}")
        
        # 模塊4: 池化步幅2
        size = ((size + 2*1 - 3) // 2 + 1)
        print(f"模塊4後: {size}×{size}")
        
        # 模塊5: 全局平均池化到1×1
        print(f"模塊5後: 1×1")
        
        return size >= 1
    
    # 測試不同輸入尺寸
    test_sizes = [32, 64, 96, 224]
    print("測試不同輸入尺寸的可行性:")
    
    for size in test_sizes:
        is_valid = trace_size_changes(size)
        print(f"輸入{size}×{size}: {'✓ 可行' if is_valid else '✗ 不可行'}")
    
    print("\n結論:")
    print("理論最小輸入尺寸約為32×32")
    print("但實際使用中，建議使用96×96或更大尺寸以保證特徵提取效果")


def exercise_3():
    """
    練習題3：比較各種網路架構的參數大小
    """
    print("\n=== 練習題3：網路架構參數比較 ===")
    
    print("各網路架構的參數數量對比:")
    print("\n網路模型          參數數量        特點")
    print("="*50)
    print("LeNet-5          約 60K         最早的CNN之一")
    print("AlexNet          約 60M         深度學習復興的里程碑")  
    print("VGG-16           約 138M        簡單重複的塊結構")
    print("VGG-19           約 144M        更深的VGG變體")
    print("GoogleNet        約 7M          Inception塊設計")
    print("ResNet-50        約 26M         殘差連接")
    print("ResNet-152       約 60M         更深的殘差網路")
    
    print("\nGoogleNet參數減少的關鍵技術:")
    print("1. 1×1卷積降維:")
    print("   - 在大卷積核前使用1×1卷積減少通道數")
    print("   - 大幅減少計算量和參數數量")
    
    print("\n2. 全局平均池化:")
    print("   - 替代大型全連接層")
    print("   - 將特徵圖直接平均為類別得分")
    print("   - 避免了全連接層的大量參數")
    
    print("\n3. 多尺度並行處理:")
    print("   - 同時使用不同大小的卷積核")
    print("   - 提高特徵提取效率")
    print("   - 減少單一大卷積核的參數需求")
    
    print("\n4. 深而窄的設計:")
    print("   - 使用較小的卷積核和適度的通道數")
    print("   - 通過深度增加表達能力而非寬度")


def create_simplified_googlenet():
    """創建適合演示的簡化版GoogleNet"""
    print("\n=== 創建簡化版GoogleNet ===")
    
    # 簡化版配置（減少通道數）
    b1 = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=7, stride=2, padding=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    b2 = nn.Sequential(
        nn.Conv2d(16, 16, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(16, 48, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 簡化的Inception塊
    b3 = nn.Sequential(
        Inception(48, 16, (24, 32), (4, 8), 8),
        Inception(64, 32, (32, 48), (8, 24), 16),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    b4 = nn.Sequential(
        Inception(120, 48, (24, 52), (4, 12), 16),
        Inception(128, 40, (28, 56), (6, 16), 16),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    b5 = nn.Sequential(
        Inception(128, 64, (40, 80), (8, 32), 32),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )
    
    simplified_net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(208, 10))
    
    # 計算參數數量
    total_params = sum(p.numel() for p in simplified_net.parameters())
    print(f"簡化版GoogleNet參數數量: {total_params:,}")
    
    return simplified_net


def train_simplified_googlenet():
    """訓練簡化版GoogleNet"""
    print("\n=== 訓練簡化版GoogleNet ===")
    
    # 創建簡化版網路
    net = create_simplified_googlenet()
    
    # 設置訓練參數
    lr, num_epochs, batch_size = 0.1, 2, 128  # 減少epochs用於演示
    
    print(f"訓練參數：學習率={lr}, 批次大小={batch_size}, 訓練輪數={num_epochs}")
    
    try:
        # 載入數據（使用96×96分辨率）
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
        print("數據載入成功")
        
        # 檢查設備
        device = try_gpu()
        print(f"使用設備：{device}")
        
        # 訓練模型
        print("開始訓練...")
        train_ch6_simple(net, train_iter, test_iter, num_epochs, lr, device)
        
    except Exception as e:
        print(f"訓練過程中出現錯誤：{e}")
        print("這可能是由於硬體資源限制造成的")


def main():
    """主函數：執行所有練習和分析"""
    print("GoogleNet (含並行連結的網路) 學習與實作")
    print("=" * 60)
    
    # 分析Inception塊結構
    analyze_inception_block()
    
    # 分析GoogleNet整體結構
    analyze_googlenet_structure()
    
    # 比較模型複雜度
    compare_model_complexity()
    
    # 練習題解答
    exercise_1()
    exercise_2()
    exercise_3()
    
    # 創建和訓練簡化版本
    create_simplified_googlenet()
    train_simplified_googlenet()
    
    print("\n=== 總結 ===")
    print("GoogleNet的主要貢獻和創新:")
    print("1. Inception塊設計：解決了卷積核大小選擇的問題")
    print("2. 並行多尺度特徵提取：同時捕獲不同尺度的特徵")
    print("3. 1×1卷積降維：有效減少參數和計算量")
    print("4. 全局平均池化：避免全連接層的參數過多問題")
    print("5. 計算效率：在保持高精度的同時大幅減少參數數量")
    print("6. 設計思想：為後續的深度學習架構提供了重要啟發")


if __name__ == "__main__":
    main()
