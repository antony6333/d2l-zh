# 根據 pytorch/chapter_convolutional-modern/resnet.ipynb 進行的練習
# ResNet 殘差網路實作與訓練
# ResNet 是 2015 年由何愷明（Kaiming He）等人提出的深度學習技術。
"""
本章節的核心概念和主要內容：

1. 函數類與嵌套函數：
   - 當我們添加新層時，希望新模型至少和原模型一樣好
   - 只有當較複雜的函數類包含較小的函數類時，才能確保提高性能
   - 恒等映射是一個重要概念：f(x) = x

2. 殘差塊的設計理念：
   - 傳統塊直接學習映射 f(x)
   - 殘差塊學習殘差映射 f(x) - x，然後加上輸入 x 得到 f(x)
   - 這樣設計使得學習恒等映射變得更容易

3. 殘差塊的結構：
   - 兩個 3×3 卷積層
   - 每個卷積層後接批量歸一化層和 ReLU 激活函數
   - 跨層連接：將輸入直接加到最後的 ReLU 激活函數前
   - 當需要改變通道數時，使用 1×1 卷積層進行維度調整

4. ResNet 網路架構：
   - 第一層：7×7 卷積層，步幅為 2，輸出通道數為 64
   - 3×3 最大池化層，步幅為 2
   - 4 個殘差模組，每個模組包含若干個殘差塊
   - 全局平均池化層
   - 全連接層輸出

5. 技術優勢：
   - 解決了深層網路訓練困難的問題
   - 通過殘差連接實現梯度的直接傳播
   - 網路可以訓練得更深而不出現性能退化
"""

import torch
from torch import nn
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import numpy as np

# 從 d2l.torch 複製過來的工具函式
class Timer:
    """記錄多次運行時間"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """啟動計時器"""
        self.tik = time.time()

    def stop(self):
        """停止計時器並將時間記錄在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均時間"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回時間總和"""
        return sum(self.times)

    def cumsum(self):
        """返回累計時間"""
        return np.cumsum(self.times)

class Accumulator:
    """在n個變量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class Animator:
    """在動畫中繪製數據"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地繪製多條線
        if legend is None:
            legend = []
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函數捕獲參數
        self.config_axes = lambda: self.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """設置matplotlib的軸"""
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_xscale(xscale)
        axes.set_yscale(yscale)
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def add(self, x, y):
        # 向圖表中添加多個數據點
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.pause(0.01)

def accuracy(y_hat, y):
    """計算預測正確的數量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = (y_hat.type(y.dtype) == y)
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU計算模型在數據集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 設置為評估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正確預測的數量，總預測的數量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微調所需的（之後將介紹）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def try_gpu(i=0):
    """如果存在，則返回gpu(i)，否則返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def load_data_fashion_mnist(batch_size, resize=None):
    """下載Fashion-MNIST數據集，然後將其加載到內存中"""
    import torchvision
    from torchvision import transforms
    from torch.utils import data
    
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),  # Windows下設為0避免多進程問題
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU訓練模型"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                       legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        # 訓練損失之和，訓練準確率之和，樣本數
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
                animator.add(epoch + (i + 1) / num_batches,
                           (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

# 殘差塊的實現
class Residual(nn.Module):
    """
    殘差塊實現
    
    Args:
        input_channels: 輸入通道數
        num_channels: 輸出通道數
        use_1x1conv: 是否使用1×1卷積調整維度
        strides: 步幅
    """
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 第一個3×3卷積層
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                              kernel_size=3, padding=1, stride=strides)
        # 第二個3×3卷積層
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                              kernel_size=3, padding=1)
        
        # 如果需要調整維度，使用1×1卷積
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                  kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        
        # 批量歸一化層
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        # 主路徑：conv1 -> bn1 -> relu -> conv2 -> bn2
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        
        # 調整輸入維度（如果需要）
        if self.conv3:
            X = self.conv3(X)
        
        # 殘差連接：Y = F(X) + X，然後通過ReLU
        Y += X
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    """
    構建殘差模組
    
    Args:
        input_channels: 輸入通道數
        num_channels: 輸出通道數
        num_residuals: 殘差塊的數量
        first_block: 是否為第一個模組
    
    Returns:
        包含多個殘差塊的序列
    """
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 第一個殘差塊負責改變通道數和減半高寬
            blk.append(Residual(input_channels, num_channels,
                               use_1x1conv=True, strides=2))
        else:
            # 其他殘差塊保持維度不變
            blk.append(Residual(num_channels, num_channels))
    return blk

def create_resnet18():
    """創建ResNet-18模型"""
    # 第一個模組：7×7卷積 + 批量歸一化 + ReLU + 3×3最大池化
    b1 = nn.Sequential(
        nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), 
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )
    
    # 四個殘差模組
    b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))   # 64通道，2個殘差塊
    b3 = nn.Sequential(*resnet_block(64, 128, 2))                   # 128通道，2個殘差塊
    b4 = nn.Sequential(*resnet_block(128, 256, 2))                  # 256通道，2個殘差塊
    b5 = nn.Sequential(*resnet_block(256, 512, 2))                  # 512通道，2個殘差塊
    
    # 完整的ResNet-18網路
    net = nn.Sequential(
        b1, b2, b3, b4, b5,
        nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
        nn.Flatten(),                   # 展平
        nn.Linear(512, 10)             # 全連接層，10個類別
    )
    
    return net

def test_residual_block():
    """測試殘差塊的功能"""
    print("=== 測試殘差塊 ===")
    
    # 測試1：輸入和輸出形狀一致的情況
    print("\n1. 測試形狀一致的殘差塊：")
    blk = Residual(3, 3)
    X = torch.rand(4, 3, 6, 6)
    Y = blk(X)
    print(f"輸入形狀: {X.shape}")
    print(f"輸出形狀: {Y.shape}")
    
    # 測試2：增加輸出通道數，減半高寬
    print("\n2. 測試改變維度的殘差塊：")
    blk = Residual(3, 6, use_1x1conv=True, strides=2)
    Y = blk(X)
    print(f"輸入形狀: {X.shape}")
    print(f"輸出形狀: {Y.shape}")

def test_resnet_architecture():
    """測試ResNet架構"""
    print("\n=== 測試ResNet架構 ===")
    
    net = create_resnet18()
    
    # 觀察不同模組的輸出形狀變化
    X = torch.rand(size=(1, 1, 224, 224))
    print(f"輸入形狀: {X.shape}")
    
    for i, layer in enumerate(net):
        X = layer(X)
        print(f"第{i+1}個模組 {layer.__class__.__name__} 輸出形狀: {X.shape}")

def train_resnet():
    """訓練ResNet模型"""
    print("\n=== 訓練ResNet模型 ===")
    
    # 創建模型
    net = create_resnet18()
    
    # 設置訓練參數
    lr, num_epochs, batch_size = 0.05, 10, 256
    
    # 加載數據（調整圖像大小為96×96以加快訓練）
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    
    # 選擇設備
    device = try_gpu()
    
    # 開始訓練
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    
    return net

def analyze_resnet_advantages():
    """分析ResNet的優勢"""
    print("\n=== ResNet優勢分析 ===")
    print("""
    1. 解決梯度消失問題：
       - 殘差連接提供了梯度的直接路徑
       - 即使在很深的網路中也能有效傳播梯度
    
    2. 易於優化：
       - 學習殘差映射比學習完整映射更容易
       - 當理想映射接近恒等映射時，只需要將權重設為接近0
    
    3. 網路可以更深：
       - ResNet可以訓練到152層甚至更深
       - 而傳統網路通常在20-30層就會出現性能退化
    
    4. 性能提升：
       - 在ImageNet等大型數據集上取得了突破性成果
       - 為後續深度學習架構設計提供了重要啟發
    
    5. 結構簡潔：
       - 相比GoogLeNet的複雜結構，ResNet更加簡潔
       - 易於實現和修改
    """)

def practice_questions():
    """練習題實作"""
    print("\n=== 練習題 ===")
    
    print("\n1. Inception塊與殘差塊的主要區別：")
    print("""
    - Inception塊：並行多個不同大小的卷積和池化操作，然後連接
    - 殘差塊：順序的卷積操作 + 跨層連接（殘差連接）
    - 相同點：都是為了提高網路表達能力
    - 不同點：Inception注重多尺度特徵，ResNet注重梯度傳播
    """)
    
    print("\n2. 不同ResNet變體的實現：")
    print("可以通過調整殘差塊數量來實現不同深度的ResNet：")
    print("- ResNet-34: [3, 4, 6, 3]")
    print("- ResNet-50: [3, 4, 6, 3] + Bottleneck結構")
    print("- ResNet-101: [3, 4, 23, 3]")
    print("- ResNet-152: [3, 8, 36, 3]")
    
    print("\n3. Bottleneck架構的優勢：")
    print("""
    - 使用1×1卷積降維，3×3卷積處理，1×1卷積升維
    - 減少計算量和參數數量
    - 適合更深的網路
    """)

if __name__ == "__main__":
    print("ResNet 殘差網路學習與實作")
    print("=" * 50)
    
    # 測試殘差塊
    test_residual_block()
    
    # 測試網路架構
    test_resnet_architecture()
    
    # 分析優勢
    analyze_resnet_advantages()
    
    # 練習題
    practice_questions()
    
    # 訓練模型（可選，因為訓練時間較長）
    print("\n是否要開始訓練模型？(y/n): ", end="")
    import sys
    response = input().lower()
    if response == 'y':
        trained_net = train_resnet()
        plt.show()
    else:
        print("跳過模型訓練")
    
    print("\nResNet學習完成！")
