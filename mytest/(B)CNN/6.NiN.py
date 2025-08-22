# NiN (Network in Network) 是 2013 年由 Lin 等人提出的深度學習技術。
# 本程式是根據 pytorch/chapter_convolutional-modern/nin.ipynb 所做的練習
# NiN (Network in Network) 網路學習與實作

"""
NiN網路核心概念與技術點：

1. MLP-conv（1x1 卷積）的引入：
   - NiN 用 1x1 卷積來替代傳統卷積後緊接的全連接層，將通道間的非線性組合放到每個空間位置上。
   - 1x1 卷積被視為對通道做逐點的多層感知器（MLP），因此稱為 "Network in Network"。

2. 用全局平均池化取代最後的全連接層：
   - 全局平均池化將每個通道壓縮為一個數值，減少參數並提升泛化能力。
   - 這使得模型尺寸小、計算需求低，並且較不容易過擬合。

3. 架構特點：
   - 在空間維度上仍使用卷積與池化來提取特徵，在通道維度使用多層1x1卷積強化非線性表達。
   - 相比於傳統大尺寸卷積核，NiN 更偏重於復合非線性映射。

4. 優點與缺點：
   - 優點：參數更少、計算高效、可移植性好、對過擬合友好（使用全局平均池化）。
   - 缺點：若不謹慎設計，模型深度或頻寬不足仍會限制表達能力。

(注意: 這個範例即使用GPU(geforce rtx 2060super)全力跑也是跑很久)
"""

import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import time
import numpy as np

# ===== 從 d2l 模組複製過來的必要函數（註：這些函數為從 d2l.torch 複製，用於資料載入/訓練/評估，並可能作小幅修改以符合本範例） =====

def get_dataloader_workers():
    """使用多個進程來加速資料讀取（從 d2l 複製）"""
    return 4


def load_data_fashion_mnist(batch_size, resize=None):
    """下載 Fashion-MNIST 並回傳 DataLoader（從 d2l 複製）"""
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
    """如果存在GPU，回傳該GPU裝置，否則回傳 CPU（從 d2l 複製）"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 數學函數（簡化封裝）
size = lambda x, *args, **kwargs: x.numel(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


def accuracy(y_hat, y):
    """計算預測正確數量（從 d2l 複製）"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = argmax(y_hat, axis=1)
    cmp = astype(y_hat, y.dtype) == y
    return float(reduce_sum(astype(cmp, y.dtype)))


class Accumulator:
    """在 n 個變數上累加（從 d2l 複製）"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在 GPU 上評估模型準確度（從 d2l 複製）"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
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
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()


def train_ch6_simple(net, train_iter, test_iter, num_epochs, lr, device):
    """簡化版訓練函數（從 d2l 複製並小幅修改輸出格式）"""
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
        metric = Accumulator(3)  # 训练损失之和，训练准确率之和，样本数
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

            if (i + 1) % (num_batches // 5 if num_batches >= 5 else 1) == 0 or i == num_batches - 1:
                print(f'epoch {epoch + 1}, step {i + 1}/{num_batches}, '
                      f'train loss {train_l:.3f}, train acc {train_acc:.3f}')

        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch + 1}, test acc {test_acc:.3f}')

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(device)}')

# ===== 結束從 d2l 複製的工具函式 =====


# ===== NiN 模型核心實作 =====

def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    """構建 NiN 塊：主卷積 + 兩個 1x1 卷積（視為局部 MLP），每層後接 ReLU

    輸入：
      - in_channels, out_channels: 通道數
      - kernel_size, stride, padding: 用於第一層的空間卷積參數
    回傳：nn.Sequential
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding),
        nn.ReLU(),
        # 1x1 conv 作為通道間的非線性組合
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )


def nin(num_classes=10):
    """構建一個適合 image input 的 NiN 網路（基於 d2l 範例的簡化版）"""
    return nn.Sequential(
        nin_block(1, 96, kernel_size=11, stride=4, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(96, 256, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nin_block(256, 384, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2),
        nn.Dropout(0.5),
        # 最後用 1x1 conv 將通道數改為類別數
        nn.Conv2d(384, num_classes, kernel_size=1),
        nn.ReLU(),
        # 全局平均池化：每個通道壓縮為標量
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )


def analyze_network_structure():
    """列印 NiN 網路各層輸出形狀的簡單分析（給開發者理解每層輸出）"""
    print("=== NiN 網路結構分析 ===")
    net = nin()
    X = torch.randn(size=(1, 1, 224, 224))
    print(f"輸入形狀：{X.shape}")
    for i, blk in enumerate(net):
        X = blk(X)
        print(f"{i+1:2d}. {blk.__class__.__name__:15s} -> {X.shape}")


def create_small_nin():
    """為 Fashion-MNIST 建立一個小型 NiN（縮小 channel 與 kernel）以降低計算量"""
    print("\n=== 創建小型 NiN 網路 ===")
    # 簡化版架構：減少通道數與較小 kernel，對 224x224 可用，也可縮放到更小的輸入
    net = nn.Sequential(
        nin_block(1, 48, kernel_size=5, stride=1, padding=2),
        nn.MaxPool2d(2, 2),
        nin_block(48, 128, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(2, 2),
        nin_block(128, 196, kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.5),
        nn.Conv2d(196, 10, kernel_size=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten()
    )

    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"小型 NiN 參數數量：{total_params:,}, 可訓練：{trainable_params:,}")
    return net


def exercise_1():
    """練習題示例：比較 NiN 中 1x1 convolution 的作用"""
    print("\n=== 練習題1：1x1 卷積的直觀觀察 ===")
    block = nin_block(3, 8, kernel_size=3, stride=1, padding=1)
    X = torch.randn(1, 3, 8, 8)
    Y = block(X)
    print(f"輸入：{X.shape} -> 輸出：{Y.shape}")
    print("說明：1x1 卷積不改變空間尺寸，但能在通道維度做複雜映射，類似於每個空間位置上的小型 MLP。")


def exercise_2():
    """練習題示例：NiN 與傳統卷積 + 全連接的比較"""
    print("\n=== 練習題2：NiN vs 傳統 CNN + FC ===")
    print("NiN 使用全局平均池化取代最後的全連接層，參數量下降且可解釋性更好（每個通道對應一類別）。")
    print("在小資料集上這通常能減少過擬合風險；缺點是在某些情況下表現可能略差於有強大 FC 的模型。")


def train_small_nin():
    """訓練小型 NiN 示範（對 Fashion-MNIST，resize=224 以匹配網路）"""
    print("\n=== 訓練小型 NiN 網路（示範） ===")
    net = create_small_nin()
    lr, num_epochs, batch_size = 0.05, 2, 128
    print(f"訓練參數：lr={lr}, epochs={num_epochs}, batch_size={batch_size}")

    try:
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
        print("資料載入成功")
        device = try_gpu()
        print(f"使用設備：{device}")
        train_ch6_simple(net, train_iter, test_iter, num_epochs, lr, device)
    except Exception as e:
        print(f"訓練過程錯誤：{e}")
        print("請確認是否有可用 GPU 或系統資源是否足夠。")


def main():
    print("NiN 網路學習與實作")
    print("=" * 50)
    analyze_network_structure()
    create_small_nin()
    exercise_1()
    exercise_2()
    train_small_nin()
    print("\n=== 總結 ===")
    print("NiN 的重點是使用 1x1 conv 當作局部 MLP 並以全局平均池化替代全連接層，能顯著降低參數並提升泛化。")


if __name__ == "__main__":
    main()
