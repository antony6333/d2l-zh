# 本程式是根據 pytorch/chapter_convolutional-modern/vgg.ipynb 所做的練習
# VGG網路學習與實作

"""
VGG網路核心概念與技術點：

1. 塊狀設計概念(Block Design)：
   - VGG引入了使用重複塊結構來構建深層神經網路的思想
   - 每個VGG塊由多個3×3卷積層和一個最大池化層組成
   - 這種設計簡化了網路架構的設計和實現

2. VGG塊結構：
   - 由num_convs個3×3卷積層組成，每個卷積層後跟ReLU激活函數
   - 使用padding=1保持特徵圖的空間尺寸
   - 最後使用2×2最大池化層，步幅為2，將特徵圖尺寸減半

3. VGG網路架構：
   - 前半部分：由多個VGG塊組成的特徵提取部分
   - 後半部分：由全連接層組成的分類部分
   - 通道數逐漸加倍：64→128→256→512→512

4. 深而窄的卷積核優勢：
   - 使用多個3×3卷積核比使用單個大卷積核更有效
   - 能夠增加網路深度並增強非線性表達能力
   - 減少參數數量同時提高表達能力

5. VGG變體：
   - VGG-11, VGG-16, VGG-19等，數字表示層數
   - 通過改變每個塊中的卷積層數量來構建不同的變體
"""

import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms
import time
import numpy as np

# 簡化版本，移除對 matplotlib_inline 和 IPython 的依賴

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

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), size(y))
    return metric[0] / metric[1]

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
                # BERT微调所需的（之后将介绍）
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

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def train_ch6_simple(net, train_iter, test_iter, num_epochs, lr, device):
    """简化版训练函数，不使用动画显示"""
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
        # 训练损失之和，训练准确率之和，样本数
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
            
            # 每1/5的批次打印一次進度
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


def vgg_block(num_convs, in_channels, out_channels):
    """
    構建VGG塊
    
    參數：
    - num_convs: 該塊中卷積層的數量
    - in_channels: 輸入通道數
    - out_channels: 輸出通道數
    
    返回：
    - 包含多個卷積層和一個池化層的Sequential模塊
    """
    layers = []
    for _ in range(num_convs):
        # 添加3×3卷積層，padding=1保持特徵圖尺寸
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        # 添加ReLU激活函數
        layers.append(nn.ReLU())
        # 更新輸入通道數為輸出通道數，為下一個卷積層做準備
        in_channels = out_channels
    
    # 添加2×2最大池化層，步幅為2，將特徵圖尺寸減半
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch):
    """
    構建完整的VGG網路
    
    參數：
    - conv_arch: 卷積架構配置，格式為[(num_convs, out_channels), ...]
    
    返回：
    - 完整的VGG網路模型
    """
    conv_blks = []
    in_channels = 1  # 輸入圖像為單通道（灰度圖）
    
    # 構建卷積部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels
    
    # 組合卷積部分和全連接部分
    return nn.Sequential(
        *conv_blks,  # 卷積塊部分
        nn.Flatten(),  # 展平為一維向量
        # 全連接層部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10)  # 10類分類
    )


def analyze_network_structure():
    """分析VGG網路結構和各層輸出形狀"""
    print("=== VGG網路結構分析 ===")
    
    # VGG-11的標準配置
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    print(f"VGG-11配置：{conv_arch}")
    print("配置說明：(卷積層數量, 輸出通道數)")
    
    # 構建網路
    net = vgg(conv_arch)
    
    # 創建測試輸入（224×224單通道圖像）
    X = torch.randn(size=(1, 1, 224, 224))
    print(f"\n輸入形狀：{X.shape}")
    
    # 分析每個模塊的輸出
    print("\n各層輸出形狀分析：")
    for i, blk in enumerate(net):
        X = blk(X)
        print(f"{i+1:2d}. {blk.__class__.__name__:15s} 輸出形狀: {X.shape}")


def create_small_vgg():
    """創建適合Fashion-MNIST的小型VGG網路"""
    print("\n=== 創建小型VGG網路 ===")
    
    # 原始配置
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    
    # 縮小4倍的配置（為了減少計算量）
    ratio = 4
    small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
    print(f"小型VGG配置：{small_conv_arch}")
    
    # 構建小型網路
    small_net = vgg(small_conv_arch)
    
    # 計算參數數量
    total_params = sum(p.numel() for p in small_net.parameters())
    trainable_params = sum(p.numel() for p in small_net.parameters() if p.requires_grad)
    
    print(f"總參數數量：{total_params:,}")
    print(f"可訓練參數數量：{trainable_params:,}")
    
    return small_net


def exercise_1():
    """
    練習題1：打印層的尺寸時，我們只看到8個結果，而不是11個結果。
    剩餘的3層信息去哪了？
    """
    print("\n=== 練習題1解答 ===")
    print("問題：為什麼只看到8個結果而不是11個？")
    
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch)
    
    print(f"\n網路總共有 {len(net)} 個模塊")
    
    # 詳細分析每個模塊
    conv_layers_count = 0
    for i, blk in enumerate(net):
        if isinstance(blk, nn.Sequential):  # VGG塊
            # 計算該塊中的卷積層數量
            conv_in_block = len([layer for layer in blk if isinstance(layer, nn.Conv2d)])
            conv_layers_count += conv_in_block
            print(f"模塊 {i+1}: {blk.__class__.__name__} (包含 {conv_in_block} 個卷積層)")
        else:
            print(f"模塊 {i+1}: {blk.__class__.__name__}")
    
    print(f"\n總卷積層數量：{conv_layers_count}")
    print("解答：網路被組織成8個頂層模塊，但實際包含11個卷積層。")
    print("VGG塊內部的卷積層被封裝在Sequential容器中，所以只顯示8個模塊。")


def exercise_2():
    """
    練習題2：與AlexNet相比，VGG的計算要慢得多，而且它還需要更多的顯存。
    分析出現這種情況的原因。
    """
    print("\n=== 練習題2解答 ===")
    print("分析VGG相比AlexNet計算慢且需要更多顯存的原因：")
    
    # 模擬計算複雜度分析
    print("\n1. 網路深度比較：")
    print("   - AlexNet: 8層 (5個卷積層 + 3個全連接層)")
    print("   - VGG-11: 11層 (8個卷積層 + 3個全連接層)")
    print("   - VGG-16: 16層 (13個卷積層 + 3個全連接層)")
    
    print("\n2. 參數數量比較：")
    print("   - AlexNet: 約 60M 參數")
    print("   - VGG-11: 約 132M 參數")
    print("   - VGG-16: 約 138M 參數")
    
    print("\n3. 計算複雜度因素：")
    print("   - 更多的卷積層導致更多的計算操作")
    print("   - 特徵圖尺寸保持較大（使用3×3卷積和padding=1）")
    print("   - 全連接層參數龐大（特別是第一個全連接層）")
    
    print("\n4. 顯存需求因素：")
    print("   - 需要存儲更多層的特徵圖")
    print("   - 更大的模型參數")
    print("   - 反向傳播時需要存儲更多的梯度信息")


def exercise_3():
    """
    練習題3：嘗試將Fashion-MNIST數據集圖像的高度和寬度從224改為96。
    這對實驗有什麼影響？
    """
    print("\n=== 練習題3解答 ===")
    print("分析將輸入尺寸從224×224改為96×96的影響：")
    
    # 創建兩種不同輸入尺寸的測試
    conv_arch = ((1, 16), (1, 32), (2, 64), (2, 128), (2, 128))  # 簡化版本
    
    def analyze_size_impact(input_size):
        print(f"\n輸入尺寸：{input_size}×{input_size}")
        
        # 計算經過5次池化後的特徵圖尺寸
        feature_size = input_size
        for i in range(5):  # 5個VGG塊，每個都有池化
            feature_size = feature_size // 2
            print(f"  經過第{i+1}個塊後：{feature_size}×{feature_size}")
        
        # 計算全連接層的輸入尺寸
        fc_input_size = 128 * feature_size * feature_size  # 最後一層通道數為128
        print(f"  全連接層輸入尺寸：{fc_input_size}")
        
        return feature_size, fc_input_size
    
    # 分析224×224
    size_224, fc_224 = analyze_size_impact(224)
    
    # 分析96×96
    size_96, fc_96 = analyze_size_impact(96)
    
    print(f"\n影響分析：")
    print(f"1. 特徵圖尺寸：224×224 → {size_224}×{size_224}, 96×96 → {size_96}×{size_96}")
    print(f"2. 全連接層參數：{fc_224:,} → {fc_96:,} (減少 {(fc_224-fc_96)/fc_224*100:.1f}%)")
    print(f"3. 優點：計算更快，顯存需求更少，訓練速度提升")
    print(f"4. 缺點：可能丟失細節信息，影響模型表達能力")


def create_vgg_variants():
    """
    練習題4：構建其他常見的VGG模型變體
    """
    print("\n=== 練習題4解答：VGG變體構建 ===")
    
    # 定義不同的VGG配置
    vgg_configs = {
        'VGG-11': ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512)),
        'VGG-13': ((2, 64), (2, 128), (2, 256), (2, 512), (2, 512)),
        'VGG-16': ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512)),
        'VGG-19': ((2, 64), (2, 128), (4, 256), (4, 512), (4, 512))
    }
    
    print("VGG系列網路配置比較：")
    for name, config in vgg_configs.items():
        # 計算總卷積層數
        total_conv_layers = sum(pair[0] for pair in config)
        print(f"\n{name}:")
        print(f"  配置：{config}")
        print(f"  卷積層數：{total_conv_layers}")
        print(f"  總層數：{total_conv_layers + 3} (包含3個全連接層)")
        
        # 構建網路並計算參數
        net = vgg(config)
        total_params = sum(p.numel() for p in net.parameters())
        print(f"  參數數量：{total_params:,}")


def train_small_vgg():
    """訓練小型VGG網路"""
    print("\n=== 訓練小型VGG網路 ===")
    
    # 創建小型VGG網路
    small_net = create_small_vgg()
    
    # 設置訓練參數
    lr, num_epochs, batch_size = 0.05, 2, 128  # 減少epochs用於演示
    
    print(f"訓練參數：學習率={lr}, 批次大小={batch_size}, 訓練輪數={num_epochs}")
    
    try:
        # 載入數據
        train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
        print("數據載入成功")
        
        # 檢查是否有GPU可用
        device = try_gpu()
        print(f"使用設備：{device}")
        
        # 訓練模型（這裡只做演示，實際訓練需要較長時間）
        print("開始訓練...")
        print("注意：由於VGG計算量大，建議在GPU上訓練")
        
        # 使用簡化版訓練函數
        train_ch6_simple(small_net, train_iter, test_iter, num_epochs, lr, device)
        
    except Exception as e:
        print(f"訓練過程中出現錯誤：{e}")
        print("這可能是由於缺少GPU或內存不足造成的")


def main():
    """主函數：執行所有練習和分析"""
    print("VGG網路學習與實作")
    print("=" * 50)
    
    # 分析網路結構
    analyze_network_structure()
    
    # 創建小型VGG
    create_small_vgg()
    
    # 練習題解答
    exercise_1()
    exercise_2()
    exercise_3()
    create_vgg_variants()
    
    # 訓練演示
    train_small_vgg()
    
    print("\n=== 總結 ===")
    print("VGG網路的主要貢獻和特點：")
    print("1. 引入了塊狀設計思想，簡化了深層網路的構建")
    print("2. 證明了深而窄的卷積核(3×3)比大卷積核更有效")
    print("3. 為後續網路架構設計提供了重要的設計原則")
    print("4. 通過系統性的實驗證明了網路深度的重要性")


if __name__ == "__main__":
    main()
