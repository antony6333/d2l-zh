"""
3. LeNet 卷積神經網路實現
本文件根據 lenet.ipynb 教學章節進行練習

LeNet 是最早發布的卷積神經網路之一，由 Yann LeCun 在 1989 年提出。
主要特點：
- 由卷積編碼器（兩個卷積層）和全連接層密集塊（三個全連接層）組成
- 使用 5×5 卷積核和 Sigmoid 激活函數
- 使用平均匯聚層進行空間降採樣
- 廣泛應用於手寫數字識別，如 ATM 機支票處理
"""

import os
import time
import torch
from torch import nn
from types import SimpleNamespace

# --- Minimal replacements for the small subset of `d2l.torch` used in this script ---
try:
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
except Exception:
    # If torchvision isn't available, the data loader will raise later with a clear error.
    datasets = None
    transforms = None
    DataLoader = None

class Accumulator:
    """Accumulate sums over n variables."""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        for i, v in enumerate(args):
            self.data[i] += float(v)

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """Compute number of correct predictions."""
    if y_hat.dim() > 1 and y_hat.shape[1] > 1:
        preds = y_hat.argmax(dim=1)
    else:
        preds = (y_hat >= 0.5).long().view(-1)
    cmp = preds.type(y.dtype) == y
    return float(cmp.type(torch.float32).sum().item())


def load_data_fashion_mnist(batch_size, resize=None, root=os.path.join('.', 'data')):
    """Load Fashion-MNIST dataset and return train/test data loaders.

    This is a minimal replacement for d2l.load_data_fashion_mnist used in the
    exercise. It relies on torchvision.datasets.FashionMNIST.
    """
    if datasets is None or transforms is None or DataLoader is None:
        raise RuntimeError('torchvision is required to load Fashion-MNIST; please install torchvision')

    transform_list = []
    if resize:
        transform_list.append(transforms.Resize(resize))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)

    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    train_iter = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    return train_iter, test_iter


def try_gpu():
    """Return GPU device if available, else CPU device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Animator:
    """A tiny no-op animator replacement that records values.

    d2l.Animator in the book shows live plots; for script runs we only need a
    simple recorder so calling sites don't break. This implementation stores
    added points and optionally prints occasional progress.
    """
    def __init__(self, xlabel=None, xlim=None, legend=None):
        self.xlabel = xlabel
        self.xlim = xlim
        self.legend = legend
        self.data = []

    def add(self, x, ys):
        # ys is a tuple matching legend length; store for potential inspection
        self.data.append((x, ys))

class Timer:
    """Simple timer to accumulate elapsed time across start/stop calls."""
    def __init__(self):
        self.start_time = None
        self.total = 0.0

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            return
        self.total += time.time() - self.start_time
        self.start_time = None

    def sum(self):
        return self.total

# Expose the small API under a `d2l` namespace so existing calls in this file
# keep working without changing all call sites.
d2l = SimpleNamespace(
    Accumulator=Accumulator,
    accuracy=accuracy,
    load_data_fashion_mnist=load_data_fashion_mnist,
    try_gpu=try_gpu,
    Animator=Animator,
    Timer=Timer,
)

# 標籤文字（方便展示預測結果）
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

# --- End of minimal d2l replacements ---

def create_lenet():
    """
    創建 LeNet-5 模型
    
    架構組成：
    1. 卷積編碼器：
       - Conv2d(1→6, kernel=5×5, padding=2) + Sigmoid + AvgPool2d(2×2, stride=2)
       - Conv2d(6→16, kernel=5×5) + Sigmoid + AvgPool2d(2×2, stride=2)
    2. 全連接層密集塊：
       - Linear(400→120) + Sigmoid
       - Linear(120→84) + Sigmoid  
       - Linear(84→10) (輸出層)
    
    Returns:
        nn.Sequential: LeNet 模型
    """
    net = nn.Sequential(
        # 第一個卷積塊
        nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 1×28×28 → 6×28×28
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),      # 6×28×28 → 6×14×14
        
        # 第二個卷積塊  
        nn.Conv2d(6, 16, kernel_size=5),            # 6×14×14 → 16×10×10
        nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),      # 16×10×10 → 16×5×5
        
        # 展平層
        nn.Flatten(),                               # 16×5×5 → 400
        
        # 全連接層密集塊
        nn.Linear(16 * 5 * 5, 120),                # 400 → 120
        nn.Sigmoid(),
        nn.Linear(120, 84),                        # 120 → 84
        nn.Sigmoid(),
        nn.Linear(84, 10)                          # 84 → 10 (10個類別)
    )
    return net


def check_model_shape(net):
    """
    檢查模型每一層的輸出形狀
    
    Args:
        net: LeNet 模型
    """
    print("LeNet 模型各層輸出形狀：")
    print("-" * 50)
    
    # 創建一個 28×28 的單通道輸入張量
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    
    for layer in net:
        X = layer(X)
        print(f'{layer.__class__.__name__:15} output shape: \t{X.shape}')
    
    print("-" * 50)
    print("說明：")
    print("- 第一個卷積層使用 padding=2 來補償 5×5 卷積核導致的特徵減少")
    print("- 第二個卷積層沒有填充，因此高度和寬度都減少了 4 個像素")
    print("- 通道數從 1 → 6 → 16，逐漸增加以捕捉更多特徵類型")
    print("- 每個匯聚層都將高度和寬度減半")
    print("- 全連接層逐漸減少維數，最終輸出 10 個類別的分類結果")


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """
    使用 GPU 計算模型在數據集上的精度
    
    Args:
        net: 神經網路模型
        data_iter: 數據迭代器
        device: 計算設備
        
    Returns:
        float: 準確率
    """
    if isinstance(net, nn.Module):
        net.eval()  # 設置為評估模式
        if not device:
            device = next(iter(net.parameters())).device
    
    # 正確預測的數量，總預測的數量
    metric = d2l.Accumulator(2)
    
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT 微調所需的（之後將介紹）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    
    return metric[0] / metric[1]


def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """
    用 GPU 訓練模型（第六章定義的訓練函數）
    
    Args:
        net: 神經網路模型
        train_iter: 訓練數據迭代器
        test_iter: 測試數據迭代器
        num_epochs: 訓練輪數
        lr: 學習率
        device: 計算設備
    """
    def init_weights(m):
        """Xavier 初始化權重"""
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    
    # 初始化權重
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    
    # 設置優化器和損失函數
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    
    # 設置動畫器來視覺化訓練過程
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        # 訓練損失之和，訓練準確率之和，樣本數
        metric = d2l.Accumulator(3)
        net.train()  # 設置為訓練模式
        
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            
            # 將數據移動到指定設備
            X, y = X.to(device), y.to(device)
            
            # 前向傳播
            y_hat = net(X)
            l = loss(y_hat, y)
            
            # 反向傳播
            l.backward()
            optimizer.step()
            
            # 統計訓練指標
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            
            # 計算當前訓練損失和準確率
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            
            # 定期更新動畫
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        
        # 計算測試準確率
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    
    # 輸出最終結果
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def predict_ch6(net, test_iter, device, n=6):
    """
    顯示前 n 張測試影像的真實標籤與模型預測（在訓練完成後使用）。
    """
    net.eval()
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            break
    preds = net(X).argmax(dim=1)
    trues = [text_labels[int(i)] for i in y[:n]]
    preds = [text_labels[int(i)] for i in preds[:n]]
    print('\nSample predictions:')
    for t, p in zip(trues, preds):
        print(f'true: {t:12s}  ->  pred: {p}')


def main():
    """
    主函數：創建模型、檢查架構、訓練模型
    """
    print("=" * 60)
    print("LeNet 卷積神經網路實現與訓練")
    print("=" * 60)
    
    # 1. 創建 LeNet 模型
    print("\n1. 創建 LeNet 模型")
    net = create_lenet()
    
    # 2. 檢查模型架構和形狀變化
    print("\n2. 檢查模型架構")
    check_model_shape(net)
    
    # 3. 載入 Fashion-MNIST 數據集
    print("\n3. 載入 Fashion-MNIST 數據集")
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
    print(f"批次大小: {batch_size}")
    
    # 4. 設置訓練參數
    print("\n4. 開始訓練 LeNet 模型")
    lr, num_epochs = 0.9, 10
    device = d2l.try_gpu()
    print(f"學習率: {lr}")
    print(f"訓練輪數: {num_epochs}")
    print(f"使用設備: {device}")
    
    # 5. 訓練模型
    train_ch6(net, train_iter, test_iter, num_epochs, lr, device)
    
    print("\n" + "=" * 60)
    print("訓練完成！")
    print("=" * 60)
    # 在訓練完成後顯示一些測試影像的預測結果
    try:
        predict_ch6(net, test_iter, device, n=10)
    except Exception as e:
        print('無法顯示預測結果:', e)


def demonstrate_pooling_effect():
    """
    演示匯聚層對位置不變性的影響
    """
    print("\n" + "=" * 60)
    print("演示匯聚層的位置不變性效果")
    print("=" * 60)
    
    # 創建一個簡單的卷積 + 匯聚層
    conv_pool = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    
    # 創建原始圖像（簡單的邊緣模式）
    original = torch.zeros(1, 1, 8, 8)
    original[0, 0, 2:6, 3:5] = 1.0  # 創建一個垂直邊緣
    
    # 創建向右平移一個像素的圖像
    shifted = torch.zeros(1, 1, 8, 8)
    shifted[0, 0, 2:6, 4:6] = 1.0  # 邊緣向右移動一個像素
    
    print("原始圖像:")
    print(original[0, 0])
    print("\n向右平移一個像素的圖像:")
    print(shifted[0, 0])
    
    # 通過卷積 + 匯聚層
    with torch.no_grad():
        output_original = conv_pool(original)
        output_shifted = conv_pool(shifted)
    
    print("\n原始圖像經過卷積+匯聚後的輸出:")
    print(output_original[0, 0])
    print("\n平移圖像經過卷積+匯聚後的輸出:")
    print(output_shifted[0, 0])
    
    # 計算差異
    diff = torch.abs(output_original - output_shifted).sum()
    print(f"\n兩個輸出之間的絕對差異總和: {diff:.4f}")
    print("匯聚層有助於減少對小幅位置變化的敏感性")


def analyze_feature_maps():
    """
    分析 LeNet 不同層的特徵圖
    """
    print("\n" + "=" * 60)
    print("分析 LeNet 特徵圖變化")
    print("=" * 60)
    
    # 創建 LeNet 模型
    net = create_lenet()
    
    # 創建一個樣本輸入
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    
    print("輸入圖像形狀:", X.shape)
    print("像素值範圍:", f"[{X.min():.3f}, {X.max():.3f}]")
    
    # 逐層分析
    layer_names = [
        "第一卷積層", "第一激活層", "第一匯聚層",
        "第二卷積層", "第二激活層", "第二匯聚層",
        "展平層", "第一全連接", "第二激活層",
        "第二全連接", "第三激活層", "輸出層"
    ]
    
    current_input = X
    for i, (layer, name) in enumerate(zip(net, layer_names)):
        current_input = layer(current_input)
        
        if len(current_input.shape) == 4:  # 卷積層輸出
            batch, channels, height, width = current_input.shape
            print(f"{name:12}: {channels:2d} 通道, {height:2d}×{width:2d} 空間尺寸")
        else:  # 全連接層輸出
            features = current_input.shape[1]
            print(f"{name:12}: {features:3d} 維特徵向量")


if __name__ == "__main__":
    # 執行主要訓練流程
    main()
    
    # 額外的演示和分析
    demonstrate_pooling_effect()
    analyze_feature_maps()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("\n核心概念總結：")
    print("1. LeNet 通過卷積層保留圖像的空間結構")
    print("2. 匯聚層提供位置不變性，減少對小幅位移的敏感性")
    print("3. 通道數逐漸增加（1→6→16），捕捉更多特徵類型")
    print("4. 空間維度逐漸減少（28×28→14×14→5×5），聚焦全局特徵")
    print("5. 全連接層將特徵映射到最終的分類結果")
    print("=" * 60)
