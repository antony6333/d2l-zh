# AlexNet 是 2012 年由 Krizhevsky 等人提出的深度學習技術。
# 根據: pytorch/chapter_convolutional-modern/alexnet.ipynb
# 本檔案為練習程式，示範 AlexNet 在 PyTorch 中的實作與簡短訓練流程。
# 章節核心概念說明：
# 1. AlexNet 是早期成功的深度卷積神經網路，透過大卷積核與較大的 stride 來快速減少空間維度，
#    並搭配 ReLU 非線性、重疊的最大池化、與 dropout 來改善訓練與泛化。
# 2. 對於現代小型資料集（例如 Fashion-MNIST），通常會把影像放大（resize）到 AlexNet 所需的較大輸入尺寸（224x224），
#    以符合原始架構的全連接層輸入尺寸假設；或是改寫最後全連接層以配合較小特徵圖尺寸。
# 3. 本實作重點：手動實作 AlexNet-like 網路、資料前處理（resize）、訓練/驗證 loop，並在程式中詳細註解每一步。

# 程式輸入/輸出契約（contract）
# - inputs: Fashion-MNIST 訓練/測試資料（單通道），會被 resize 到 224x224，batch_size 可設定
# - outputs: 每個 epoch 的訓練損失與驗證準確度（在範例中執行 1 個 epoch 作為煙霧測試）
# - 錯誤模式: 若無網路連線下載資料，會在 torchvision 下載步驟失敗；程式會提示使用者
# - 成功標準: 程式能在本地執行並回傳訓練損失與驗證準確度數值

# 常見邊界情況 (edge cases)
# - GPU 不可用：程式會自動退回到 CPU
# - 記憶體不足：可降低 batch_size 或將 resize 改小（但會改變模型輸出介面）
# - 下載資料失敗：請檢查網路或手動下載 Fashion-MNIST

# 實作程式碼
import os
import random
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import torch.amp as amp
import argparse

# 固定隨機種子以利重現
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 定義 AlexNet-like 模型（輸入為 1 通道，輸出類別數可配置）
class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(AlexNet, self).__init__()
        # AlexNet 的特徵抽取部分：
        # - 第一層使用較大的 kernel 與 stride（原文為 11x11, stride 4），可快速減少空間維度
        # - 中間使用 3x3 的 conv 作更深層的特徵組合
        # - 使用 MaxPool 做下採樣
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # 分類器（全連接層）
        # 注意：當輸入尺寸為 224x224 且經過上述卷積/池化後，特徵圖通常為 6x6，channel=256，因此 flatten 大小為 256*6*6
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 工具函式：計算模型在資料集上的準確度
def evaluate_accuracy(net: nn.Module, data_iter: DataLoader, device: torch.device) -> float:
    """Evaluate accuracy. Use non_blocking transfers when possible."""
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in data_iter:
            # use non_blocking=True if data loader uses pinned memory
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            outputs = net(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    net.train()
    return correct / total if total > 0 else 0.0

# 訓練 one epoch
def train_epoch(net: nn.Module, train_iter: DataLoader, loss_fn, optimizer, device: torch.device,
                scaler: amp.GradScaler = None) -> Tuple[float, float]:
    """Train one epoch. If a GradScaler is provided, use mixed precision (AMP)."""
    net.train()
    total_loss = 0.0
    total_samples = 0
    use_amp = scaler is not None
    for X, y in train_iter:
        # use non_blocking=True to overlap host->device copies when pin_memory=True
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad()
        if use_amp:
            # specify device_type to silence FutureWarning and ensure correct backend
            with amp.autocast(device_type='cuda'):
                outputs = net(X)
                loss = loss_fn(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = net(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()

        batch_size = X.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, total_samples

# 建立資料載入器（使用 Fashion-MNIST 作為示範）
def get_data_loaders(batch_size: int = 64, resize: int = 224, root: str = './data') -> Tuple[DataLoader, DataLoader]:
    # 轉換：resize -> ToTensor -> Normalize
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        # 使用常見的 ImageNet normalization 也可以，但 Fashion-MNIST 為單通道，我們使用簡單的 mean/std
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 下載或載入資料
    train_dataset = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)

    # 為了快速 smoke-test，我們可以選取資料的子集（這裡註解為選項）；預設使用全資料
    # small_train = Subset(train_dataset, indices=list(range(2000)))
    # small_test = Subset(test_dataset, indices=list(range(1000)))

    # If CUDA is available, enable pin_memory and use multiple workers to speed host->device transfer
    cuda_available = torch.cuda.is_available()
    # choose a reasonable number of workers but avoid too many on Windows to prevent spawn overhead
    num_workers = min(4, (os.cpu_count() or 1)) if cuda_available else 0
    pin_memory = True if cuda_available else False

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory,
                              persistent_workers=(num_workers > 0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory,
                             persistent_workers=(num_workers > 0))

    return train_loader, test_loader

# 主執行流程（簡短示範）
def main(epochs: int = 1, batch_size: int = 64, lr: float = 0.01, device: str = None):
    # 選擇裝置（若無提供則自動偵測）
    device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用裝置: {device}")

    # 取得資料
    train_iter, test_iter = get_data_loaders(batch_size=batch_size, resize=224, root=os.path.join(os.getcwd(), 'data'))

    # 如果使用 CUDA，啟用 cuDNN benchmark 以加速固定大小輸入的卷積運算
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 初始化模型
    net = AlexNet(num_classes=10, in_channels=1)
    net = net.to(device)

    # 損失與優化器
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    # Mixed precision scaler (only meaningful when using CUDA)
    # Create GradScaler (new API does not require device_type argument)
    scaler = amp.GradScaler() if device.type == 'cuda' else None

    # 執行簡短訓練以驗證程式流程
    for epoch in range(epochs):
        start = time.time()
        avg_loss, _ = train_epoch(net, train_iter, loss_fn, optimizer, device, scaler)
        val_acc = evaluate_accuracy(net, test_iter, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.4f} - time: {elapsed:.1f}s")

    # 最後輸出一個簡短的模型摘要（參數總數）
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"模型參數量（可訓練）: {num_params:,}")


def benchmark_search(device: torch.device, resize: int = 224, subset_size: int = 2048,
                     batch_sizes=(32, 64, 128), num_workers_list=(0, 1, 2, 4)):
    """Quick benchmark to search for best batch_size and num_workers.

    Measures time to run one training epoch over a subset of the training data
    (forward + backward + optimizer.step) and reports samples/sec.
    """
    print(f"Starting benchmark on device={device} (subset={subset_size})")
    cuda_available = device.type == 'cuda'
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.FashionMNIST(root=os.path.join(os.getcwd(), 'data'), train=True,
                                          transform=transform, download=True)

    # pick subset for quick benchmark
    subset_size = min(subset_size, len(train_dataset))
    indices = list(range(subset_size))
    train_subset = Subset(train_dataset, indices=indices)

    results = []
    # small model instance for compute; move to device
    model = AlexNet(num_classes=10, in_channels=1).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scaler = amp.GradScaler() if cuda_available else None

    for bs in batch_sizes:
        for nw in num_workers_list:
            pin_memory = True if cuda_available else False
            persistent = (nw > 0)
            loader = DataLoader(train_subset, batch_size=bs, shuffle=True,
                                num_workers=nw, pin_memory=pin_memory,
                                persistent_workers=persistent)

            # warm-up
            model.train()
            start_w = time.time()
            for i, (X, y) in enumerate(loader):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if scaler is not None:
                    with amp.autocast(device_type='cuda'):
                        outputs = model(X)
                        loss = loss_fn(outputs, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(X)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                # only do a single warm-up batch
                break
            torch.cuda.synchronize() if cuda_available else None
            warmup = time.time() - start_w

            # measured run
            model.zero_grad()
            start = time.time()
            samples = 0
            for X, y in loader:
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if scaler is not None:
                    with amp.autocast(device_type='cuda'):
                        outputs = model(X)
                        loss = loss_fn(outputs, y)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(X)
                    loss = loss_fn(outputs, y)
                    loss.backward()
                    optimizer.step()
                samples += X.size(0)
            if cuda_available:
                torch.cuda.synchronize()
            elapsed = time.time() - start
            samples_per_sec = samples / elapsed if elapsed > 0 else 0.0
            results.append(((bs, nw), samples_per_sec, elapsed))
            print(f"bs={bs:4d} nw={nw:2d} -> {samples_per_sec:7.1f} samples/s (time={elapsed:.2f}s)")

    # choose best
    best = max(results, key=lambda r: r[1])
    (best_bs, best_nw), best_thru, best_time = best
    print(f"Best config: batch_size={best_bs}, num_workers={best_nw} -> {best_thru:.1f} samples/s (time={best_time:.2f}s)")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlexNet small training / benchmark')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--benchmark', action='store_true', help='Run auto search benchmark for batch_size/num_workers')
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.benchmark:
        benchmark_search(device=device, resize=224, subset_size=2048,
                         batch_sizes=(32, 64, 128), num_workers_list=(0, 1, 2, 4))
    else:
        main(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device)
