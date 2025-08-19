"""
簡易可執行的 多層感知機 (MLP) 範例

目的: 讀取 Fashion-MNIST 數據集，使用簡潔 API (nn.Sequential) 定義多層感知機
並訓練、評估模型，最後列印若干預測結果。

此檔案參考原先的 softmax 分類訓練程式，重構為文件中「多層感知機的簡潔實現」版本。
"""
import os
import time
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 簡單繪圖輔助類（訓練過程即時繪圖）
class Animator:
    """簡單的增量繪圖工具，用於在訓練過程中顯示曲線。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, fmts=('-', 'm--', 'g-.', 'r:'), figsize=(7, 4)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.legend = legend or []
        self.xlim = xlim
        self.ylim = ylim
        self.fmts = fmts
        self.X = None
        self.Y = None

        if self.xlabel:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)

    def add(self, x, y):
        if not hasattr(y, '__len__'):
            y = [y]
        n = len(y)
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not hasattr(x, '__len__'):
            x = [x] * n
        for i, (xi, yi) in enumerate(zip(x, y)):
            if xi is None or yi is None:
                continue
            self.X[i].append(xi)
            self.Y[i].append(yi)
        self.ax.cla()
        for xi, yi, fmt in zip(self.X, self.Y, self.fmts):
            self.ax.plot(xi, yi, fmt)
        if self.legend:
            self.ax.legend(self.legend)
        if self.xlabel:
            self.ax.set_xlabel(self.xlabel)
        if self.ylabel:
            self.ax.set_ylabel(self.ylabel)
        if self.xlim:
            try:
                self.ax.set_xlim(self.xlim)
            except Exception:
                pass
        if self.ylim:
            try:
                self.ax.set_ylim(self.ylim)
            except Exception:
                pass
        plt.pause(0.001)

    def show(self):
        plt.show()

# 使用 GPU（若可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- 超參數 -----------------------------
batch_size = 256
lr = 0.1 # 學習率
num_epochs = 10

# 標籤文字（方便展示預測結果）
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def load_data_fashion_mnist(batch_size, resize=None, root=None):
    """下載並建立 Fashion-MNIST 訓練與測試的 DataLoader。

    參數:
    - batch_size: mini-batch 大小
    - resize: 若需改變輸入尺寸，請給定整數
    - root: 資料儲存路徑
    """
    if root is None:
        root = os.path.join(os.path.dirname(__file__), '..', 'data')
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    return DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)

# ------------------------ 模型（多層感知機，簡潔實現） ------------------------
num_inputs = 784
num_outputs = 10

# 使用 nn.Sequential 定義 MLP：展平 -> 全連接(784->256) -> ReLU -> 全連接(256->10)
net = nn.Sequential(
    nn.Flatten(),  #將輸入的 28x28 圖片展平成一維向量（784 維）
    nn.Linear(num_inputs, 256),  #第一個全連接層，將 784 維輸入映射到 256 維
    nn.ReLU(),  #激活函數，將非線性引入模型
    nn.Linear(256, num_outputs)  #第二個全連接層，將 256 維輸入映射到 10 維（對應 10 個分類）
)

# 權重初始化：對每個線性層的權重進行初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        # 權重初始化為均值為 0，標準差為 0.01 的正態分布
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

net.apply(init_weights)
net.to(device)

# 損失與優化器
# 交叉熵損失函數，適用於多分類問題
loss_fn = nn.CrossEntropyLoss()
# 隨機梯度下降 (SGD) 優化器，用於更新模型參數
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# ------------------------ 訓練與評估輔助函式 ------------------------

def accuracy(y_hat, y):
    """計算正確預測的數量（非比例）。"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net_fn, data_iter):
    """在 data_iter 上評估模型準確度。

    以 no_grad 模式執行以省記憶體與計算。
    """
    metric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric[0] += accuracy(net_fn(X), y)
            metric[1] += y.shape[0]
    return metric[0] / metric[1]


def train_epoch_ch3(net_fn, train_iter, loss_fn, updater):
    """訓練一個 epoch 的迴圈（與書中 chapter3 風格一致）。"""
    metric = [0.0, 0.0, 0.0]  # loss sum, correct, num
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net_fn(X)
        l = loss_fn(y_hat, y)
        l.backward()
        updater()
        metric[0] += l.detach().item() * y.shape[0]
        metric[1] += accuracy(y_hat, y)
        metric[2] += y.shape[0]
    return metric[0] / metric[2], metric[1] / metric[2]


# updater 為無參數函式，封裝 optimizer 的 step 與 zero_grad
def updater():
    trainer.step()
    trainer.zero_grad()


def train_ch3(net_fn, train_iter, test_iter, loss_fn, num_epochs, updater_fn):
    """完整訓練流程：多個 epoch，並在每個 epoch 評估測試集。"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=None,
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc = train_epoch_ch3(net_fn, train_iter, loss_fn, updater_fn)
        test_acc = evaluate_accuracy(net_fn, test_iter)
        t = time.time() - start
        animator.add(epoch + 1, [train_loss, train_acc, test_acc])
        print(f'epoch {epoch+1:02d}, loss {train_loss:.4f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}, time {t:.1f} sec')
    try:
        animator.show()
    except Exception:
        plt.show()


def predict_ch3(net_fn, test_iter, n=6):
    """顯示前 n 張測試影像的真實標籤與模型預測。"""
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        break
    trues = [text_labels[int(i)] for i in y[:n]]
    preds = [text_labels[int(i)] for i in net_fn(X).argmax(axis=1)[:n]]
    for t, p in zip(trues, preds):
        print(f'true: {t:12s}  ->  pred: {p}')


# ----------------------------- 主程式 -----------------------------

def main():
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    print('device:', device)
    train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, updater)
    print('\nSample predictions:')
    predict_ch3(net, test_iter, n=10)


if __name__ == '__main__':
    main()
