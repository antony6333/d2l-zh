"""
簡易可執行的 softmax 回歸範例（來自 notebook：softmax-regression-scratch）

目的: 讀取Fashion-MNIST數據集上進行訓練和測試，讓模型學會辨識服飾類別。

原理: Softmax 回歸的分類過程可以理解為以下幾個步驟：
  1.圖像展平：
    原始圖像的每個像素點（例如 28x28 的灰階圖像，共 784 個像素）被展平成一個向量，作為輸入特徵。
  2.線性變換：
    每個像素點的值（特徵）與權重矩陣 ( W ) 相乘，並加上偏置 ( b )，得到每個類別的分數（logits）。這些分數表示模型對每個類別的信心。
  3.Softmax 函數：
    將這些分數轉換為機率分佈。Softmax 函數會將分數歸一化，使得所有類別的機率總和為 1。
  4.分類決策：
    最終的分類結果是選擇機率最大的類別，這表示模型認為該類別最有可能。
因此，Softmax 並不是直接判斷圖像的每個像素點，而是通過整體像素值的加權和來計算每個類別的機率，
從而進行分類。這種方法能夠捕捉圖像的全局特徵，而不是單獨依賴某個像素點。
"""
import os
import time
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 將 Animator 類定義放在檔案上方，確保在 train_ch3 被呼叫時可用
class Animator:
    """簡單的繪圖類，用於在訓練過程中增量繪製多條曲線。

    使用:
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0,1], legend=['loss','train acc','test acc'])
        animator.add(epoch, [loss, train_acc, test_acc])
        animator.show()
    """
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
        # 重新繪製
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


# 可選：使用 GPU（若可用）
# device 用來決定 tensor / model 要放在哪個裝置 (CPU 或 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------- 超參數 ---------------------------------
# 批次大小：一次送入 model 的樣本數
batch_size = 256
# 學習率
lr = 0.1
# 訓練週期數 (epochs)
num_epochs = 10

# 標籤文字（與 notebook 相同）
# 用於將數字標籤映射回可讀字串，僅在展示預測結果時使用
text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']


def load_data_fashion_mnist(batch_size, resize=None, root=None):
    """
    下載並建立 Fashion-MNIST 訓練與測試的 DataLoader。

    參數:
    - batch_size: 每個 mini-batch 的大小
    - resize: (可選) 將影像 resize 到指定大小（整數），若為 None 則保留 28x28
    - root: (可選) 資料儲存路徑，預設為專案 ../data

    回傳: (train_loader, test_loader)
    - 每個 loader 都是 torch.utils.data.DataLoader
    - transform 包含 ToTensor()，輸出 tensor 的形狀為 (C, H, W)，像素值範圍為 [0,1]

    注意: 設定 num_workers=4 以加速資料載入（若系統支援）
    """
    if root is None:
        # 預設資料放在專案的 ../data
        root = os.path.join(os.path.dirname(__file__), '..', 'data')
    '''
       transforms.ToTensor() 是一個轉換器，用於將影像資料從 PIL 格式或 NumPy 陣列轉換為 
       PyTorch 的張量（Tensor）。同時，它會將影像的像素值從範圍 [0, 255]（整數）正規化到 [0, 1]（浮點數）。    
    '''
    trans = [transforms.ToTensor()]
    if resize:
        # 若需要，先調整尺寸再轉為 tensor
        trans.insert(0, transforms.Resize(resize))
    # transforms.Compose 是一個將多個轉換組合在一起的工具 (例如先 resize 再轉為 tensor)
    trans = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root=root, train=True, transform=trans, download=True)
    mnist_test = datasets.FashionMNIST(root=root, train=False, transform=trans, download=True)
    '''
    DataLoader 物件作用：
    批次化數據：將 mnist_train 資料集分成大小為 batch_size 的小批次。
    隨機打亂數據：因為 shuffle=True，每個 epoch 開始時，數據會被隨機打亂。
    多進程加載：使用 num_workers=4，啟用 4 個子進程來加速數據加載。
    '''
    return DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=4)


# ------------------------ 模型參數初始化（已改為使用 PyTorch 高階 API） ------------------------
num_inputs = 784
num_outputs = 10

# 使用 nn.Sequential 定義模型（展平 + 全連接層）
'''
nn.Linear(num_inputs, num_outputs): 這裡是矩陣乘法的核心，求得每張圖片的特徵值
- 輸入：展平後的圖片特徵 X (形狀：batch_size × 784)
- 權重矩陣：W (形狀：784 × 10)
- 偏置向量：b (形狀：10)
- 矩陣乘法運算：y_hat = X @ W + b
'''
net = nn.Sequential(nn.Flatten(), nn.Linear(num_inputs, num_outputs))

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)

net.apply(init_weights)
net.to(device)

# 使用 PyTorch 內建的交叉熵（使用預設的 reduction='mean' 以回傳平均損失）
loss_fn = nn.CrossEntropyLoss()

# 優化器
trainer = torch.optim.SGD(net.parameters(), lr=lr)

def accuracy(y_hat, y):
    """
    計算正確預測的數量（非比例）。

    - 若 y_hat 具有類別機率 (batch_size, num_classes)，會先取 argmax；
      若是已經是類別索引 (batch_size,) 則直接比較。

    回傳: 正確預測的數量（float），以便累加計算平均準確度。
    """
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 若 y_hat 為機率分佈，取最大值的索引
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y  # 轉型以便比較
    return float(cmp.type(y.dtype).sum())  # 回傳整數個數 (轉成 float)


def evaluate_accuracy(net_fn, data_iter):
    """
    在測試集 (或任意 data_iter) 上評估模型的準確度。

    參數:
    - net_fn: 前向函式，接受一個 batch 的 X 並回傳預測（機率或類別）
    - data_iter: DataLoader，會回傳 (X, y)

    此函式在 with torch.no_grad() 下執行以關閉梯度計算，加速且節省記憶體。
    """
    metric = [0.0, 0.0]
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            metric[0] += accuracy(net_fn(X), y)  # 累加正確數
            metric[1] += y.shape[0]  # 累加樣本數（使用 shape[0]）
    # 回傳整體的平均準確度
    return metric[0] / metric[1]


def train_epoch_ch3(net_fn, train_iter, loss_fn, updater):
    """
    訓練一個 epoch 的迴圈（使用 chapter3 的風格命名）。

    參數:
    - net_fn: 前向函式
    - train_iter: 訓練資料的 DataLoader
    - loss_fn: 損失函式，會回傳平均損失
    - updater: 更新函式，無參數，負責執行 optimizer.step() 與 zero_grad()

    回傳: (平均損失, 平均準確度)
    - 平均損失是對整個 epoch 的所有樣本的平均損失
    - 平均準確度是整個 epoch 的平均正確率
    """
    metric = [0.0, 0.0, 0.0]  # loss sum, correct, num
    for X, y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net_fn(X)
        l = loss_fn(y_hat, y)  # 回傳的是平均損失（reduction='mean'）
        # l 已經是平均損失，直接進行 backward
        l.backward()
        # 使用 updater() 由外部處理 optimizer 的 step 與 zero_grad
        updater()
        # 使用 detach() 先將 tensor 從計算圖中分離，再轉為標量，避免警告
        metric[0] += l.detach().item() * y.shape[0]  # 累加損失總和（將平均損失乘以 batch 大小）
        metric[1] += accuracy(y_hat, y)  # 累加正確預測數
        metric[2] += y.shape[0]  # 累加樣本數（使用 shape[0] 明確為 batch 大小）
    # 回傳平均損失與平均準確度
    return metric[0] / metric[2], metric[1] / metric[2]


# 用於與 train_epoch_ch3 相容的 updater（無參數）
def updater():
    trainer.step()
    trainer.zero_grad()


def train_ch3(net_fn, train_iter, test_iter, loss_fn, num_epochs, updater_fn):
    """
    封裝整個訓練流程：多個 epoch 的迴圈，每個 epoch 訓練並評估測試集準確率。

    參數:
    - net_fn: 前向函式
    - train_iter, test_iter: 訓練與測試的 DataLoader
    - loss_fn: 損失函式
    - num_epochs: 訓練週期數
    - updater_fn: 參數更新函式

    每個 epoch 會印出訓練損失、訓練準確度、測試準確度與花費時間，並更新可視化圖表。
    """
    # 初始化動畫器，顯示三條曲線：訓練損失、訓練準確度、測試準確度
    # 不指定固定的 ylim，讓 matplotlib 自動縮放，避免 train loss 被裁切看不到
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=None,
                        legend=['train loss', 'train acc', 'test acc'])

    for epoch in range(num_epochs):
        start = time.time()
        train_loss, train_acc = train_epoch_ch3(net_fn, train_iter, loss_fn, updater_fn)
        test_acc = evaluate_accuracy(net_fn, test_iter)
        t = time.time() - start
        # 更新動畫器（將 train_loss, train_acc, test_acc 以 list 形式傳入）
        animator.add(epoch + 1, [train_loss, train_acc, test_acc])
        # 格式化輸出，顯示 epoch 編號、平均損失、訓練/測試準確度與時間
        print(f'epoch {epoch+1:02d}, loss {train_loss:.4f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}, time {t:.1f} sec')

    # 訓練結束，顯示最終圖表（在非互動環境會阻塞直到視窗關閉）
    try:
        animator.show()
    except Exception:
        # 若 animator.show 發生例外，退回到直接呼叫 plt.show()
        plt.show()


def predict_ch3(net_fn, test_iter, n=6):
    """
    顯示前 n 張測試影像的真實標籤與模型預測標籤（文字形式）。

    - 這裡會從 test_iter 取得第一個 batch，然後列印前 n 筆結果。
    """
    for X, y in test_iter:
        X, y = X.to(device), y.to(device)
        break
    # 將真實標籤與預測結果轉換為文字
    trues = [text_labels[int(i)] for i in y[:n]]
    preds = [text_labels[int(i)] for i in net_fn(X).argmax(axis=1)[:n]]
    for t, p in zip(trues, preds):
        print(f'true: {t:12s}  ->  pred: {p}')


def main():
    """
    程式進入點：載入資料、訓練模型並列印範例預測。
    """
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    print('device:', device)
    # 開始訓練：傳入使用 PyTorch 高階 API 的 net 與 loss_fn、updater
    train_ch3(net, train_iter, test_iter, loss_fn, num_epochs, updater)
    print('\nSample predictions:')
    predict_ch3(net, test_iter, n=10)


if __name__ == '__main__':
    main()

# 新增 Animator 類以在訓練過程中繪製進度（放在 imports 之後）
