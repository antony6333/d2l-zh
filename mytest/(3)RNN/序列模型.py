# 序列模型練習
# 根據教學章節：pytorch/chapter_recurrent-neural-networks/sequence.ipynb

"""
本章節核心概念與主要內容：

1. 序列數據的特性與挑戰：
   - 時間相關性：序列數據具有時間順序，未來不能影響過去
   - 錨定效應、享樂適應、季節性等心理學現象影響序列預測
   - 內插法 vs 外推法：預測未來比估計過去更困難

2. 統計工具與模型：
   - 自回歸模型 (Autoregressive Models)：使用固定長度的歷史數據預測未來
   - 隱變量自回歸模型 (Latent Autoregressive Models)：維護隱藏狀態總結歷史信息
   - 馬爾可夫模型：假設當前狀態只依賴於前一個或少數幾個狀態

3. 主要技術點：
   - 序列建模的數學框架：P(x_t | x_{t-1}, ..., x_1)
   - 馬爾可夫條件的應用 (Markov, 1906年提出)
   - 因果關係在時間序列中的重要性
   - k步預測的誤差累積問題

4. 實際應用挑戰：
   - 單步預測 vs 多步預測的性能差異
   - 誤差累積導致長期預測性能急速下降
   - 訓練數據的時間順序重要性
"""

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 設置中文字體
plt.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

# 從d2l.torch複製的工具函式
def load_array(data_arrays, batch_size, is_train=True):
    """構造一個PyTorch數據迭代器"""
    from torch.utils import data
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def evaluate_loss(net, data_iter, loss):
    """評估給定數據集上模型的損失"""
    metric = [0.0, 0]  # 損失總和，樣本數量
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            metric[0] += float(l.sum())
            metric[1] += int(y.numel())
    return metric[0] / metric[1]

def plot_sequence_data(time_list, data_list, x_label='time', y_label='x', 
                      legend=None, xlim=None, figsize=(6, 3)):
    """繪製序列數據"""
    plt.figure(figsize=figsize)
    for i, data in enumerate(data_list):
        label = legend[i] if legend and i < len(legend) else f'data_{i}'
        # 確保time和data都是一維的
        if isinstance(time_list, list):
            time_data = time_list[i] if i < len(time_list) else time_list[0]
        else:
            time_data = time_list
        
        # 將tensor轉換為numpy並確保是一維的
        if hasattr(time_data, 'detach'):
            time_data = time_data.detach().numpy()
        if hasattr(data, 'detach'):
            data = data.detach().numpy()
        
        # 確保維度匹配
        time_data = time_data.flatten()
        data = data.flatten()
        
        # 截取相同長度
        min_len = min(len(time_data), len(data))
        plt.plot(time_data[:min_len], data[:min_len], label=label)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if legend:
        plt.legend()
    if xlim:
        plt.xlim(xlim)
    plt.grid(True, alpha=0.3)
    plt.show()

def main():
    print("=" * 60)
    print("序列模型學習與實踐")
    print("=" * 60)
    
    # 1. 生成序列數據
    print("\n1. 生成正弦波序列數據（帶噪聲）")
    T = 1000  # 總共產生1000個點
    time = torch.arange(1, T + 1, dtype=torch.float32)
    # 使用正弦函數和一些可加性噪聲來生成序列數據
    x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
    
    print(f"生成了{T}個時間步的序列數據")
    print(f"數據範圍：[{x.min():.3f}, {x.max():.3f}]")
    
    # 繪製原始數據
    plot_sequence_data([time], [x], 
                      legend=['原始數據'], xlim=[1, 1000])
    
    # 2. 準備訓練數據
    print("\n2. 準備訓練數據（特徵-標籤對）")
    tau = 4  # 使用過去4個時間步作為特徵
    
    # 將序列轉換為特徵-標籤對
    # features[i] = [x_{i}, x_{i+1}, x_{i+2}, x_{i+3}]
    # labels[i] = x_{i+4}
    features = torch.zeros((T - tau, tau))
    for i in range(tau):
        features[:, i] = x[i: T - tau + i]
    labels = x[tau:].reshape((-1, 1))
    
    print(f"特徵矩陣形狀：{features.shape}")
    print(f"標籤向量形狀：{labels.shape}")
    print(f"每個樣本使用過去{tau}個時間步預測下一個時間步")
    
    # 準備訓練和測試數據
    batch_size, n_train = 16, 600
    train_iter = load_array((features[:n_train], labels[:n_train]),
                           batch_size, is_train=True)
    
    print(f"訓練樣本數：{n_train}")
    print(f"測試樣本數：{len(features) - n_train}")
    
    # 3. 構建神經網絡模型
    print("\n3. 構建多層感知機模型")
    
    def init_weights(m):
        """初始化網絡權重的函數"""
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    def get_net():
        """創建一個簡單的多層感知機"""
        net = nn.Sequential(
            nn.Linear(4, 10),   # 輸入層到隱藏層
            nn.ReLU(),          # ReLU激活函數
            nn.Linear(10, 1)    # 隱藏層到輸出層
        )
        net.apply(init_weights)
        return net
    
    # 定義損失函數
    loss = nn.MSELoss(reduction='none')
    
    print("模型架構：")
    print("- 輸入層：4個神經元（過去4個時間步）")
    print("- 隱藏層：10個神經元，ReLU激活")
    print("- 輸出層：1個神經元（預測下一個時間步）")
    print("- 損失函數：均方誤差損失")
    
    # 4. 訓練模型
    print("\n4. 訓練模型")
    
    def train(net, train_iter, loss, epochs, lr):
        """訓練神經網絡"""
        trainer = torch.optim.Adam(net.parameters(), lr)
        for epoch in range(epochs):
            for X, y in train_iter:
                trainer.zero_grad()         # 梯度清零
                l = loss(net(X), y)        # 前向傳播計算損失
                l.sum().backward()         # 反向傳播計算梯度
                trainer.step()             # 更新參數
            
            # 每個epoch後評估損失
            train_loss = evaluate_loss(net, train_iter, loss)
            print(f'epoch {epoch + 1}, loss: {train_loss:.6f}')
    
    net = get_net()
    print("開始訓練...")
    """
    train_iter中有n_train個樣本，每個樣本有tau=4個特徵(過去4個時間步)
    輸入為4個時間步的特徵值，輸出為下一個時間步的樣本
    ===> 即找出4個時間步對應到下一個時間步的映射關係
    """
    train(net, train_iter, loss, 5, 0.01)
    
    # 5. 單步預測評估
    print("\n5. 單步預測評估")
    
    # 對所有數據進行單步預測
    with torch.no_grad():
        onestep_preds = net(features)
    
    # 計算預測誤差
    pred_error = torch.mean((onestep_preds.squeeze() - labels.squeeze()) ** 2)
    print(f"單步預測均方誤差：{pred_error:.6f}")
    
    # 繪製單步預測結果
    plot_sequence_data([time, time[tau:]], 
                      [x, onestep_preds.squeeze()],
                      legend=['原始數據', '單步預測'], xlim=[1, 1000])
    
    # 6. 多步預測評估
    print("\n6. 多步預測評估")
    
    # 進行多步預測
    multistep_preds = torch.zeros(T)
    multistep_preds[:n_train + tau] = x[:n_train + tau]  # 使用真實值初始化
    
    # 從訓練數據結束後開始進行多步預測
    for i in range(n_train + tau, T):
        # 使用前tau個預測值作為輸入
        input_seq = multistep_preds[i - tau:i].reshape((1, -1))
        with torch.no_grad():
            multistep_preds[i] = net(input_seq)
    
    # 繪製多步預測結果
    plot_sequence_data([time, time[tau:], time[n_train + tau:]], 
                      [x, 
                       onestep_preds.squeeze(),
                       multistep_preds[n_train + tau:]],
                      legend=['原始數據', '單步預測', '多步預測'], 
                      xlim=[1, 1000])
    
    # 7. k步預測分析
    print("\n7. k步預測性能分析")
    
    max_steps = 64
    
    # 準備k步預測的特徵矩陣
    features_k = torch.zeros((T - tau - max_steps + 1, tau + max_steps))
    
    # 前tau列是真實觀測值
    for i in range(tau):
        features_k[:, i] = x[i: i + T - tau - max_steps + 1]
    
    # 後面的列是k步預測結果
    for i in range(tau, tau + max_steps):
        with torch.no_grad():
            features_k[:, i] = net(features_k[:, i - tau:i]).reshape(-1)
    
    # 分析不同k值的預測性能
    steps = [1, 4, 16, 64]
    print("不同k步預測的性能比較：")
    
    for k in steps:
        # 計算k步預測的均方誤差
        true_values = x[tau + k - 1: T - max_steps + k]
        pred_values = features_k[:, tau + k - 1]
        mse = torch.mean((pred_values - true_values) ** 2)
        print(f"{k}步預測均方誤差：{mse:.6f}")
    
    # 繪製不同k步預測結果
    time_indices = [time[tau + i - 1: T - max_steps + i] for i in steps]
    pred_values = [features_k[:, tau + i - 1] for i in steps]
    
    plot_sequence_data(time_indices, pred_values,
                      legend=[f'{i}步預測' for i in steps], 
                      xlim=[5, 1000])
    
    # 8. 練習題實作
    print("\n8. 練習題實作")
    
    # 練習1：嘗試不同的歷史觀測數量
    print("\n練習1：比較不同歷史觀測數量的效果")
    
    tau_values = [2, 4, 8, 16]
    for tau_test in tau_values:
        print(f"\n測試tau={tau_test}:")
        
        # 準備數據
        features_test = torch.zeros((T - tau_test, tau_test))
        for i in range(tau_test):
            features_test[:, i] = x[i: T - tau_test + i]
        labels_test = x[tau_test:].reshape((-1, 1))
        
        # 構建和訓練模型
        def get_net_flexible(input_size):
            net = nn.Sequential(
                nn.Linear(input_size, 10),
                nn.ReLU(),
                nn.Linear(10, 1)
            )
            net.apply(init_weights)
            return net
        
        net_test = get_net_flexible(tau_test)
        train_iter_test = load_array((features_test[:n_train], labels_test[:n_train]),
                                   batch_size, is_train=True)
        
        # 快速訓練3個epoch
        trainer = torch.optim.Adam(net_test.parameters(), 0.01)
        for epoch in range(3):
            for X, y in train_iter_test:
                trainer.zero_grad()
                l = loss(net_test(X), y)
                l.sum().backward()
                trainer.step()
        
        # 評估性能
        with torch.no_grad():
            preds_test = net_test(features_test)
        mse_test = torch.mean((preds_test.squeeze() - labels_test.squeeze()) ** 2)
        print(f"  單步預測均方誤差：{mse_test:.6f}")
    
    # 練習2：改變網絡架構
    print("\n練習2：比較不同網絡架構的效果")
    
    architectures = [
        ("小型網絡", [4, 5, 1]),
        ("標準網絡", [4, 10, 1]),
        ("大型網絡", [4, 20, 1]),
        ("深層網絡", [4, 10, 10, 1])
    ]
    
    for name, layers in architectures:
        print(f"\n測試{name}:")
        
        # 構建網絡
        modules = []
        for i in range(len(layers) - 1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:  # 除了最後一層都加ReLU
                modules.append(nn.ReLU())
        
        net_arch = nn.Sequential(*modules)
        net_arch.apply(init_weights)
        
        # 快速訓練
        trainer = torch.optim.Adam(net_arch.parameters(), 0.01)
        for epoch in range(3):
            for X, y in train_iter:
                trainer.zero_grad()
                l = loss(net_arch(X), y)
                l.sum().backward()
                trainer.step()
        
        # 評估性能
        train_loss = evaluate_loss(net_arch, train_iter, loss)
        print(f"  訓練損失：{train_loss:.6f}")
        print(f"  網絡結構：{' -> '.join(map(str, layers))}")
    
    print("\n" + "=" * 60)
    print("序列模型學習總結")
    print("=" * 60)
    print("1. 序列數據具有時間相關性，需要特殊的建模方法")
    print("2. 自回歸模型通過歷史數據預測未來，但受限於固定的時間窗口")
    print("3. 單步預測通常表現良好，但多步預測會出現誤差累積")
    print("4. 增加歷史觀測數量可能提升性能，但也增加模型複雜度")
    print("5. 網絡架構的選擇需要在容量和泛化能力之間平衡")
    print("6. 馬爾可夫假設簡化了序列建模，但可能丟失長期依賴關係")

if __name__ == "__main__":
    main()
