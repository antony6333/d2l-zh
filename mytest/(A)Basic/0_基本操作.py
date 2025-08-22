# 這是根據chapter_preliminaries的ndarray.ipynb、pandas.ipynb、linear-algebra.ipynb所作的練習
# 涵蓋三個章節：數據操作(ndarray)、數據預處理(pandas)、線性代數(linear-algebra)

import torch
import pandas as pd
import os

print("=== 第一部分：數據操作 (ndarray) ===")
print("基於 chapter_preliminaries/ndarray.ipynb")

# 1. 張量基本操作
print("\n1. 張量基本操作")

# 創建張量
x = torch.arange(12)
print(f"向量 x: {x}")
print(f"形狀: {x.shape}")
print(f"元素總數: {x.numel()}")

# 改變形狀
X = x.reshape(3, 4)
print(f"重塑為3x4矩陣:\n{X}")

# 創建特殊張量
zeros_tensor = torch.zeros((2, 3, 4))
ones_tensor = torch.ones((2, 3, 4))
random_tensor = torch.randn(3, 4)
manual_tensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(f"全零張量形狀: {zeros_tensor.shape}")
print(f"全一張量形狀: {ones_tensor.shape}")
print(f"隨機張量:\n{random_tensor}")
print(f"手動創建張量:\n{manual_tensor}")

# 2. 運算符
print("\n2. 按元素運算")
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(f"x: {x}")
print(f"y: {y}")
print(f"x + y: {x + y}")
print(f"x - y: {x - y}")
print(f"x * y: {x * y}")
print(f"x / y: {x / y}")
print(f"x ** y: {x ** y}")

# 指數運算
print(f"exp(x): {torch.exp(x)}")

# 3. 連結張量
print("\n3. 張量連結")
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"X:\n{X}")
print(f"Y:\n{Y}")
print(f"沿軸0連結:\n{torch.cat((X, Y), dim=0)}")
print(f"沿軸1連結:\n{torch.cat((X, Y), dim=1)}")

# 4. 邏輯運算
print("\n4. 邏輯運算")
print(f"X == Y:\n{X == Y}")

# 5. 求和
print(f"X所有元素求和: {X.sum()}")

# 6. 廣播機制
print("\n6. 廣播機制")
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(f"a (3x1):\n{a}")
print(f"b (1x2):\n{b}")
print(f"a + b (廣播結果):\n{a + b}")

# 7. 索引和切片
print("\n7. 索引和切片")
print(f"X[-1]: {X[-1]}")  # 最後一個元素
print(f"X[1:3]:\n{X[1:3]}")  # 第二到第三行

# 修改元素
X[1, 2] = 9
print(f"修改X[1,2]=9後:\n{X}")

# 8. 節省內存
print("\n8. 節省內存操作")
before = id(X)
X[:] = X + 1  # 原地操作
print(f"原地操作前後ID相同: {id(X) == before}")

# 9. 轉換為其他Python對象
print("\n9. 數據類型轉換")
A = X.numpy()
B = torch.tensor(A)
print(f"張量類型: {type(X)}")
print(f"NumPy數組類型: {type(A)}")
print(f"轉換回張量類型: {type(B)}")

a = torch.tensor([3.5])
print(f"標量張量: {a}")
print(f"轉為Python標量: {a.item()}")

print("\n=== 第二部分：數據預處理 (pandas) ===")
print("基於 chapter_preliminaries/pandas.ipynb")

# 1. 創建和讀取數據集
print("\n1. 創建和讀取CSV數據")

# 創建數據目錄和文件
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一個數據樣本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 讀取數據
data = pd.read_csv(data_file)
print("原始數據:")
print(data)

# 2. 處理缺失值
print("\n2. 處理缺失值")

# 分離輸入和輸出
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print("分離後的輸入:")
print(inputs)
print("輸出:")
print(outputs)

# 嘗試把可轉為數值的欄位先轉為數值（僅當原始非空值均可轉為數值時）
for col in inputs.columns:
    converted = pd.to_numeric(inputs[col], errors='coerce')
    non_null_mask = ~inputs[col].isnull()
    if non_null_mask.any() and converted[non_null_mask].notnull().all():
        inputs[col] = converted

# 用均值填充數值列的缺失值（僅對數值列，避免對字串欄位計算均值導致錯誤）
numeric_cols = inputs.select_dtypes(include=['number']).columns
if len(numeric_cols) > 0:
    inputs[numeric_cols] = inputs[numeric_cols].fillna(inputs[numeric_cols].mean())
# 對類別欄位填充特定字串，以便後續使用get_dummies處理
cat_cols = inputs.select_dtypes(include=['object', 'category']).columns
if len(cat_cols) > 0:
    inputs[cat_cols] = inputs[cat_cols].fillna('NA')
print("用均值填充後的輸入:")
print(inputs)

# 對類別值進行獨熱編碼
inputs = pd.get_dummies(inputs, dummy_na=True)
# 確保所有欄位為數值型態（float32），以便轉為torch張量
inputs = inputs.astype('float32')
print("獨熱編碼後的輸入:")
print(inputs)

# 轉換outputs為數值型，若有缺失則填補為0（可按需調整）
outputs = pd.to_numeric(outputs, errors='coerce')
if outputs.isnull().any():
    outputs = outputs.fillna(0)

# 3. 轉換為張量格式
print("\n3. 轉換為張量格式")
X_pandas, y_pandas = torch.tensor(inputs.values, dtype=torch.float32), torch.tensor(outputs.values, dtype=torch.float32)
print(f"輸入張量:\n{X_pandas}")
print(f"輸出張量: {y_pandas}")

print("\n=== 第三部分：線性代數 (linear-algebra) ===")
print("基於 chapter_preliminaries/linear-algebra.ipynb")

# 1. 標量
print("\n1. 標量操作")
x = torch.tensor(3.0)
y = torch.tensor(2.0)
print(f"標量運算: {x} + {y} = {x + y}")
print(f"標量運算: {x} * {y} = {x * y}")
print(f"標量運算: {x} / {y} = {x / y}")
print(f"標量運算: {x} ** {y} = {x ** y}")

# 2. 向量
print("\n2. 向量操作")
x = torch.arange(4)
print(f"向量 x: {x}")
print(f"訪問元素 x[3]: {x[3]}")
print(f"向量長度: {len(x)}")
print(f"向量形狀: {x.shape}")

# 3. 矩陣
print("\n3. 矩陣操作")
A = torch.arange(20).reshape(5, 4)
print(f"矩陣 A:\n{A}")
print(f"矩陣轉置 A.T:\n{A.T}")

# 對稱矩陣
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(f"矩陣 B:\n{B}")
print(f"B是否等於其轉置: {torch.equal(B, B.T)}")

# 4. 張量
print("\n4. 高階張量")
X = torch.arange(24).reshape(2, 3, 4)
print(f"3階張量 X 形狀: {X.shape}")

# 5. 張量算法的基本性質
print("\n5. 張量算法基本性質")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 創建副本
print(f"矩陣相加 A + B 的形狀: {(A + B).shape}")

# Hadamard積（按元素乘法）
print(f"Hadamard積 A * B 的形狀: {(A * B).shape}")

# 標量乘法
a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(f"標量乘法 a * X 的形狀: {(a * X).shape}")

# 6. 降維
print("\n6. 降維操作")
x = torch.arange(4, dtype=torch.float32)
print(f"向量 x: {x}")
print(f"向量求和: {x.sum()}")

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(f"矩陣 A 形狀: {A.shape}")
print(f"矩陣所有元素求和: {A.sum()}")

# 沿軸求和
A_sum_axis0 = A.sum(axis=0)
A_sum_axis1 = A.sum(axis=1)
print(f"沿軸0求和結果形狀: {A_sum_axis0.shape}")
print(f"沿軸1求和結果形狀: {A_sum_axis1.shape}")

# 平均值
print(f"矩陣平均值: {A.mean()}")
print(f"沿軸0平均值: {A.mean(axis=0)}")

# 非降維求和
sum_A = A.sum(axis=1, keepdims=True)
print(f"保持維度的求和形狀: {sum_A.shape}")

# 廣播除法
print(f"廣播除法 A / sum_A 的形狀: {(A / sum_A).shape}")

# 累積總和
print(f"沿軸0累積總和形狀: {A.cumsum(axis=0).shape}")

# 7. 點積
print("\n7. 點積")
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(f"向量 x: {x}")
print(f"向量 y: {y}")
print(f"點積 x·y: {torch.dot(x, y)}")
print(f"等價計算: {torch.sum(x * y)}")

# 8. 矩陣-向量積
print("\n8. 矩陣-向量積")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.arange(4, dtype=torch.float32)
print(f"矩陣 A 形狀: {A.shape}")
print(f"向量 x 形狀: {x.shape}")
mv_result = torch.mv(A, x)
print(f"矩陣-向量積結果形狀: {mv_result.shape}")

# 9. 矩陣-矩陣乘法
print("\n9. 矩陣乘法")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.ones(4, 3)
print(f"矩陣 A 形狀: {A.shape}")
print(f"矩陣 B 形狀: {B.shape}")
mm_result = torch.mm(A, B)
print(f"矩陣乘法結果形狀: {mm_result.shape}")

# 10. 範數
print("\n10. 範數")
u = torch.tensor([3.0, -4.0])
print(f"向量 u: {u}")
print(f"L2範數: {torch.norm(u)}")
print(f"L1範數: {torch.abs(u).sum()}")

# 矩陣的Frobenius範數
F_norm = torch.norm(torch.ones((4, 9)))
print(f"4x9全一矩陣的Frobenius範數: {F_norm}")

print("\n=== 練習題部分 ===")

# ndarray練習
print("\n【ndarray練習】")
print("1. 條件判斷操作:")
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(f"X < Y:\n{X < Y}")
print(f"X > Y:\n{X > Y}")

print("2. 三維張量廣播:")
a = torch.arange(6).reshape((2, 3, 1))
b = torch.arange(4).reshape((1, 1, 4))
print(f"3D廣播結果形狀: {(a + b).shape}")

# pandas練習
print("\n【pandas練習】")
print("創建更複雜的數據集:")
complex_data_file = os.path.join('..', 'data', 'complex_data.csv')
with open(complex_data_file, 'w') as f:
    f.write('A,B,C,D,E,Target\n')
    f.write('1,NA,3,4,NA,100\n')
    f.write('NA,2,NA,4,5,200\n')
    f.write('1,2,3,NA,5,150\n')
    f.write('NA,NA,3,4,5,180\n')
    f.write('1,2,NA,NA,NA,120\n')

complex_data = pd.read_csv(complex_data_file)
print("複雜數據集:")
print(complex_data)

# 找出缺失值最多的列
missing_counts = complex_data.isnull().sum()
print(f"各列缺失值數量:\n{missing_counts}")
max_missing_col = missing_counts.idxmax()
print(f"缺失值最多的列: {max_missing_col}")

# 刪除缺失值最多的列
complex_data_cleaned = complex_data.drop(columns=[max_missing_col])
print(f"刪除{max_missing_col}列後的數據:")
print(complex_data_cleaned)

# linear-algebra練習
print("\n【linear-algebra練習】")
print("1. 證明轉置的轉置等於原矩陣:")
A = torch.randn(3, 4)
print(f"(A^T)^T == A: {torch.equal(A.T.T, A)}")

print("2. 證明轉置的和等於和的轉置:")
A = torch.randn(3, 4)
B = torch.randn(3, 4)
print(f"A^T + B^T == (A + B)^T: {torch.equal(A.T + B.T, (A + B).T)}")

print("3. 檢查A + A^T是否總是對稱:")
A = torch.randn(4, 4)  # 方陣
symmetric_sum = A + A.T
print(f"A + A^T == (A + A^T)^T: {torch.equal(symmetric_sum, symmetric_sum.T)}")

print("4. 三維張量的len():")
X = torch.randn(2, 3, 4)
print(f"形狀(2,3,4)張量的len(): {len(X)}")
print("len(X)對應第0軸的長度")

print("5. 張量除法廣播問題分析:")
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
try:
    result = A / A.sum(axis=1)  # 這會出錯
except Exception as e:
    print(f"A/A.sum(axis=1)出錯: {e}")
    print("原因: A是(5,4), A.sum(axis=1)是(5,), 維度不匹配")
    print("解決方案: A / A.sum(axis=1, keepdims=True)")
    correct_result = A / A.sum(axis=1, keepdims=True)
    print(f"正確結果形狀: {correct_result.shape}")

print("6. 不同軸上的求和:")
X = torch.randn(2, 3, 4)
print(f"原張量形狀: {X.shape}")
print(f"沿軸0求和形狀: {X.sum(axis=0).shape}")
print(f"沿軸1求和形狀: {X.sum(axis=1).shape}")
print(f"沿軸2求和形狀: {X.sum(axis=2).shape}")

print("7. 多軸張量的範數:")
multi_tensor = torch.randn(2, 3, 4)
print(f"多軸張量的範數: {torch.norm(multi_tensor)}")
print("對任意形狀張量，torch.norm計算所有元素的L2範數")

print("\n=== 程式執行完成 ===")
print("本程式涵蓋了數據操作、數據預處理和線性代數的核心概念")
print("包含了基本張量操作、pandas數據處理和線性代數運算")
