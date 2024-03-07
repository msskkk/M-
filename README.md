# M法-
誤報率をαに合わせるようなK点を算出し、異常検出ルールを定める

```
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import chi2
import matplotlib.pyplot as plt
#離散変量が2つの場合
cell= [[0,0], [1,0],[0,1],[1,1]]
cell=pd.DataFrame(cell)
```

```
#単位空間データ
n = 800
p = [0.25, 0.25, 0.25, 0.25]
# 平均
means = [[0, 0], [1, 1], [2, 2], [3, 3]]

# 相関行列
covs = [
    [[1, 0.8], [0.8, 1]],
    [[1, -0.8], [-0.8, 1]],
    [[1, 0.8], [0.8, 1]],
    [[1, -0.8], [-0.8, 1]]
]

combined_data = []


for i in range(4):
    n_cell = int(n * p[i])

    if i == 0:
        x_1 = np.zeros(n_cell)
        x_2 = np.zeros(n_cell)
        
    elif i == 1:
        x_1 = np.ones(n_cell)
        x_2 = np.zeros(n_cell)
    elif i == 2:
        x_1 = np.zeros(n_cell)
        x_2 = np.ones(n_cell)
    elif i == 3:
        x_1 = np.ones(n_cell)
        x_2 = np.ones(n_cell)

    mean=means[i]   
    cov=covs[i]
    y1, y2 = np.random.multivariate_normal(mean, cov, n_cell).T
    
    # データを結合して出力
    data = np.column_stack((x_1, x_2, y1, y2))
    combined_data.append(data)

# データを縦に結合
df0= np.vstack(combined_data)
# DataFrameに変換
df0 = pd.DataFrame(df0, columns=['x1', 'x2', 'y1', 'y2'])



# train
df0['cell番号'] = -1
for j in range(len(df0)):
  for i in range(4):
    if (df0.iloc[j, 0] == cell.iloc[i, 0]) & (df0.iloc[j, 1] == cell.iloc[i, 1]) :
      df0.loc[j, 'cell番号'] = i

# cell番号ごとにデータフレームを作成
unique_cell_numbers = df0['cell番号'].unique()
for cell_number in unique_cell_numbers:
    df_name = 'df_' + str(cell_number)
    globals()[df_name] = df0[df0['cell番号'] == cell_number]

# df_iから"cell番号"列を削除
for i in range(4):
    df_name = 'df_' + str(i)
    globals()[df_name] = globals()[df_name].drop("cell番号", axis=1)

# 4個のデータフレームを処理
for i in range(4):
    df_name = 'df_' + str(i)

    # 各データフレームを取得
    df = globals()[df_name]

    # 2値データの列と連続量データの列に分ける
    binary_cols = list(df.columns[:2])  # 最初の3列を2値データの列とする
    continuous_cols = list(df.columns[2:])  # 残りの列を連続量データの列とする

    # 2値データの列を含むデータフレームを作成
    df_name_dis = df[binary_cols].copy()
    df_name_dis_name = df_name + '_dis'

    # 連続量データの列を含むデータフレームを作成
    df_name_con = df[continuous_cols].copy()

    df_name_con_name = df_name + '_con'

    # 新しいデータフレームをグローバル変数として定義
    globals()[df_name_dis_name] = df_name_dis
    globals()[df_name_con_name] = df_name_con
```

```
#異常空間データ
n = 200
q = [0.25, 0.25, 0.25, 0.25]
# 平均
means = [[0, 2], [2, 2], [4, 4], [1, 5]]

# 相関行列
covs = [
    [[1, 0.8], [0.8, 1]],
    [[1, -0.8], [-0.8, 1]],
    [[1, -0.8], [-0.8, 1]],
    [[1, 0.8], [0.8, 1]]
]

combined_data = []


for i in range(4):
    n_cell = int(n * q[i])

    if i == 0:
        x_1 = np.zeros(n_cell)
        x_2 = np.zeros(n_cell)

    elif i == 1:
        x_1 = np.ones(n_cell)
        x_2 = np.zeros(n_cell)
    elif i == 2:
        x_1 = np.zeros(n_cell)
        x_2 = np.ones(n_cell)
    elif i == 3:
        x_1 = np.ones(n_cell)
        x_2 = np.ones(n_cell)

    mean=means[i]
    cov=covs[i]
    y1, y2 = np.random.multivariate_normal(mean, cov, n_cell).T

    # データを結合して出力
    data = np.column_stack((x_1, x_2, y1, y2))
    combined_data.append(data)

# データを縦に結合
df1= np.vstack(combined_data)
# DataFrameに変換
df1 = pd.DataFrame(df1, columns=['x1', 'x2', 'y1', 'y2'])

# test
df1['cell番号'] = -1
for j in range(len(df1)):
  for i in range(4):
    if (df1.iloc[j, 0] == cell.iloc[i, 0]) & (df1.iloc[j, 1] == cell.iloc[i, 1]) :
      df1.loc[j, 'cell番号'] = i

# cell番号ごとにデータフレームを作成
unique_cell_numbers = df1['cell番号'].unique()
for cell_number in unique_cell_numbers:
    df_name = 'df_' + str(cell_number)+'_test'
    globals()[df_name] = df1[df1['cell番号'] == cell_number]

# df_iから"cell番号"列を削除
for i in range(4):
    df_name = 'df_' + str(i)+'_test'
    globals()[df_name] = globals()[df_name].drop("cell番号", axis=1)

# ８個のデータフレームを処理
for i in range(4):
    df_name = 'df_' + str(i)+'_test'

    # 各データフレームを取得
    df = globals()[df_name]

    # 2値データの列と連続量データの列に分ける
    binary_cols = list(df.columns[:2])  # 最初の3列を2値データの列とする
    continuous_cols = list(df.columns[2:])  # 残りの列を連続量データの列とする

    # 2値データの列を含むデータフレームを作成
    df_name_dis = df[binary_cols].copy()
    df_name_dis_name = df_name + '_dis'

    # 連続量データの列を含むデータフレームを作成
    df_name_con = df[continuous_cols].copy()

    df_name_con_name = df_name + '_con'

    # 新しいデータフレームをグローバル変数として定義
    globals()[df_name_dis_name] = df_name_dis
    globals()[df_name_con_name] = df_name_co
```

```
class Mhou:
  def __init__(self,traindata,testdata,p,q,K):
    self.K = K
    self.p = p
    self.q = q
    self.c = [(1 - pi) / pi for pi in self.p]
    self.data=traindata
    self.testdata=testdata
    self.mu = None
    self.stt=None
    self.teststt=None
    self.sigma = None
    self.threshold = None
    self.corr=None
    self.min=None
    self.mahalanobis_dist=None
    self.mahalanobis_dist_test =None
    self.proportion=None
    self.testproportion=None

  def unitspace(self):
    self.mu = self.data.mean()
    self.sigma = self.data.std()
    self.stt=(self.data-self.mu)/self.sigma
    return self.stt
  def cor(self):
    self.corr=self.data.corr()
    self.min= np.linalg.inv(self.corr)
    return self.min
  def mh(self):
    if self.stt is None:
            self.unitspace()

    if self.corr is None:
            self.cor()

    self.mahalanobis_dist = []

    for row in self.stt.values:
      
      row = row.reshape(-1, 1)  # 行ベクトルを列ベクトルに変形
      distance = np.dot(row.T, np.dot(self.min, row))+self.c
      self.mahalanobis_dist.append(distance[0, 0])
    return self.mahalanobis_dist

  def plot_mahalanobis(self):
    if self.mahalanobis_dist is None:
      self.mh()
    x=np.arange(0, 5.3, 0.3)
    # 自由度5の非心カイ2乗分布を計算
    chi = chi2.pdf(x+self.c,self.data.shape[1])
    plt.plot(x+self.c, chi)
    plt.hist(self.mahalanobis_dist, bins=x,density=True)
    plt.xlabel('Mahalanobis Distance')
    plt.ylabel('Frequency')
    plt.title('Frequency of Mahalanobis Distances')
    plt.grid(True)
    plt.show()


    #test
  def unitspace_test(self):
    self.mu = self.data.mean()
    self.sigma = self.data.std()
    self.teststt=(self.testdata-self.mu)/self.sigma
    return self.teststt

    #MH_test
  def mh_test(self):
     if self.teststt is None:
            self.unitspace_test()

     if self.corr is None:
            self.cor()

     self.mahalanobis_dist_test = []
     c=((1-p[i])/p[i])
     for row in self.teststt.values:

      row = row.reshape(-1, 1)  # 行ベクトルを列ベクトルに変形
      distance = np.dot(row.T, np.dot(self.min, row))+c
      self.mahalanobis_dist_test.append(distance[0, 0])



     return self.mahalanobis_dist_test

  def mh_det(self):
    if self.mahalanobis_dist_test is None:
      self.mh_test()

    import scipy.stats as sps
   
    #非心χ２分布の条件付き検出力
    count = sum(dist >= self.K for dist in self.mahalanobis_dist_test)
    self.testproportion = count / len(self.mahalanobis_dist_test)
    return self.testproportion

  def mh_sig(self):
    if self.mahalanobis_dist is None:
      self.mh()

    import scipy.stats as sps
     #ゴールシークで合計α＝0.05の点Kを探す
    
    count = sum(dist >= self.K for dist in self.mahalanobis_dist)
    self.proportion = count / len(self.mahalanobis_dist)
    return self.proportion
```

```
def generate_unit_space_data(n=800, p=[0.25, 0.25, 0.25, 0.25], means=[[0, 0], [1, 1], [2, 2], [3, 3]],
                             covs=[[[1, 0.8], [0.8, 1]], [[1, -0.8], [-0.8, 1]], 
                                   [[1, 0.8], [0.8, 1]], [[1, -0.8], [-0.8, 1]]]):
    combined_data = []

    for i in range(4):
        n_cell = int(n * p[i])

        if i == 0:
            x_1 = np.zeros(n_cell)
            x_2 = np.zeros(n_cell)

        elif i == 1:
            x_1 = np.ones(n_cell)
            x_2 = np.zeros(n_cell)
        elif i == 2:
            x_1 = np.zeros(n_cell)
            x_2 = np.ones(n_cell)
        elif i == 3:
            x_1 = np.ones(n_cell)
            x_2 = np.ones(n_cell)

        mean = means[i]
        cov = covs[i]
        y1, y2 = np.random.multivariate_normal(mean, cov, n_cell).T

        # データを結合して出力
        data = np.column_stack((x_1, x_2, y1, y2))
        combined_data.append(data)

    # データを縦に結合
    df0 = np.vstack(combined_data)
    # DataFrameに変換
    df0 = pd.DataFrame(df0, columns=['x1', 'x2', 'y1', 'y2'])

    return df0

def preprocess_data(df0, cell):
    # train
    df0['cell番号'] = -1
    for j in range(len(df0)):
        for i in range(4):
            if (df0.iloc[j, 0] == cell.iloc[i, 0]) & (df0.iloc[j, 1] == cell.iloc[i, 1]):
                df0.loc[j, 'cell番号'] = i

    # cell番号ごとにデータフレームを作成
    unique_cell_numbers = df0['cell番号'].unique()
    for cell_number in unique_cell_numbers:
        df_name = 'df_' + str(cell_number)
        globals()[df_name] = df0[df0['cell番号'] == cell_number]

    # df_iから"cell番号"列を削除
    for i in range(4):
        df_name = 'df_' + str(i)
        globals()[df_name] = globals()[df_name].drop("cell番号", axis=1)

    # 4個のデータフレームを処理
    for i in range(4):
        df_name = 'df_' + str(i)

        # 各データフレームを取得
        df = globals()[df_name]

        # 2値データの列と連続量データの列に分ける
        binary_cols = list(df.columns[:2])  # 最初の3列を2値データの列とする
        continuous_cols = list(df.columns[2:])  # 残りの列を連続量データの列とする

        # 2値データの列を含むデータフレームを作成
        df_name_dis = df[binary_cols].copy()
        df_name_dis_name = df_name + '_dis'

        # 連続量データの列を含むデータフレームを作成
        df_name_con = df[continuous_cols].copy()

        df_name_con_name = df_name + '_con'

        # 新しいデータフレームをグローバル変数として定義
        globals()[df_name_dis_name] = df_name_dis
        globals()[df_name_con_name] = df_name_con

```

```
def generate_anomaly_space_data(n=200, q=[0.25, 0.25, 0.25, 0.25],
                                means=[[0, 2], [2, 2], [4, 4], [1, 5]],
                                covs=[[[1, 0.8], [0.8, 1]], [[1, -0.8], [-0.8, 1]],
                                      [[1, -0.8], [-0.8, 1]], [[1, 0.8], [0.8, 1]]]):
    combined_data = []

    for i in range(4):
        n_cell = int(n * q[i])

        if i == 0:
            x_1 = np.zeros(n_cell)
            x_2 = np.zeros(n_cell)

        elif i == 1:
            x_1 = np.ones(n_cell)
            x_2 = np.zeros(n_cell)
        elif i == 2:
            x_1 = np.zeros(n_cell)
            x_2 = np.ones(n_cell)
        elif i == 3:
            x_1 = np.ones(n_cell)
            x_2 = np.ones(n_cell)

        mean = means[i]
        cov = covs[i]
        y1, y2 = np.random.multivariate_normal(mean, cov, n_cell).T

        # データを結合して出力
        data = np.column_stack((x_1, x_2, y1, y2))
        combined_data.append(data)

    # データを縦に結合
    df1 = np.vstack(combined_data)
    # DataFrameに変換
    df1 = pd.DataFrame(df1, columns=['x1', 'x2', 'y1', 'y2'])

    return df1

def preprocess_anomaly_data(df1, cell):
    # test
    df1['cell番号'] = -1
    for j in range(len(df1)):
        for i in range(4):
            if (df1.iloc[j, 0] == cell.iloc[i, 0]) & (df1.iloc[j, 1] == cell.iloc[i, 1]):
                df1.loc[j, 'cell番号'] = i

    # cell番号ごとにデータフレームを作成
    unique_cell_numbers = df1['cell番号'].unique()
    for cell_number in unique_cell_numbers:
        df_name = 'df_' + str(cell_number) + '_test'
        globals()[df_name] = df1[df1['cell番号'] == cell_number]

    # df_iから"cell番号"列を削除
    for i in range(4):
        df_name = 'df_' + str(i) + '_test'
        globals()[df_name] = globals()[df_name].drop("cell番号", axis=1)

    # 8個のデータフレームを処理
    for i in range(4):
        df_name = 'df_' + str(i) + '_test'

        # 各データフレームを取得
        df = globals()[df_name]

        # 2値データの列と連続量データの列に分ける
        binary_cols = list(df.columns[:2])  # 最初の3列を2値データの列とする
        continuous_cols = list(df.columns[2:])  # 残りの列を連続量データの列とする

        # 2値データの列を含むデータフレームを作成
        df_name_dis = df[binary_cols].copy()
        df_name_dis_name = df_name + '_dis'

        # 連続量データの列を含むデータフレームを作成
        df_name_con = df[continuous_cols].copy()

        df_name_con_name = df_name + '_con'

        # 新しいデータフレームをグローバル変数として定義
        globals()[df_name_dis_name] = df_name_dis
        globals()[df_name_con_name] = df_name_con

# 使用例
cell = pd.DataFrame([[0, 0], [1, 0], [0, 1], [1, 1]], columns=['x1', 'x2'])
df1 = generate_anomaly_space_data()
preprocess_anomaly_data(df1, cell)
```
```
def calculate_alpha(K, p):
    alpha_list = []
    for i in range(4):
        c = (1 - p[i]) / p[i]
        kai = 1 - sps.chi2.cdf(K - c, df=2)
        alpha = p[i] * kai
        alpha_list.append(alpha)
    return sum(alpha_list)
```
```
from tqdm import tqdm
import time

# 初期値
K =15
target_alpha = 0.05
step_size = 0.1  

# 最大反復回数
max_iterations = 100

# 反復
for iteration in tqdm(range(max_iterations)):
    time.sleep(0.1)
    current_alpha = calculate_alpha(K, p)
    # 差分
    if abs(current_alpha - target_alpha) < 0.0001:
        break
    # 更新
    if current_alpha > target_alpha:
      K += step_size
    if current_alpha <target_alpha:
      K -= step_size

print("Optimal K:", K)
print("Achieved alpha:", calculate_alpha(K, p))
K
```
```
# 反復
kens = [[] for _ in range(4)]
gohous = [[] for _ in range(4)]

j=100

for iteration in tqdm(range(j)):
  time.sleep(0.1)
  df0 = generate_unit_space_data()
  preprocess_data(df0, cell)
  df1 = generate_anomaly_space_data()
  preprocess_anomaly_data(df1, cell)
  

  for i in range(4):
    df_name = 'df_' + str(i)
    df_name_test = 'df_' + str(i) + '_test'
    df_name_con_name = df_name + "_con"
    df_name_con_name_test = df_name_test + "_con"

    mt = Mhou(globals()[df_name_con_name], globals()[df_name_con_name_test],p,q,K)
    ken = mt.mh_det()
    gohou = mt.mh_sig()
    kens[i].append(ken)
    gohous[i].append(gohou)

# 結果のリストを表示
for i in range(4):
    print(f"df_{i}:")
    print("検出力:", np.mean(kens[i]))
    print("誤報率:", np.mean(gohous[i]))
```
