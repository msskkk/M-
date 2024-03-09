# M法-
誤報率をαに合わせるようなK点を算出し、異常検出ルールを定める

```
mport pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import chi2
import matplotlib.pyplot as plt
#離散変量が2つの場合
n_un=800
n_an=200
means_un=[[0,0], [1, 1], [2, 2], [3, 3]]
means_an=[[0,2], [2, 2], [4, 4], [1, 5]]
cell= [[0,0], [1,0],[0,1],[1,1]]
p=[0.25, 0.25, 0.25, 0.25]
q=[0.25, 0.25, 0.25, 0.25]
cell=pd.DataFrame(cell)
# 分散パラメータ
variances = [1.0, 1.0, 1.0, 1.0]  # ここに各変数の分散を設定
# 相関行列
covs_un = [
    [[variances[0], 0.8], [0.8, variances[1]]],
    [[variances[1], -0.8], [-0.8, variances[1]]],
    [[variances[2], 0.8], [0.8, variances[2]]],
    [[variances[3], -0.8], [-0.8, variances[3]]]
]

covs_an = [
    [[variances[0], 0.8], [0.8, variances[1]]],
    [[variances[1], -0.8], [-0.8, variances[1]]],
    [[variances[2], -0.8], [-0.8, variances[2]]],
    [[variances[3], 0.8], [0.8, variances[3]]]
]
```

```
def generate_unit_space_data(n=n_un, p=p, means=means_un, covs=covs_un):

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
def generate_anomaly_space_data(n=n_an, p=p, means=means_an, covs=covs_an):
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
```
４つのカイ二乗分布の右側面積の合計がαに一致するようなKを探す
```
import scipy.stats as sps
#初期値K
K=9
cc=[]
kai_list=[]
alpha_list=[]
for i in range(4):
  c=(1-p[i])/p[i]
  kai=1-sps.chi2.cdf(K-c, df = 2)
  alpha=p[i]*kai
  kai_list.append(kai)
  cc.append(c)
  alpha_list.append(alpha)
sum(alpha_list)
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
max_iterations =100

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
detcounts = [[] for _ in range(4)]
sigcounts = [[] for _ in range(4)]

j=30

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
    detcount = mt.det_count()
    sigcount = mt.sig_count()

    gohou = mt.mh_sig()
    kens[i].append(ken)
    detcounts[i].append(detcount)
    gohous[i].append(gohou)
    sigcounts[i].append(sigcount)

# 結果のリストを表示
for i in range(4):
    print(f"df_{i}:")
    print("検出力:", np.mean(kens[i]))
    print("検出個数:", np.mean(detcounts[i]))
    print("誤報率:", np.mean(gohous[i]))
    print("誤報個数:", np.mean(sigcounts[i]))
```
