"""
M法（マハラノビス法）のコアロジックモジュール

誤報率をαに合わせるようなK点を算出し、
マハラノビス距離ベースの異常検出ルールを定める。
"""

import numpy as np
from scipy.stats import chi2


class MahalanobisMethod:
    def __init__(self, p, q, n_cells=4):
        """
        p: 単位空間の各cellの確率（リスト）
        q: 異常空間の各cellの確率（リスト）
        n_cells: cell数
        """
        self.p = p
        self.q = q
        self.n_cells = n_cells
        self.optimal_k = None

    def calculate_alpha(self, K, p):
        """4つのカイ二乗分布の右側面積の合計を計算する。

        各cellについて c = (1 - p[i]) / p[i] を求め、
        自由度2のカイ二乗分布で K - c における右側確率を算出し、
        p[i] で重み付けした合計を返す。

        Parameters
        ----------
        K : float
            閾値パラメータ
        p : list of float
            各cellの確率

        Returns
        -------
        float
            αの値（誤報率に相当）
        """
        alpha_list = []
        for i in range(self.n_cells):
            c = (1 - p[i]) / p[i]
            kai = 1 - chi2.cdf(K - c, df=2)
            alpha = p[i] * kai
            alpha_list.append(alpha)
        return sum(alpha_list)

    def find_optimal_k(self, target_alpha=0.05, initial_k=15,
                       step_size=0.1, max_iterations=100):
        """αに一致するK点を反復探索する。

        calculate_alpha で得られるαが target_alpha に十分近づくまで
        K を増減させる。

        Parameters
        ----------
        target_alpha : float
            目標とする誤報率α（デフォルト 0.05）
        initial_k : float
            Kの初期値（デフォルト 15）
        step_size : float
            各反復でのKの更新幅（デフォルト 0.1）
        max_iterations : int
            最大反復回数（デフォルト 100）

        Returns
        -------
        float
            最適なK値
        """
        K = initial_k

        for _ in range(max_iterations):
            current_alpha = self.calculate_alpha(K, self.p)

            if abs(current_alpha - target_alpha) < 0.0001:
                break

            if current_alpha > target_alpha:
                K += step_size
            if current_alpha < target_alpha:
                K -= step_size

        self.optimal_k = K
        return K

    def detect(self, train_con, test_con, K):
        """1つのcellに対する異常検出を行う。

        訓練データからマハラノビス距離の基準（平均・共分散逆行列）を算出し、
        テストデータの各サンプルについてマハラノビス距離を計算する。
        距離が K 以上のサンプルを異常と判定する。

        Parameters
        ----------
        train_con : array-like, shape (n_train, n_features)
            訓練データ（連続量）
        test_con : array-like, shape (n_test, n_features)
            テストデータ（連続量）
        K : float
            異常判定の閾値

        Returns
        -------
        dict
            - "distances": 各テストサンプルのマハラノビス距離 (np.ndarray)
            - "is_anomaly": 異常判定結果の真偽値 (np.ndarray)
            - "detection_rate": 検出率 (float)
            - "false_alarm_rate": 誤報率 (float)
        """
        train_con = np.asarray(train_con, dtype=float)
        test_con = np.asarray(test_con, dtype=float)

        # 訓練データの平均と共分散行列の逆行列を計算
        mean = np.mean(train_con, axis=0)
        cov = np.cov(train_con, rowvar=False)
        cov_inv = np.linalg.inv(cov)

        # テストデータのマハラノビス距離を計算
        diff = test_con - mean
        # 各サンプルについて (x - mu)^T * Sigma^{-1} * (x - mu)
        distances = np.array([
            d @ cov_inv @ d for d in diff
        ])

        is_anomaly = distances >= K

        n_test = len(test_con)
        detection_rate = np.sum(is_anomaly) / n_test if n_test > 0 else 0.0
        false_alarm_rate = detection_rate  # 単一cellでは検出率と同義

        return {
            "distances": distances,
            "is_anomaly": is_anomaly,
            "detection_rate": detection_rate,
            "false_alarm_rate": false_alarm_rate,
        }

    def evaluate(self, train_data, test_data, K, n_iterations=30):
        """全cellにわたる評価を反復実行する。

        各反復で train_data / test_data を生成し直す想定の呼び出し元から
        cell 毎の連続量データを受け取り、検出率と誤報率を集計する。

        Parameters
        ----------
        train_data : list of array-like
            各cellの訓練用連続量データのリスト（長さ n_cells）
        test_data : list of array-like
            各cellのテスト用連続量データのリスト（長さ n_cells）
        K : float
            異常判定の閾値
        n_iterations : int
            評価の反復回数（デフォルト 30）

        Returns
        -------
        dict
            - "detection_rates": cell毎の平均検出率リスト (list of float)
            - "false_alarm_rates": cell毎の平均誤報率リスト (list of float)
            - "mean_detection": 全cellの検出率の平均 (float)
            - "mean_false_alarm": 全cellの誤報率の平均 (float)
        """
        # cell毎の結果を蓄積するリスト
        detection_rates_per_cell = [[] for _ in range(self.n_cells)]
        false_alarm_rates_per_cell = [[] for _ in range(self.n_cells)]

        for _ in range(n_iterations):
            for i in range(self.n_cells):
                result = self.detect(train_data[i], test_data[i], K)
                detection_rates_per_cell[i].append(result["detection_rate"])
                false_alarm_rates_per_cell[i].append(result["false_alarm_rate"])

        # cell毎の平均を算出
        detection_rates = [
            float(np.mean(detection_rates_per_cell[i]))
            for i in range(self.n_cells)
        ]
        false_alarm_rates = [
            float(np.mean(false_alarm_rates_per_cell[i]))
            for i in range(self.n_cells)
        ]

        return {
            "detection_rates": detection_rates,
            "false_alarm_rates": false_alarm_rates,
            "mean_detection": float(np.mean(detection_rates)),
            "mean_false_alarm": float(np.mean(false_alarm_rates)),
        }
