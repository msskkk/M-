"""
M法（マハラノビス法）による異常検出 - メインスクリプト

離散変量が2つの場合のマハラノビス距離ベース異常検出を実行する。
誤報率αに合わせるK点を算出し、反復実験で検出力・誤報率を評価する。
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from m_method.data_generator import (
    generate_unit_space_data,
    generate_anomaly_space_data,
    preprocess_data,
    preprocess_anomaly_data,
)
from m_method.mahalanobis import MahalanobisMethod
from m_method.visualizer import plot_results, plot_distribution, print_summary


def main():
    # =========================================================================
    # パラメータ設定
    # =========================================================================
    n_un = 800
    n_an = 200

    cell = pd.DataFrame([[0, 0], [1, 0], [0, 1], [1, 1]])

    p = [0.25, 0.25, 0.25, 0.25]
    q = [0.25, 0.25, 0.25, 0.25]

    means_un = [[0, 0], [1, 1], [2, 2], [3, 3]]
    means_an = [[0, 2], [2, 2], [4, 4], [1, 5]]

    variances = [1.0, 1.0, 1.0, 1.0]

    covs_un = [
        [[variances[0], 0.8],  [0.8, variances[1]]],
        [[variances[1], -0.8], [-0.8, variances[1]]],
        [[variances[2], 0.8],  [0.8, variances[2]]],
        [[variances[3], -0.8], [-0.8, variances[3]]],
    ]
    covs_an = [
        [[variances[0], 0.8],  [0.8, variances[1]]],
        [[variances[1], -0.8], [-0.8, variances[1]]],
        [[variances[2], -0.8], [-0.8, variances[2]]],
        [[variances[3], 0.8],  [0.8, variances[3]]],
    ]

    target_alpha = 0.05
    n_experiments = 30
    n_cells = 4

    # =========================================================================
    # Step 1: 最適K点の算出
    # =========================================================================
    print("=" * 60)
    print("Step 1: 最適K点の算出")
    print("=" * 60)

    mm = MahalanobisMethod(p=p, q=q, n_cells=n_cells)

    optimal_k = mm.find_optimal_k(
        target_alpha=target_alpha,
        initial_k=15,
        step_size=0.1,
        max_iterations=100,
    )

    achieved_alpha = mm.calculate_alpha(optimal_k, p)
    print(f"  最適K点: {optimal_k:.4f}")
    print(f"  達成誤報率: {achieved_alpha:.6f}")
    print(f"  目標誤報率: {target_alpha}")
    print()

    # =========================================================================
    # Step 2: 反復実験
    # =========================================================================
    print("=" * 60)
    print(f"Step 2: 反復実験 ({n_experiments} 回)")
    print("=" * 60)

    kens = [[] for _ in range(n_cells)]
    gohous = [[] for _ in range(n_cells)]
    detcounts = [[] for _ in range(n_cells)]
    sigcounts = [[] for _ in range(n_cells)]

    for iteration in tqdm(range(n_experiments), desc="実験進行中"):
        df0 = generate_unit_space_data(n=n_un, p=p, means=means_un, covs=covs_un)
        df1 = generate_anomaly_space_data(n=n_an, q=q, means=means_an, covs=covs_an)

        unit_data = preprocess_data(df0, cell)
        anomaly_data = preprocess_anomaly_data(df1, cell)

        for i in range(n_cells):
            train_con = unit_data[f"df_{i}_con"]
            test_con = anomaly_data[f"df_{i}_con"]

            result = mm.detect(train_con, test_con, optimal_k)

            kens[i].append(result["detection_rate"])
            detcounts[i].append(int(np.sum(result["is_anomaly"])))
            gohous[i].append(result["false_alarm_rate"])
            sigcounts[i].append(int(np.sum(result["is_anomaly"])))

    # =========================================================================
    # Step 3: 結果表示
    # =========================================================================
    print()
    print("=" * 60)
    print("Step 3: 実験結果")
    print("=" * 60)

    results = {}
    for i in range(n_cells):
        results[f"cell_{i}"] = {
            "detection_rates": kens[i],
            "false_alarm_rates": gohous[i],
        }
        print(f"\n  df_{i}:")
        print(f"    検出力:   {np.mean(kens[i]):.4f}")
        print(f"    検出個数: {np.mean(detcounts[i]):.2f}")
        print(f"    誤報率:   {np.mean(gohous[i]):.4f}")
        print(f"    誤報個数: {np.mean(sigcounts[i]):.2f}")

    # =========================================================================
    # Step 4: 可視化
    # =========================================================================
    print()
    print("=" * 60)
    print("Step 4: サマリー")
    print("=" * 60)

    print_summary(results)
    print("\n完了しました。")


if __name__ == "__main__":
    main()
