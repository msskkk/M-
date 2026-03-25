"""M法（マハラノビス法）の結果可視化モジュール"""

import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
except Exception:
    try:
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    except Exception:
        # フォントが見つからない場合はデフォルトを使用
        pass


def plot_results(results, save_path=None):
    """全cellの検出力と誤報率を棒グラフで表示する。

    Parameters
    ----------
    results : dict
        {"cell_0": {"detection_rates": [...], "false_alarm_rates": [...]}, ...}
    save_path : str or None
        指定時はファイルに保存する。
    """
    cell_names = sorted(results.keys())
    detection_means = [np.mean(results[c]["detection_rates"]) for c in cell_names]
    false_alarm_means = [np.mean(results[c]["false_alarm_rates"]) for c in cell_names]

    x = np.arange(len(cell_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, detection_means, width, label="検出力", color="steelblue")
    bars2 = ax.bar(x + width / 2, false_alarm_means, width, label="誤報率", color="salmon")

    ax.set_xlabel("Cell")
    ax.set_ylabel("割合")
    ax.set_title("各Cellの検出力と誤報率")
    ax.set_xticks(x)
    ax.set_xticklabels(cell_names)
    ax.legend()
    ax.set_ylim(0, 1.05)

    # バーの上に数値を表示
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.2f}",
                ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_distribution(train_data, test_data, cell_index, save_path=None):
    """特定cellの訓練データと異常データの散布図を重ねて表示する。

    Parameters
    ----------
    train_data : array-like
        訓練データ（y1, y2 の2列を持つ）。
    test_data : array-like
        異常データ（y1, y2 の2列を持つ）。
    cell_index : int
        対象のcell番号。
    save_path : str or None
        指定時はファイルに保存する。
    """
    train_data = np.asarray(train_data)
    test_data = np.asarray(test_data)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(train_data[:, 0], train_data[:, 1],
               c="blue", alpha=0.5, label="訓練データ（正常）", s=20)
    ax.scatter(test_data[:, 0], test_data[:, 1],
               c="red", alpha=0.5, label="異常データ", s=20)

    ax.set_xlabel("y1")
    ax.set_ylabel("y2")
    ax.set_title(f"Cell {cell_index} のデータ分布")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_mahalanobis_distances(distances, K, cell_index, save_path=None):
    """マハラノビス距離のヒストグラムを表示する。

    Parameters
    ----------
    distances : array-like
        マハラノビス距離の配列。
    K : float
        閾値。
    cell_index : int
        対象のcell番号。
    save_path : str or None
        指定時はファイルに保存する。
    """
    distances = np.asarray(distances)

    fig, ax = plt.subplots(figsize=(8, 5))
    n_bins = max(10, int(np.sqrt(len(distances))))
    counts, bins, patches = ax.hist(distances, bins=n_bins, color="steelblue",
                                     edgecolor="white", alpha=0.7,
                                     label="マハラノビス距離")

    # 閾値の縦線
    ax.axvline(x=K, color="red", linestyle="--", linewidth=2, label=f"閾値 K={K:.2f}")

    # 閾値以上の領域を薄い赤で塗りつぶし
    ax.axvspan(K, ax.get_xlim()[1], color="red", alpha=0.15)

    ax.set_xlabel("マハラノビス距離")
    ax.set_ylabel("頻度")
    ax.set_title(f"Cell {cell_index} のマハラノビス距離分布")
    ax.legend()

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_summary(results):
    """各cellの検出力・誤報率をコンソールに表示する。

    Parameters
    ----------
    results : dict
        {"cell_0": {"detection_rates": [...], "false_alarm_rates": [...]}, ...}

    出力フォーマット例:
        df_0: 検出力: 0.85, 誤報率: 0.04
    """
    for key in sorted(results.keys()):
        # cell_0 -> df_0
        idx = key.split("_", 1)[1]
        det_mean = np.mean(results[key]["detection_rates"])
        fa_mean = np.mean(results[key]["false_alarm_rates"])
        print(f"df_{idx}: 検出力: {det_mean:.2f}, 誤報率: {fa_mean:.2f}")
