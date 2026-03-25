"""M法（マハラノビス法）のデータ生成・前処理モジュール."""

import numpy as np
import pandas as pd


def generate_unit_space_data(
    n: int,
    p: list[float],
    means: list[list[float]],
    covs: list[list[list[float]]],
) -> pd.DataFrame:
    """単位空間データを生成する.

    離散変量 (x1, x2) と連続変量 (y1, y2) を含む DataFrame を返す。
    各セルのサンプル数は n * p[i] で決まる。

    Parameters
    ----------
    n : int
        総サンプル数.
    p : list[float]
        各セルの混合比率 (長さ4).
    means : list[list[float]]
        各セルの連続変量の平均ベクトル (長さ4, 各要素は長さ2).
    covs : list[list[list[float]]]
        各セルの連続変量の共分散行列 (長さ4, 各要素は 2x2).

    Returns
    -------
    pd.DataFrame
        列 ['x1', 'x2', 'y1', 'y2'] を持つ DataFrame.
    """
    # セルごとの離散変量パターン: (x1, x2)
    cell_patterns = [(0, 0), (1, 0), (0, 1), (1, 1)]
    combined_data = []

    for i in range(4):
        n_cell = int(n * p[i])
        x1_val, x2_val = cell_patterns[i]
        x_1 = np.full(n_cell, x1_val, dtype=float)
        x_2 = np.full(n_cell, x2_val, dtype=float)

        y1, y2 = np.random.multivariate_normal(means[i], covs[i], n_cell).T
        data = np.column_stack((x_1, x_2, y1, y2))
        combined_data.append(data)

    df = pd.DataFrame(np.vstack(combined_data), columns=["x1", "x2", "y1", "y2"])
    return df


def generate_anomaly_space_data(
    n: int,
    q: list[float],
    means: list[list[float]],
    covs: list[list[list[float]]],
) -> pd.DataFrame:
    """異常空間データを生成する.

    構造は generate_unit_space_data と同じ。

    Parameters
    ----------
    n : int
        総サンプル数.
    q : list[float]
        各セルの混合比率 (長さ4).
    means : list[list[float]]
        各セルの連続変量の平均ベクトル.
    covs : list[list[list[float]]]
        各セルの連続変量の共分散行列.

    Returns
    -------
    pd.DataFrame
        列 ['x1', 'x2', 'y1', 'y2'] を持つ DataFrame.
    """
    cell_patterns = [(0, 0), (1, 0), (0, 1), (1, 1)]
    combined_data = []

    for i in range(4):
        n_cell = int(n * q[i])
        x1_val, x2_val = cell_patterns[i]
        x_1 = np.full(n_cell, x1_val, dtype=float)
        x_2 = np.full(n_cell, x2_val, dtype=float)

        y1, y2 = np.random.multivariate_normal(means[i], covs[i], n_cell).T
        data = np.column_stack((x_1, x_2, y1, y2))
        combined_data.append(data)

    df = pd.DataFrame(np.vstack(combined_data), columns=["x1", "x2", "y1", "y2"])
    return df


def preprocess_data(df: pd.DataFrame, cell: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """単位空間 DataFrame をセル番号ごとに分割し、連続量データを辞書で返す.

    Parameters
    ----------
    df : pd.DataFrame
        generate_unit_space_data で生成した DataFrame.
    cell : pd.DataFrame
        セル定義 DataFrame (shape: (4, 2)).
        各行が (x1, x2) の離散変量パターンに対応する。

    Returns
    -------
    dict[str, pd.DataFrame]
        キー "df_0_con", "df_1_con", ... に対応する連続量 (y1, y2) の DataFrame.
    """
    df = df.copy()
    df["cell番号"] = -1

    for j in range(len(df)):
        for i in range(len(cell)):
            if df.iloc[j, 0] == cell.iloc[i, 0] and df.iloc[j, 1] == cell.iloc[i, 1]:
                df.loc[df.index[j], "cell番号"] = i

    result = {}
    for i in range(len(cell)):
        cell_df = df[df["cell番号"] == i].drop("cell番号", axis=1)
        continuous_cols = list(cell_df.columns[2:])  # y1, y2
        result[f"df_{i}_con"] = cell_df[continuous_cols].copy().reset_index(drop=True)

    return result


def preprocess_anomaly_data(
    df: pd.DataFrame, cell: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """異常空間（テスト）DataFrame をセル番号ごとに分割し、連続量データを辞書で返す.

    Parameters
    ----------
    df : pd.DataFrame
        generate_anomaly_space_data で生成した DataFrame.
    cell : pd.DataFrame
        セル定義 DataFrame (shape: (4, 2)).

    Returns
    -------
    dict[str, pd.DataFrame]
        キー "df_0_con", "df_1_con", ... に対応する連続量 (y1, y2) の DataFrame.
    """
    df = df.copy()
    df["cell番号"] = -1

    for j in range(len(df)):
        for i in range(len(cell)):
            if df.iloc[j, 0] == cell.iloc[i, 0] and df.iloc[j, 1] == cell.iloc[i, 1]:
                df.loc[df.index[j], "cell番号"] = i

    result = {}
    for i in range(len(cell)):
        cell_df = df[df["cell番号"] == i].drop("cell番号", axis=1)
        continuous_cols = list(cell_df.columns[2:])  # y1, y2
        result[f"df_{i}_con"] = cell_df[continuous_cols].copy().reset_index(drop=True)

    return result
