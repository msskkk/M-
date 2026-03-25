"""M法（マハラノビス法）プロジェクトのテストスイート."""

import numpy as np
import pandas as pd
import pytest
from m_method.data_generator import (
    generate_unit_space_data,
    generate_anomaly_space_data,
    preprocess_data,
    preprocess_anomaly_data,
)
from m_method.mahalanobis import MahalanobisMethod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def params():
    """テスト用の共通パラメータを返す."""
    p = [0.25, 0.25, 0.25, 0.25]
    q = [0.25, 0.25, 0.25, 0.25]
    means_un = [[0, 0], [1, 1], [2, 2], [3, 3]]
    means_an = [[0, 2], [2, 2], [4, 4], [1, 5]]
    variances = [1.0, 1.0, 1.0, 1.0]
    covs_un = [
        [[variances[0], 0.8], [0.8, variances[1]]],
        [[variances[1], -0.8], [-0.8, variances[1]]],
        [[variances[2], 0.8], [0.8, variances[2]]],
        [[variances[3], -0.8], [-0.8, variances[3]]],
    ]
    covs_an = [
        [[variances[0], 0.8], [0.8, variances[1]]],
        [[variances[1], -0.8], [-0.8, variances[1]]],
        [[variances[2], -0.8], [-0.8, variances[2]]],
        [[variances[3], 0.8], [0.8, variances[3]]],
    ]
    cell = pd.DataFrame([[0, 0], [1, 0], [0, 1], [1, 1]])
    return {
        "p": p,
        "q": q,
        "means_un": means_un,
        "means_an": means_an,
        "covs_un": covs_un,
        "covs_an": covs_an,
        "cell": cell,
    }


@pytest.fixture()
def unit_space_df(params):
    """再現性のある単位空間データを生成する."""
    np.random.seed(42)
    return generate_unit_space_data(
        n=100, p=params["p"], means=params["means_un"], covs=params["covs_un"]
    )


@pytest.fixture()
def anomaly_space_df(params):
    """再現性のある異常空間データを生成する."""
    np.random.seed(42)
    return generate_anomaly_space_data(
        n=100, q=params["q"], means=params["means_an"], covs=params["covs_an"]
    )


@pytest.fixture()
def mm(params):
    """MahalanobisMethod インスタンスを返す."""
    return MahalanobisMethod(p=params["p"], q=params["q"])


# ---------------------------------------------------------------------------
# データ生成テスト
# ---------------------------------------------------------------------------

class TestDataGeneration:
    """データ生成・前処理に関するテスト."""

    def test_generate_unit_space_data_shape(self, unit_space_df):
        """n=100, 4cell, 各cell25%の場合、合計100行x4列のDataFrame."""
        assert unit_space_df.shape == (100, 4)

    def test_generate_unit_space_data_columns(self, unit_space_df):
        """カラム名が ['x1', 'x2', 'y1', 'y2'] であること."""
        assert list(unit_space_df.columns) == ["x1", "x2", "y1", "y2"]

    def test_preprocess_data_returns_dict(self, unit_space_df, params):
        """preprocess_data が辞書を返し、キーに 'df_0_con' 等を含むこと."""
        result = preprocess_data(unit_space_df, params["cell"])
        assert isinstance(result, dict)
        for i in range(4):
            assert f"df_{i}_con" in result

    def test_preprocess_data_continuous_columns(self, unit_space_df, params):
        """連続量データが 'y1', 'y2' の2列であること."""
        result = preprocess_data(unit_space_df, params["cell"])
        for i in range(4):
            df_con = result[f"df_{i}_con"]
            assert list(df_con.columns) == ["y1", "y2"]
            assert df_con.shape[1] == 2


# ---------------------------------------------------------------------------
# アルゴリズムテスト
# ---------------------------------------------------------------------------

class TestMahalanobisAlgorithm:
    """MahalanobisMethod のアルゴリズムに関するテスト."""

    def test_calculate_alpha_positive(self, mm, params):
        """calculate_alpha が正の値を返すこと."""
        alpha = mm.calculate_alpha(K=9, p=params["p"])
        assert alpha > 0

    def test_calculate_alpha_decreases_with_k(self, mm, params):
        """K が大きいほど alpha が小さくなること."""
        alpha_small_k = mm.calculate_alpha(K=5, p=params["p"])
        alpha_large_k = mm.calculate_alpha(K=15, p=params["p"])
        assert alpha_small_k > alpha_large_k

    def test_find_optimal_k(self, mm, params):
        """find_optimal_k の結果で calculate_alpha が 0.05 に近いこと (tolerance=0.001)."""
        optimal_k = mm.find_optimal_k(target_alpha=0.05)
        achieved_alpha = mm.calculate_alpha(optimal_k, params["p"])
        assert abs(achieved_alpha - 0.05) < 0.001

    def test_detect_returns_expected_keys(self, mm, unit_space_df, anomaly_space_df, params):
        """detect の返り値に必要なキーが含まれること."""
        np.random.seed(42)
        train_result = preprocess_data(unit_space_df, params["cell"])
        test_result = preprocess_anomaly_data(anomaly_space_df, params["cell"])

        train_con = train_result["df_0_con"]
        test_con = test_result["df_0_con"]

        result = mm.detect(train_con, test_con, K=9)

        expected_keys = {"distances", "is_anomaly", "detection_rate", "false_alarm_rate"}
        assert expected_keys == set(result.keys())
