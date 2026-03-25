"""
M法（マハラノビス法）による異常検出パッケージ

離散変量を含むデータに対してマハラノビス距離ベースの異常検出を行う。
誤報率をαに合わせるようなK点を算出し、異常検出ルールを定める。
"""

from m_method.data_generator import (
    generate_unit_space_data,
    generate_anomaly_space_data,
    preprocess_data,
    preprocess_anomaly_data,
)
from m_method.mahalanobis import MahalanobisMethod
from m_method.visualizer import (
    plot_results,
    plot_distribution,
    plot_mahalanobis_distances,
    print_summary,
)

__all__ = [
    "generate_unit_space_data",
    "generate_anomaly_space_data",
    "preprocess_data",
    "preprocess_anomaly_data",
    "MahalanobisMethod",
    "plot_results",
    "plot_distribution",
    "plot_mahalanobis_distances",
    "print_summary",
]
