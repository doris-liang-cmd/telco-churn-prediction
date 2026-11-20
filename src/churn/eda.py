"""流失分析通用工具函数（EDA 用）。

用法::

    from churn.eda import (
        analyze_churn_by_feature,
        add_bin_features_cut,
        add_bin_features_qcut,
        analyze_interaction,
    )
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from IPython.display import display
except ImportError:
    display = print

_CJK_FONTS = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC",
              "WenQuanYi Micro Hei", "DejaVu Sans"]


def setup_plot_style() -> None:
    """设置中文字体（按系统可用字体回退）与负号显示。"""
    plt.rcParams["font.sans-serif"] = _CJK_FONTS
    plt.rcParams["axes.unicode_minus"] = False


setup_plot_style()

__all__ = [
    "analyze_churn_by_feature",
    "add_bin_features_cut",
    "add_bin_features_qcut",
    "analyze_interaction",
    "setup_plot_style",
]


def analyze_churn_by_feature(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """按单个特征统计各类别的人数、流失人数、流失率，并画出占比与流失率图。"""
    total = len(data)
    base_rate = (data["Churn"] == "Yes").mean()  # 整体流失率基线

    result = data.groupby(feature, observed=True)["Churn"].agg(
        总人数="count",
        流失人数=lambda x: (x == "Yes").sum(),
        流失率=lambda x: (x == "Yes").mean(),
    ).sort_values("流失率", ascending=False)
    result["占比"] = result["总人数"] / total
    result["流失相对倍数"] = result["流失率"] / base_rate

    print(f"特征 {feature} 与流失的数据统计（整体流失率 {base_rate:.2%}）:")
    display(result.round(4))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = result.index.astype(str)

    # 图1：各类别人数占比，基线 = 总人数（100%）
    sns.barplot(x=x, y=result["占比"].values, ax=axes[0])
    axes[0].axhline(1.0, color="red", linestyle="--",
                    label=f"总人数 {total}（100%）")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"{feature} 各类别人数占比")
    axes[0].set_xlabel(f"特征 {feature}")
    axes[0].set_ylabel("占总人数比例")
    axes[0].legend()

    # 图2：各类别流失率，基线 = 整体流失率
    sns.barplot(x=x, y=result["流失率"].values, ax=axes[1])
    axes[1].axhline(base_rate, color="red", linestyle="--",
                    label=f"整体流失率 {base_rate:.2%}")
    axes[1].set_title(f"{feature} 各类别流失率")
    axes[1].set_xlabel(f"特征 {feature}")
    axes[1].set_ylabel("流失率")
    axes[1].legend()

    for ax in axes:
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1%}",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.show()
    return result


def add_bin_features_cut(data: pd.DataFrame, feature: str, cut_size: int):
    """等宽分箱。注意：会**原地**在 data 中新增 `{feature}_bin` 列，并返回 (列, 列名)。"""
    mx = data[feature].max()
    bins = list(np.arange(0, mx + cut_size, cut_size))
    labels = [f"{bins[i]} ~ {bins[i + 1]}" for i in range(len(bins) - 1)]
    cut_res = pd.cut(data[feature], bins=bins, labels=labels, include_lowest=True)
    data[f"{feature}_bin"] = cut_res
    return data[f"{feature}_bin"], f"{feature}_bin"


def add_bin_features_qcut(data: pd.DataFrame, feature: str, q: int,
                          labels: list = None):
    """等频分箱。注意：会**原地**在 data 中新增 `{feature}_qcut` 列，并返回 (列, 列名)。"""
    qcut_res = pd.qcut(data[feature], q=q, labels=labels)  # qcut 默认已包含最小值，无需 include_lowest
    data[f"{feature}_qcut"] = qcut_res
    return data[f"{feature}_qcut"], f"{feature}_qcut"


def analyze_interaction(data: pd.DataFrame, full_data: pd.DataFrame,
                        title: str = "") -> None:
    """统计某个特征组合子集（data）的人数、占比、流失率，并与整体（full_data）对比作图。"""
    total = len(full_data)
    tot = len(data)
    proportion = tot / total
    churn_rate = (data["Churn"] == "Yes").mean()
    base_rate = (full_data["Churn"] == "Yes").mean()  # 整体流失率基线

    print(f"该组合总人数为：{tot}")
    print(f"人数占比为：{proportion:.2%}")
    print(f"该组合流失率为：{churn_rate:.2%}，整体流失率 {base_rate:.2%}，"
          f"相对倍数 {churn_rate / base_rate:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 图1：组合人数 vs 总人数
    sns.barplot(x=["组合人数", "总人数"], y=[tot, total], ax=axes[0])
    axes[0].set_title(f"{title} 人数对比（占比 {proportion:.2%}）")
    axes[0].set_ylabel("客户人数")
    for p in axes[0].patches:
        axes[0].annotate(f"{int(p.get_height())}",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom")

    # 图2：该组合的流失分布，并与整体流失率对比
    res = data["Churn"].value_counts(normalize=True).reindex(["No", "Yes"]).fillna(0)
    sns.barplot(x=res.index, y=res.values, ax=axes[1])
    axes[1].axhline(base_rate, color="red", linestyle="--",
                    label=f"整体流失率 {base_rate:.2%}")
    axes[1].set_title(f"{title} 流失情况")
    axes[1].set_ylabel("比例")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for p in axes[1].patches:
        axes[1].annotate(f"{p.get_height():.2%}",
                         (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha="center", va="bottom")

    plt.tight_layout()
    plt.show()