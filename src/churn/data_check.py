"""数据质量检查：数值分布，以及 TotalCharges 与 tenure × MonthlyCharges 的一致性。"""
import pandas as pd

try:
    from IPython.display import display
except ImportError:  # 脚本环境下退化为 print
    display = print


def check_numeric_features(data: pd.DataFrame) -> None:
    excludes = ["id", "SeniorCitizen"]  # 无关数值
    num_cols = data.select_dtypes("number").columns.difference(excludes).tolist()
    print("数值数据分布:")
    display(data[num_cols].describe())
    print()

    gap = data["TotalCharges"] - data["tenure"] * data["MonthlyCharges"]
    print("绝对偏差分布 (TotalCharges - tenure × MonthlyCharges):")
    display(gap.describe())
    print()

    rel = gap.abs() / data["TotalCharges"]
    print("相对偏差率分布:")
    display(rel.describe())

    print("\n各阈值下的人数分布")
    thresholds = [0.1, 0.2, 0.5, 1, 5]
    tab = pd.DataFrame({
        "相对偏差率": thresholds,
        "人数": [(rel > t).sum() for t in thresholds],
        "占比": [(rel > t).mean() for t in thresholds],
    })
    display(tab.style.hide(axis="index").format({"人数": "{:,}", "占比": "{:.2%}"}))
