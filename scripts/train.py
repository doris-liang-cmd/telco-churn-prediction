"""复现最终结果：XGBoost + LightGBM 概率平均，输出 submissions/submission.csv。

用法::

    python scripts/train.py --validate      # 先在 20% 留出集上验证 AUC
    python scripts/train.py                 # 全量训练并生成提交文件
    python scripts/train.py --fast          # 小树数冒烟测试（几秒钟）
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from churn.config import (
    ID_COL, LGB_N_ESTIMATORS, LGB_PARAMS, RAW_DIR, SEED, SUBMISSION_DIR,
    TARGET, XGB_N_ESTIMATORS, XGB_PARAMS,
)
from churn.features import build_pipelines


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=RAW_DIR, help="包含 train.csv / test.csv 的目录")
    p.add_argument("--out", type=Path, default=SUBMISSION_DIR / "submission.csv")
    p.add_argument("--validate", action="store_true", help="仅在 80/20 留出集上评估，不生成提交文件")
    p.add_argument("--fast", action="store_true", help="冒烟测试：把树数降到 50")
    return p.parse_args()


def fit_models(X, y, n_xgb: int, n_lgb: int):
    xgb = XGBClassifier(**XGB_PARAMS, n_estimators=n_xgb).fit(X, y)
    lgb = LGBMClassifier(**LGB_PARAMS, n_estimators=n_lgb).fit(X, y)
    return xgb, lgb


def predict_blend(models, X) -> np.ndarray:
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def main() -> None:
    args = parse_args()
    n_xgb, n_lgb = (50, 50) if args.fast else (XGB_N_ESTIMATORS, LGB_N_ESTIMATORS)

    train = pd.read_csv(args.data_dir / "train.csv")
    _, tree_pipe = build_pipelines()

    X = train.drop(columns=TARGET)
    y = train[TARGET].map({"Yes": 1, "No": 0}).values

    if args.validate:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=SEED, stratify=y)
        Xm_tr = tree_pipe.fit_transform(X_tr)
        Xm_val = tree_pipe.transform(X_val)
        models = fit_models(Xm_tr, y_tr, n_xgb, n_lgb)
        for name, m in zip(["XGBoost", "LightGBM"], models):
            print(f"{name:9s} 留出集 AUC = {roc_auc_score(y_val, m.predict_proba(Xm_val)[:, 1]):.5f}")
        print(f"{'Blend':9s} 留出集 AUC = {roc_auc_score(y_val, predict_blend(models, Xm_val)):.5f}")
        return

    test = pd.read_csv(args.data_dir / "test.csv")
    X_full = tree_pipe.fit_transform(X)       # 预处理只在训练集上 fit，测试集只 transform
    X_test = tree_pipe.transform(test)
    assert X_full.shape[1] == X_test.shape[1], "训练/测试特征维度不一致"

    models = fit_models(X_full, y, n_xgb, n_lgb)
    pred = predict_blend(models, X_test)

    sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred})
    sample_path = args.data_dir / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        assert (sample[ID_COL].values == test[ID_COL].values).all(), "id 顺序与 sample_submission 不一致"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(f"已写入 {args.out}  形状={sub.shape}  缺失={sub[TARGET].isna().sum()}")


if __name__ == "__main__":
    main()
