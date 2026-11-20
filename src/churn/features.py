"""特征工程与预处理管道。"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import BINARY_COLS, CROSS_COLS, NUM_COLS, ONEHOT_COLS, SERVICE_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """二值映射 + 交叉特征（光纤 且 未开通某附加服务）。不修改入参。"""
    df = df.copy()
    for c in BINARY_COLS[1:]:  # SeniorCitizen 原本就是 0/1
        df[c] = df[c].map({"Yes": 1, "No": 0})
    for s in SERVICE_COLS:
        df[f"Fiber_No_{s}"] = (
            (df["InternetService"] == "Fiber optic") & (df[s] == "No")
        ).astype(int)
    return df


def build_pipelines() -> tuple[Pipeline, Pipeline]:
    """返回 (线性模型用管道, 树模型用管道)。线性模型额外对数值列标准化。"""
    def _prep(scale: bool) -> ColumnTransformer:
        return ColumnTransformer([
            ("num", StandardScaler() if scale else "passthrough", NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ONEHOT_COLS),
            ("bin", "passthrough", BINARY_COLS + CROSS_COLS),
        ])

    linear = Pipeline([("feat", FunctionTransformer(add_features)), ("prep", _prep(True))])
    tree = Pipeline([("feat", FunctionTransformer(add_features)), ("prep", _prep(False))])
    return linear, tree
