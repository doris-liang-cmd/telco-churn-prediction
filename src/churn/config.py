"""项目路径、随机种子与特征分组的统一配置。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SUBMISSION_DIR = ROOT / "submissions"
REPORT_DIR = ROOT / "reports"

SEED = 42
TARGET = "Churn"
ID_COL = "id"

# ---- 特征分组（与 notebooks/03_modeling.ipynb 保持一致）----
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]            # 数值特征
ONEHOT_COLS = ["MultipleLines", "InternetService", "Contract", "PaymentMethod"]  # 独热编码
BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"]     # 二值特征
SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
CROSS_COLS = [f"Fiber_No_{s}" for s in SERVICE_COLS]               # 交叉特征：光纤 × 未开通附加服务

# ---- 最终模型参数（Optuna 调参结果，见 notebooks/03_modeling.ipynb）----
XGB_PARAMS = {
    "max_depth": 3,
    "min_child_weight": 1.8499842251284397,
    "subsample": 0.9130491236562078,
    "colsample_bytree": 0.9458048670145641,
    "reg_lambda": 5.647395524814135,
    "reg_alpha": 0.002040115167270367,
    "gamma": 0.12964334666307942,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "eval_metric": "auc",
    "n_jobs": -1,
    "random_state": SEED,
}
LGB_PARAMS = {
    "num_leaves": 10,
    "min_child_samples": 85,
    "subsample": 0.9281577616934087,
    "subsample_freq": 1,
    "colsample_bytree": 0.5537981064820668,
    "reg_lambda": 35.50228641308715,
    "reg_alpha": 0.001317653231802295,
    "min_split_gain": 0.18102084748890865,
    "max_depth": -1,
    "learning_rate": 0.05,
    "n_jobs": -1,
    "random_state": SEED,
    "verbose": -1,
}
# 全量训练时的树数 = 5 折早停树数均值 × 1.1（来自 notebook 03 的 CV 结果）
XGB_N_ESTIMATORS = int(sum([3113, 2941, 2735, 2904, 2346]) / 5 * 1.1)
LGB_N_ESTIMATORS = int(sum([2256, 3092, 2436, 3112, 3013]) / 5 * 1.1)
