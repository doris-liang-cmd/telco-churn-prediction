# 电信客户流失预测（Customer Churn Prediction）

基于 59 万条客户记录，完成 **探索性分析 → 特征工程 → 多模型对比 → Optuna 调参 → XGBoost + LightGBM 融合** 的完整建模流程，并将可复现的训练流程封装为 Python 包与命令行脚本。

## 结果概览

| 项目 | 数值 |
|---|---|
| 训练 / 测试样本 | 594,194 / 254,655 |
| 整体流失率 | 22.52% |
| 评估指标 | ROC-AUC（另报告 Average Precision、F1） |
| 5 折 CV AUC（XGB + LGB 平均融合） | **0.91626** |
| 20% 留出集 AUC（融合） | **0.91682** |
| 线上榜单得分 |**0.91504** |

**模型对比（80% 训练集上 5 折分层 CV，默认参数初筛）**

| 模型 | ROC-AUC | Average Precision |
|---|---|---|
| Dummy 基线 | 0.5000 | 0.2252 |
| 逻辑回归 | 0.9067 | 0.7220 |
| 决策树 | 0.9083 | 0.7246 |
| AdaBoost | 0.9095 | 0.7315 |
| 随机森林 | 0.9127 | 0.7443 |
| HistGradientBoosting | 0.9146 | 0.7499 |
| LightGBM | 0.9147 | 0.7501 |
| XGBoost | 0.9150 | 0.7514 |

调参后 XGBoost / LightGBM 的 CV AUC 均约 0.9161；两者 OOF 预测相关系数 0.9976，平均融合小幅提升至 0.9163。逻辑回归加入融合后 AUC 反而下降（权重越大越低），因此最终不采用。

## EDA 主要发现

- **合同类型**影响最大：按月合同流失率 42.05%，一年合同 5.76%，两年合同 1.00%；按月合同的流失集中在在网前 18 个月（约占全部流失客户的 68%）。
- **光纤用户**流失率 41.54%（DSL 10.31%）；**电子支票**用户 48.91%。
- 光纤 × 按月合同、电子支票 × 按月合同的组合流失率均约 55%（整体的 2.4 倍以上）；非光纤的按月合同流失率仅 18%，说明"按月付"的高流失主要来自光纤用户。
- 未开通 OnlineSecurity / TechSupport 等附加服务的高流失，大部分由"光纤用户本身高流失"造成（混杂），因此构造了"光纤 且 未开通某服务"的交叉特征。
- `TotalCharges` 与 `tenure` 强相关（Spearman 0.84），线性模型需注意共线性。

完整分析见 `notebooks/01_eda_univariate.ipynb`、`notebooks/02_eda_interaction.ipynb`。

## 项目结构

```
.
├── README.md
├── pyproject.toml / requirements.txt
├── data/
│   ├── README.md              # 数据放置说明（数据本身不入库）
│   └── raw/                   # train.csv / test.csv / sample_submission.csv
├── docs/data_dictionary.md    # 字段说明
├── notebooks/
│   ├── 01_eda_univariate.ipynb    # 单变量分析
│   ├── 02_eda_interaction.ipynb   # 特征联合分析
│   └── 03_modeling.ipynb          # 建模、调参、融合
├── src/churn/
│   ├── config.py              # 路径、随机种子、特征分组、最终超参
│   ├── features.py            # 特征工程 + sklearn 预处理管道
│   ├── eda.py                 # EDA 作图/统计工具函数
│   └── data_check.py          # 数据质量检查
├── scripts/train.py           # 一键复现：验证 / 训练 / 生成提交文件
├── tests/                     # pytest 单元测试
├── reports/                   # 分析报告与图表
└── submissions/               # 生成的提交文件（不入库）
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # 同时以可编辑模式安装本项目 (churn 包)

# 将数据放入 data/raw/ 后：
python scripts/train.py --validate   # 20% 留出集上评估
python scripts/train.py              # 全量训练，输出 submissions/submission.csv
pytest                               # 运行测试
jupyter lab                          # 打开 notebooks/
```

## 方法说明

1. **预处理**：数值列（线性模型标准化 / 树模型不缩放）、4 个多值类别列独热编码、二值列映射为 0/1。
2. **特征工程**：`Fiber_No_{OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport}`，即"光纤用户且未开通该服务"。
3. **验证**：80/20 分层划分（`random_state=42`）；80% 内做 5 折分层 CV 用于调参与融合权重选择，20% 留出集仅用于最终评估。
4. **调参**：Optuna（TPE + MedianPruner），学习率固定 0.08 搜索结构参数，最终用 0.05 配合早停确定树数；全量训练时树数取 CV 平均早停轮数 × 1.1。
5. **融合**：XGBoost 与 LightGBM 预测概率简单平均。
6. **防泄漏**：预处理器只在训练数据上 `fit`，测试集仅 `transform`。

## 已知局限

- XGBoost 的 Optuna 搜索计划 60 轮，实际完成 48 轮后手动中断，最终参数取已完成试验中的最优。
- 特征工程目前只验证了"光纤 × 附加服务"一类交叉特征；按月合同 × 光纤/电子支票等组合特征留待后续消融实验验证增量。
