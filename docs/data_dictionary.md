# 数据字段说明

> 根据 notebook 中出现的列整理。

| 字段 | 类型 | 取值 / 含义 |
|---|---|---|
| id | 整数 | 客户编号（建模时不使用） |
| gender | 类别 | Male / Female |
| SeniorCitizen | 0/1 | 是否老年客户 |
| Partner | Yes/No | 是否有伴侣 |
| Dependents | Yes/No | 是否有受抚养人 |
| tenure | 整数 | 在网时长（月） |
| PhoneService | Yes/No | 是否开通电话服务 |
| MultipleLines | 类别 | Yes / No / No phone service |
| InternetService | 类别 | DSL / Fiber optic / No |
| OnlineSecurity、OnlineBackup、DeviceProtection、TechSupport、StreamingTV、StreamingMovies | 类别 | Yes / No / No internet service |
| Contract | 类别 | Month-to-month / One year / Two year |
| PaperlessBilling | Yes/No | 是否无纸化账单 |
| PaymentMethod | 类别 | 4 种付款方式（含 Electronic check） |
| MonthlyCharges | 浮点 | 月费用 |
| TotalCharges | 浮点 | 总费用 |
| **Churn** | Yes/No | **目标变量：是否流失** |
