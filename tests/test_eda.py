import matplotlib
matplotlib.use("Agg")

from churn.eda import add_bin_features_cut, add_bin_features_qcut, analyze_churn_by_feature


def test_bin_features_add_column_in_place(toy_df):
    _, name_cut = add_bin_features_cut(toy_df, "tenure", 12)
    _, name_q = add_bin_features_qcut(toy_df, "MonthlyCharges", 5)
    assert name_cut in toy_df and name_q in toy_df
    assert toy_df[name_q].nunique() == 5


def test_analyze_churn_by_feature_rates(toy_df):
    res = analyze_churn_by_feature(toy_df, "Contract")
    assert res["总人数"].sum() == len(toy_df)
    assert ((res["流失率"] >= 0) & (res["流失率"] <= 1)).all()
