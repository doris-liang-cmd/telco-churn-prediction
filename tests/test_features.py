import pandas as pd

from churn.config import CROSS_COLS
from churn.features import add_features, build_pipelines


def test_add_features_does_not_mutate_input(toy_df):
    before = toy_df.copy()
    add_features(toy_df)
    pd.testing.assert_frame_equal(toy_df, before)


def test_binary_mapping_and_cross_features(toy_df):
    out = add_features(toy_df)
    assert set(out["Partner"].unique()) <= {0, 1}
    assert set(out[CROSS_COLS].stack().unique()) <= {0, 1}
    # 交叉特征：光纤且未开通 OnlineSecurity
    expected = ((toy_df["InternetService"] == "Fiber optic")
                & (toy_df["OnlineSecurity"] == "No")).astype(int)
    pd.testing.assert_series_equal(out["Fiber_No_OnlineSecurity"], expected,
                                   check_names=False)


def test_pipelines_output_shape_and_unseen_category(toy_df):
    X = toy_df.drop(columns="Churn")
    linear, tree = build_pipelines()
    Xl, Xt = linear.fit_transform(X), tree.fit_transform(X)
    assert Xl.shape == Xt.shape
    unseen = X.head(3).copy()
    unseen["Contract"] = "Three year"        # 训练中没出现的类别
    assert tree.transform(unseen).shape[1] == Xt.shape[1]
