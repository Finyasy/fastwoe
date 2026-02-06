from __future__ import annotations

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def test_mapping_includes_positive_woe_standard_error() -> None:
    model = FastWoe()
    model.fit(["A", "A", "B", "C", "C"], [1, 0, 1, 0, 1])
    mapping = model.get_mapping()
    assert len(mapping) >= 2
    assert all(row.woe_se > 0.0 for row in mapping)


def test_binary_iv_analysis_and_dataframe_mode() -> None:
    rows = [
        ["A", "x"],
        ["A", "y"],
        ["B", "x"],
        ["C", "z"],
        ["C", "y"],
    ]
    y = [1, 0, 0, 1, 1]
    model = FastWoe()
    model.fit_matrix(rows, y, feature_names=["cat", "bucket"])

    iv_rows = model.get_iv_analysis()
    assert len(iv_rows) == 2
    assert all(r.iv >= 0.0 for r in iv_rows)
    assert all(r.iv_ci_lower <= r.iv <= r.iv_ci_upper for r in iv_rows)

    pd = pytest.importorskip("pandas")
    iv_df = model.get_iv_analysis(as_frame=True)
    assert isinstance(iv_df, pd.DataFrame)
    assert list(iv_df.columns) == [
        "feature",
        "iv",
        "iv_se",
        "iv_ci_lower",
        "iv_ci_upper",
        "iv_significance",
    ]


def test_multiclass_iv_analysis_per_class() -> None:
    rows = [
        ["A", "x"],
        ["A", "y"],
        ["B", "x"],
        ["C", "z"],
        ["C", "y"],
        ["B", "z"],
    ]
    labels = ["c0", "c1", "c2", "c0", "c1", "c2"]
    model = FastWoe()
    model.fit_matrix_multiclass(rows, labels, feature_names=["cat", "bucket"])

    iv_rows = model.get_iv_analysis_multiclass("c0")
    assert len(iv_rows) == 2
    assert all(r.iv >= 0.0 for r in iv_rows)

    iv_single = model.get_iv_analysis_multiclass("c0", feature_name="cat")
    assert len(iv_single) == 1
    assert iv_single[0].feature == "cat"


def test_ci_interval_shrinks_with_more_training_data() -> None:
    small_rows = [["A"], ["A"], ["B"], ["C"]]
    small_y = [1, 0, 0, 1]

    model_small = FastWoe()
    model_small.fit_matrix(small_rows, small_y, feature_names=["cat"])
    ci_small = model_small.predict_ci_matrix([["A"]], alpha=0.05)[0]
    width_small = ci_small[2] - ci_small[1]

    big_rows = small_rows * 25
    big_y = small_y * 25
    model_big = FastWoe()
    model_big.fit_matrix(big_rows, big_y, feature_names=["cat"])
    ci_big = model_big.predict_ci_matrix([["A"]], alpha=0.05)[0]
    width_big = ci_big[2] - ci_big[1]

    assert width_big < width_small
