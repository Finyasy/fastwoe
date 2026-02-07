from __future__ import annotations

from collections import defaultdict

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
WoePreprocessor = fastwoe_mod.WoePreprocessor


def test_preprocessor_reduces_high_cardinality_and_groups_rare_values() -> None:
    rows = [["A"], ["A"], ["A"], ["B"], ["B"], ["C"], ["D"], ["E"], [None]]
    pre = WoePreprocessor(top_p=0.9, min_count=2, other_token="__other__")
    out = pre.fit_transform(rows)
    values = [r[0] for r in out]

    # Keep frequent categories and canonical missing category; group sparse tail.
    assert values.count("A") == 3
    assert values.count("B") == 2
    assert "__missing__" in values
    assert "__other__" in values


def test_preprocessor_feature_selection_by_name_on_dataframe() -> None:
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "cat": ["A", "A", "B", "C", "D"],
            "untouched": ["x", "y", "x", "z", "x"],
        }
    )
    pre = WoePreprocessor(top_p=0.7, min_count=1)
    out = pre.fit_transform(df, cat_features=["cat"])

    assert isinstance(out, pd.DataFrame)
    assert out["untouched"].tolist() == df["untouched"].tolist()
    assert "__other__" in set(out["cat"].tolist())


def test_preprocessor_unknown_category_maps_to_other_on_transform() -> None:
    pre = WoePreprocessor(top_p=1.0, min_count=1)
    pre.fit([["A"], ["B"]])
    out = pre.transform([["A"], ["Z"]])
    assert out[0][0] == "A"
    assert out[1][0] == "__other__"


def test_preprocessor_summary_has_expected_columns() -> None:
    pre = WoePreprocessor(top_p=0.8, min_count=1)
    pre.fit([["A"], ["A"], ["B"], ["C"]])
    summary = pre.get_reduction_summary()
    assert len(summary) == 1
    row = summary[0]
    assert set(row.keys()) == {"feature", "original_unique", "reduced_unique", "coverage"}
    assert 0.0 <= row["coverage"] <= 1.0


def test_preprocessor_summary_dataframe_mode() -> None:
    pd = pytest.importorskip("pandas")
    pre = WoePreprocessor(top_p=0.8, min_count=1)
    pre.fit([["A"], ["A"], ["B"], ["C"]])
    summary_df = pre.get_reduction_summary(as_frame=True)
    assert isinstance(summary_df, pd.DataFrame)
    assert list(summary_df.columns) == [
        "feature",
        "original_unique",
        "reduced_unique",
        "coverage",
    ]


def test_preprocessor_raises_on_invalid_feature_selector() -> None:
    pre = WoePreprocessor()
    with pytest.raises(ValueError, match="Unknown feature name"):
        pre.fit([["A", "x"], ["B", "y"]], cat_features=["does_not_exist"])


def test_preprocessor_requires_fit_before_transform_or_summary() -> None:
    pre = WoePreprocessor()
    with pytest.raises(RuntimeError, match="not fitted"):
        pre.transform([["A"]])
    with pytest.raises(RuntimeError, match="not fitted"):
        pre.get_reduction_summary()


def test_preprocessor_numeric_quantile_binning() -> None:
    rows = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    pre = WoePreprocessor(n_bins=3, binning_method="quantile")
    out = pre.fit_transform(rows, numerical_features=[0])
    labels = {r[0] for r in out}
    assert labels.issubset({"bin_0", "bin_1", "bin_2", "__missing__"})
    assert len(labels) >= 2


def test_preprocessor_numeric_uniform_binning_with_missing() -> None:
    rows = [[0.0], [1.0], [2.0], [None], [3.0]]
    pre = WoePreprocessor(n_bins=2, binning_method="uniform")
    out = pre.fit_transform(rows, numerical_features=[0])
    labels = [r[0] for r in out]
    assert "__missing__" in labels
    assert any(v.startswith("bin_") for v in labels if v != "__missing__")


def test_preprocessor_numeric_kmeans_binning() -> None:
    rows = [[0.0], [0.2], [0.3], [10.0], [10.1], [10.3], [20.0], [20.1]]
    pre = WoePreprocessor(n_bins=3, binning_method="kmeans")
    out = pre.fit_transform(rows, numerical_features=[0])
    labels = {r[0] for r in out}
    assert labels.issubset({"bin_0", "bin_1", "bin_2", "__missing__"})
    assert len(labels) >= 2


def test_preprocessor_numeric_and_categorical_integration() -> None:
    rows = [
        [1000.0, "A"],
        [1100.0, "A"],
        [1200.0, "B"],
        [1300.0, "C"],
        [1400.0, "D"],
    ]
    pre = WoePreprocessor(top_p=0.7, min_count=1, n_bins=2, binning_method="quantile")
    out = pre.fit_transform(rows, numerical_features=[0], cat_features=[1])

    # Numeric feature gets binned.
    assert all(str(r[0]).startswith("bin_") for r in out)
    # Categorical feature reduction still applies.
    assert "__other__" in {r[1] for r in out}


def test_preprocessor_rejects_invalid_binning_method() -> None:
    with pytest.raises(ValueError, match="binning_method"):
        WoePreprocessor(binning_method="does_not_exist")


def test_preprocessor_tree_binning_requires_target() -> None:
    rows = [[1.0], [2.0], [3.0]]
    pre = WoePreprocessor(n_bins=2, binning_method="tree")
    with pytest.raises(ValueError, match="target is required"):
        pre.fit(rows, numerical_features=[0])


def test_preprocessor_tree_binning_rejects_non_binary_target() -> None:
    rows = [[1.0], [2.0], [3.0], [4.0]]
    pre = WoePreprocessor(n_bins=2, binning_method="tree")
    with pytest.raises(ValueError, match="binary"):
        pre.fit(rows, numerical_features=[0], target=[0, 2, 1, 1])


def test_preprocessor_numeric_tree_binning_uses_target_signal() -> None:
    rows = [[1.0], [2.0], [3.0], [100.0], [110.0], [120.0]]
    target = [0, 0, 0, 1, 1, 1]
    pre = WoePreprocessor(n_bins=2, binning_method="tree")
    out = pre.fit_transform(rows, numerical_features=[0], target=target)

    first_label = out[0][0]
    second_label = out[-1][0]
    assert all(r[0] == first_label for r in out[:3])
    assert all(r[0] == second_label for r in out[3:])
    assert first_label != second_label


def test_preprocessor_numeric_binning_clamps_out_of_range_values() -> None:
    rows = [[0.0], [1.0], [2.0], [3.0]]
    pre = WoePreprocessor(n_bins=2, binning_method="uniform")
    pre.fit(rows, numerical_features=[0])

    out = pre.transform([[-10.0], [100.0]])
    assert out[0][0] == "bin_0"
    assert out[1][0] == "bin_1"


def test_preprocessor_monotonic_constraints_require_numerical_features() -> None:
    rows = [["A"], ["B"], ["C"]]
    pre = WoePreprocessor()
    with pytest.raises(ValueError, match="numerical_features"):
        pre.fit(rows, monotonic_constraints="increasing")


def test_preprocessor_monotonic_constraints_require_target() -> None:
    rows = [[1.0], [2.0], [3.0], [4.0]]
    pre = WoePreprocessor(n_bins=4, binning_method="quantile")
    with pytest.raises(ValueError, match="target is required"):
        pre.fit(rows, numerical_features=[0], monotonic_constraints="increasing")


def test_preprocessor_monotonic_constraints_reject_non_numeric_feature() -> None:
    rows = [[1.0, "A"], [2.0, "B"], [3.0, "C"], [4.0, "D"]]
    target = [0, 1, 0, 1]
    pre = WoePreprocessor(n_bins=2, binning_method="quantile")
    with pytest.raises(ValueError, match="numerical features"):
        pre.fit(
            rows,
            numerical_features=[0],
            target=target,
            monotonic_constraints={1: "increasing"},
        )


def test_preprocessor_monotonic_constraints_enforce_increasing_event_rate() -> None:
    rows = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    target = [0, 0, 1, 1, 0, 0, 1, 1]  # rates by 2-sample bins: [0, 1, 0, 1]
    pre = WoePreprocessor(n_bins=4, binning_method="quantile")
    transformed = pre.fit_transform(
        rows,
        numerical_features=[0],
        target=target,
        monotonic_constraints="increasing",
    )

    by_bin: dict[int, list[int]] = defaultdict(list)
    for row, y in zip(transformed, target):
        bin_idx = int(str(row[0]).split("_")[1])
        by_bin[bin_idx].append(y)

    rates = [sum(by_bin[i]) / len(by_bin[i]) for i in sorted(by_bin)]
    assert len(rates) >= 1
    assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


def test_preprocessor_monotonic_constraints_enforce_decreasing_event_rate() -> None:
    rows = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    target = [1, 1, 0, 0, 1, 1, 0, 0]  # rates by 2-sample bins: [1, 0, 1, 0]
    pre = WoePreprocessor(n_bins=4, binning_method="quantile")
    transformed = pre.fit_transform(
        rows,
        numerical_features=[0],
        target=target,
        monotonic_constraints="decreasing",
    )

    by_bin: dict[int, list[int]] = defaultdict(list)
    for row, y in zip(transformed, target):
        bin_idx = int(str(row[0]).split("_")[1])
        by_bin[bin_idx].append(y)

    rates = [sum(by_bin[i]) / len(by_bin[i]) for i in sorted(by_bin)]
    assert len(rates) >= 1
    assert all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
