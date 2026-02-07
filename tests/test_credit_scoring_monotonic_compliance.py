from __future__ import annotations

from collections import defaultdict
from random import Random

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe
WoePreprocessor = fastwoe_mod.WoePreprocessor


def _event_rates_by_bin(labels: list[str], target: list[int]) -> list[float]:
    by_bin: dict[int, list[int]] = defaultdict(list)
    for label, y in zip(labels, target):
        bin_idx = int(label.split("_")[1])
        by_bin[bin_idx].append(y)
    return [sum(by_bin[idx]) / len(by_bin[idx]) for idx in sorted(by_bin)]


def test_credit_usage_monotonic_constraint_is_enforced_and_predictive() -> None:
    usage = [
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        70.0,
        80.0,
        85.0,
        88.0,
        89.0,
        90.0,
        91.0,
        92.0,
        93.0,
        94.0,
        95.0,
        96.0,
        97.0,
        98.0,
        99.0,
    ]
    target = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    rows = [[value] for value in usage]

    pre = WoePreprocessor(n_bins=5, binning_method="quantile")
    transformed = pre.fit_transform(
        rows,
        numerical_features=[0],
        target=target,
        monotonic_constraints="increasing",
    )

    train_labels = [str(row[0]) for row in transformed]
    rates = _event_rates_by_bin(train_labels, target)
    assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    model = FastWoe()
    model.fit_matrix(transformed, target, feature_names=["credit_usage"])

    grid_rows = [[30.0], [50.0], [70.0], [89.0], [92.0], [96.0], [99.0]]
    grid_bins = pre.transform(grid_rows)
    grid_probs = model.predict_proba_matrix(grid_bins)

    assert all(grid_probs[i] <= grid_probs[i + 1] for i in range(len(grid_probs) - 1))


def test_credit_usage_monotonic_fit_is_reproducible_under_row_ordering() -> None:
    usage = [float(v) for v in range(20, 100)]
    target = [0 if v < 90 else 1 for v in range(20, 100)]
    rows = [[value] for value in usage]

    pre_a = WoePreprocessor(n_bins=6, binning_method="quantile")
    pre_a.fit(
        rows,
        numerical_features=[0],
        target=target,
        monotonic_constraints="increasing",
    )

    pairs = list(zip(rows, target))
    Random(42).shuffle(pairs)
    shuffled_rows = [row for row, _ in pairs]
    shuffled_target = [y for _, y in pairs]

    pre_b = WoePreprocessor(n_bins=6, binning_method="quantile")
    pre_b.fit(
        shuffled_rows,
        numerical_features=[0],
        target=shuffled_target,
        monotonic_constraints="increasing",
    )

    probe = [[25.0], [45.0], [65.0], [85.0], [95.0], [99.0]]
    assert pre_a.transform(probe) == pre_b.transform(probe)
    assert pre_a.get_reduction_summary() == pre_b.get_reduction_summary()
