from __future__ import annotations

import math
import random
from collections import defaultdict

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe
WoePreprocessor = fastwoe_mod.WoePreprocessor


def _make_binary_dataset(seed: int, n_rows: int = 120) -> tuple[list[list[str]], list[int]]:
    rng = random.Random(seed)
    cats0 = ["A", "B", "C", "D", "E"]
    cats1 = ["x", "y", "z"]
    rows = [[rng.choice(cats0), rng.choice(cats1)] for _ in range(n_rows)]

    targets: list[int] = []
    for c0, c1 in rows:
        event = (c0 in {"A", "B"} and c1 == "x") or (c0 == "E" and c1 in {"x", "y"})
        if rng.random() < 0.1:
            event = not event
        targets.append(1 if event else 0)

    # Keep dataset valid for binary fitting.
    if not any(targets):
        targets[0] = 1
    if all(targets):
        targets[0] = 0
    return rows, targets


def _make_multiclass_dataset(seed: int, n_rows: int = 150) -> tuple[list[list[str]], list[str]]:
    rng = random.Random(seed)
    classes = ["c0", "c1", "c2"]
    cats0 = ["A", "B", "C", "D", "E"]
    cats1 = ["x", "y", "z"]
    rows = [[rng.choice(cats0), rng.choice(cats1)] for _ in range(n_rows)]

    labels: list[str] = []
    for c0, c1 in rows:
        if c0 in {"A", "B"}:
            label = "c0"
        elif c1 == "x":
            label = "c1"
        else:
            label = "c2"
        if rng.random() < 0.1:
            label = rng.choice(classes)
        labels.append(label)

    # Ensure all classes appear.
    labels[0] = "c0"
    labels[1] = "c1"
    labels[2] = "c2"
    return rows, labels


def _assert_probability_bounds(value: float, eps: float = 1e-12) -> None:
    assert math.isfinite(value)
    assert -eps <= value <= 1.0 + eps


def _assert_ci_bounds(pred: float, lo: float, hi: float, eps: float = 1e-12) -> None:
    _assert_probability_bounds(pred, eps=eps)
    _assert_probability_bounds(lo, eps=eps)
    _assert_probability_bounds(hi, eps=eps)
    assert lo - eps <= pred <= hi + eps
    assert lo <= hi + eps


def test_binary_probability_and_ci_invariants_randomized() -> None:
    feature_names = ["f0", "f1"]
    for seed in range(5):
        rows, targets = _make_binary_dataset(seed)
        predict_rows = rows + [["__unknown__", "__unknown__"]]

        model = FastWoe()
        model.fit_matrix(rows, targets, feature_names=feature_names)
        probs = model.predict_proba_matrix(predict_rows)
        ci_rows = model.predict_ci_matrix(predict_rows, alpha=0.05)

        assert len(probs) == len(predict_rows)
        assert len(ci_rows) == len(predict_rows)
        for p in probs:
            _assert_probability_bounds(p)
        for pred, lo, hi in ci_rows:
            _assert_ci_bounds(pred, lo, hi)

        # Determinism: same fit data should produce same predictions.
        model_2 = FastWoe()
        model_2.fit_matrix(rows, targets, feature_names=feature_names)
        probs_2 = model_2.predict_proba_matrix(predict_rows)
        assert probs_2 == pytest.approx(probs, abs=1e-12, rel=1e-12)


def test_multiclass_probability_and_ci_invariants_randomized() -> None:
    feature_names = ["f0", "f1"]
    for seed in range(5):
        rows, labels = _make_multiclass_dataset(seed)
        predict_rows = rows + [["__unknown__", "__unknown__"]]

        model = FastWoe()
        model.fit_matrix_multiclass(rows, labels, feature_names=feature_names)
        class_labels = model.get_class_labels()
        probs = model.predict_proba_matrix_multiclass(predict_rows)
        ci = model.predict_ci_matrix_multiclass(predict_rows, alpha=0.05)

        assert len(probs) == len(predict_rows)
        assert len(ci) == len(predict_rows)
        assert len(class_labels) >= 2

        for row_probs in probs:
            assert len(row_probs) == len(class_labels)
            for p in row_probs:
                _assert_probability_bounds(p)
            assert math.isclose(sum(row_probs), 1.0, abs_tol=1e-9, rel_tol=1e-9)

        for row_ci in ci:
            assert len(row_ci) == len(class_labels)
            for pred, lo, hi in row_ci:
                _assert_ci_bounds(pred, lo, hi)

        # Class-specific helper must match corresponding multiclass column.
        for class_idx, class_label in enumerate(class_labels):
            class_probs = model.predict_proba_matrix_class(predict_rows, class_label)
            expected_probs = [row[class_idx] for row in probs]
            assert class_probs == pytest.approx(expected_probs, abs=1e-12, rel=1e-12)


def test_preprocessor_monotonic_invariant_under_row_order_randomized() -> None:
    base_rows = [[float(i)] for i in range(1, 81)]
    base_target = [0 if i <= 40 else 1 for i in range(1, 81)]

    for seed in range(5):
        rng = random.Random(seed)
        pairs = list(zip(base_rows, base_target))
        rng.shuffle(pairs)
        rows = [r for r, _ in pairs]
        target = [y for _, y in pairs]

        pre = WoePreprocessor(n_bins=8, binning_method="kmeans")
        transformed = pre.fit_transform(
            rows,
            numerical_features=[0],
            target=target,
            monotonic_constraints="increasing",
        )

        by_bin: dict[int, list[int]] = defaultdict(list)
        for row, y in zip(transformed, target):
            idx = int(str(row[0]).split("_")[1])
            by_bin[idx].append(y)

        rates = [sum(by_bin[i]) / len(by_bin[i]) for i in sorted(by_bin)]
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
