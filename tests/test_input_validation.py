from __future__ import annotations

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def test_binary_targets_do_not_silently_coerce_non_binary_values() -> None:
    model = FastWoe()
    with pytest.raises(ValueError, match="target must be binary"):
        model.fit(["A", "B", "C"], [0.2, 1.0, 0.0])


def test_binary_targets_accept_string_binary_values() -> None:
    model = FastWoe()
    model.fit(["A", "B", "C"], ["1", "0", "1"])
    probs = model.predict_proba(["A", "B", "C"])
    assert len(probs) == 3


def test_fit_matrix_rejects_duplicate_feature_names() -> None:
    rows = [["A", "x"], ["B", "y"], ["A", "y"], ["B", "x"]]
    target = [1, 0, 1, 0]
    model = FastWoe()

    with pytest.raises(ValueError, match="feature_names must be unique"):
        model.fit_matrix(rows, target, feature_names=["dup", "dup"])


def test_fit_matrix_multiclass_rejects_duplicate_feature_names() -> None:
    rows = [["A", "x"], ["B", "y"], ["A", "y"], ["B", "x"]]
    labels = ["c0", "c1", "c0", "c1"]
    model = FastWoe()

    with pytest.raises(ValueError, match="feature_names must be unique"):
        model.fit_matrix_multiclass(rows, labels, feature_names=["dup", "dup"])
