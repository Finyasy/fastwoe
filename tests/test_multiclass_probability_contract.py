from __future__ import annotations

import math

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def test_multiclass_probability_and_ci_class_contract() -> None:
    rows = [
        ["A", "x"],
        ["A", "y"],
        ["B", "x"],
        ["B", "z"],
        ["C", "y"],
        ["C", "z"],
        ["D", "x"],
        ["D", "y"],
        ["E", "z"],
        ["E", "x"],
        ["F", "y"],
        ["F", "z"],
    ]
    labels = ["c0", "c0", "c1", "c1", "c2", "c2", "c0", "c1", "c2", "c0", "c1", "c2"]
    predict_rows = rows + [["__unknown__", "__unknown__"]]

    model = FastWoe()
    model.fit_matrix_multiclass(rows, labels, feature_names=["f0", "f1"])

    class_labels = model.get_class_labels()
    probs = model.predict_proba_matrix_multiclass(predict_rows)
    ci_all = model.predict_ci_matrix_multiclass(predict_rows, alpha=0.05)

    assert len(class_labels) == 3
    assert len(probs) == len(predict_rows)
    assert len(ci_all) == len(predict_rows)

    for row_probs in probs:
        assert math.isclose(sum(row_probs), 1.0, abs_tol=1e-9, rel_tol=1e-9)

    for class_idx, class_label in enumerate(class_labels):
        class_probs = model.predict_proba_matrix_class(predict_rows, class_label)
        class_ci = model.predict_ci_matrix_class(predict_rows, class_label, alpha=0.05)

        expected_probs = [row[class_idx] for row in probs]
        expected_ci = [row[class_idx] for row in ci_all]

        assert class_probs == pytest.approx(expected_probs, abs=1e-12, rel=1e-12)
        assert class_ci == pytest.approx(expected_ci, abs=1e-12, rel=1e-12)

        for _probability, (prediction, lower, upper) in zip(class_probs, class_ci):
            assert 0.0 <= prediction <= 1.0
            assert 0.0 <= lower <= prediction <= upper <= 1.0
