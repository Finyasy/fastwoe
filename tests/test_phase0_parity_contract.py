from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe

FIXTURE_PATH = Path("tests/fixtures/parity/phase0_v1.json")


def _load_fixture() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture missing: {FIXTURE_PATH}. "
            "Generate with tools/generate_phase0_fixtures.py"
        )
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _assert_deep_close(actual: Any, expected: Any, abs_tol: float, rel_tol: float) -> None:
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for a, e in zip(actual, expected):
            _assert_deep_close(a, e, abs_tol=abs_tol, rel_tol=rel_tol)
        return

    if isinstance(expected, (int, float)):
        assert isinstance(actual, (int, float))
        assert math.isclose(float(actual), float(expected), abs_tol=abs_tol, rel_tol=rel_tol)
        return

    assert actual == expected


def test_binary_phase0_parity_contract() -> None:
    fixture = _load_fixture()["binary"]
    model = FastWoe()
    model.fit_matrix(fixture["rows"], fixture["targets"], feature_names=fixture["feature_names"])

    assert model.get_feature_names() == fixture["feature_names_out"]
    _assert_deep_close(
        model.transform_matrix(fixture["rows"]),
        fixture["transform_matrix"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )
    _assert_deep_close(
        model.predict_proba_matrix(fixture["rows"]),
        fixture["predict_proba_matrix"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )
    _assert_deep_close(
        model.predict_ci_matrix(fixture["rows"], alpha=0.05),
        fixture["predict_ci_matrix_alpha_0_05"],
        abs_tol=1e-8,
        rel_tol=1e-8,
    )
    _assert_deep_close(
        _iv_rows_to_dict(model.get_iv_analysis(alpha=0.05)),
        fixture["iv_analysis_alpha_0_05"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )


def test_multiclass_phase0_parity_contract() -> None:
    fixture = _load_fixture()["multiclass"]
    model = FastWoe()
    model.fit_matrix_multiclass(
        fixture["rows"], fixture["labels"], feature_names=fixture["feature_names"]
    )

    assert model.get_class_labels() == fixture["class_labels_out"]
    assert model.get_feature_names_multiclass() == fixture["feature_names_multiclass_out"]

    _assert_deep_close(
        model.transform_matrix_multiclass(fixture["rows"]),
        fixture["transform_matrix_multiclass"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )
    _assert_deep_close(
        model.predict_proba_matrix_multiclass(fixture["rows"]),
        fixture["predict_proba_matrix_multiclass"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )
    _assert_deep_close(
        model.predict_ci_matrix_multiclass(fixture["rows"], alpha=0.05),
        fixture["predict_ci_matrix_multiclass_alpha_0_05"],
        abs_tol=1e-8,
        rel_tol=1e-8,
    )
    _assert_deep_close(
        model.predict_proba_matrix_class(fixture["rows"], "c0"),
        fixture["predict_proba_matrix_class_c0"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )
    _assert_deep_close(
        model.predict_ci_matrix_class(fixture["rows"], "c0", alpha=0.05),
        fixture["predict_ci_matrix_class_c0_alpha_0_05"],
        abs_tol=1e-8,
        rel_tol=1e-8,
    )
    _assert_deep_close(
        _iv_rows_to_dict(model.get_iv_analysis_multiclass("c0", alpha=0.05)),
        fixture["iv_analysis_class_c0_alpha_0_05"],
        abs_tol=1e-9,
        rel_tol=1e-9,
    )


def _iv_rows_to_dict(rows: list[Any]) -> list[dict]:
    return [
        {
            "feature": row.feature,
            "iv": row.iv,
            "iv_se": row.iv_se,
            "iv_ci_lower": row.iv_ci_lower,
            "iv_ci_upper": row.iv_ci_upper,
            "iv_significance": row.iv_significance,
        }
        for row in rows
    ]
