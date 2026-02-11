from __future__ import annotations

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def test_woe_row_repr_is_readable() -> None:
    model = FastWoe()
    model.fit(["A", "A", "B", "C"], [1, 0, 0, 1])

    row = model.get_mapping()[0]
    text = repr(row)
    assert text.startswith("WoeRow(")
    assert "category=" in text
    assert "woe=" in text
    assert "woe_se=" in text


def test_iv_row_str_uses_generic_ci_label() -> None:
    rows = [["A", "x"], ["A", "y"], ["B", "x"], ["C", "z"], ["C", "y"]]
    targets = [1, 0, 0, 1, 1]
    model = FastWoe()
    model.fit_matrix(rows, targets, feature_names=["cat", "bucket"])

    iv_row = model.get_iv_analysis(alpha=0.10)[0]
    text = str(iv_row)
    assert "CI=[" in text
    assert "95%CI" not in text


def test_reduction_summary_row_repr_is_readable() -> None:
    rust_pre_cls = getattr(fastwoe_mod, "RustPreprocessor", None)
    if rust_pre_cls is None:
        pytest.skip("RustPreprocessor not available in current extension build.")

    pre = rust_pre_cls(
        max_categories=2,
        top_p=0.9,
        min_count=1,
        other_token="__other__",
        missing_token="__missing__",
    )
    rows = [["A"], ["B"], ["A"], ["C"]]
    pre.fit(rows, ["f0"], [0])
    row = pre.get_reduction_summary()[0]

    assert repr(row).startswith("ReductionSummaryRow(")
    text = str(row)
    assert "feature=" in text
    assert "coverage" in text
