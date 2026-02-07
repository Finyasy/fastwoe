from __future__ import annotations

import math

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def test_sparse_bins_produce_finite_probability_and_ci_outputs() -> None:
    rows = [["common"]] * 200 + [["rare_a"], ["rare_b"], ["rare_c"], ["rare_d"]]
    target = [0] * 150 + [1] * 50 + [1, 0, 1, 0]

    model = FastWoe()
    model.fit_matrix(rows, target, feature_names=["segment"])

    ci_rows = model.predict_ci_matrix(
        [["common"], ["rare_a"], ["rare_b"], ["__unknown__"]],
        alpha=0.05,
    )

    for prediction, lower, upper in ci_rows:
        assert math.isfinite(prediction)
        assert math.isfinite(lower)
        assert math.isfinite(upper)
        assert 0.0 <= lower <= prediction <= upper <= 1.0

    width_common = ci_rows[0][2] - ci_rows[0][1]
    width_rare = ci_rows[1][2] - ci_rows[1][1]
    width_unknown = ci_rows[3][2] - ci_rows[3][1]

    assert width_rare > width_common
    assert width_unknown >= width_common


def test_sparse_bins_keep_iv_uncertainty_outputs_stable() -> None:
    rows = [["common"]] * 300 + [["rare_event"], ["rare_non_event"]]
    target = [0] * 220 + [1] * 80 + [1, 0]

    model = FastWoe()
    model.fit_matrix(rows, target, feature_names=["segment"])

    iv_rows = model.get_iv_analysis(alpha=0.05)
    assert len(iv_rows) == 1

    iv = iv_rows[0]
    assert math.isfinite(iv.iv)
    assert math.isfinite(iv.iv_se)
    assert math.isfinite(iv.iv_ci_lower)
    assert math.isfinite(iv.iv_ci_upper)
    assert iv.iv_se >= 0.0
    assert 0.0 <= iv.iv_ci_lower <= iv.iv <= iv.iv_ci_upper
