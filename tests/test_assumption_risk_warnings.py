from __future__ import annotations

import warnings

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def test_assumption_warning_for_strong_feature_dependence() -> None:
    rows = (
        [["A", "A"]] * 50
        + [["B", "B"]] * 50
    )
    target = ([0] * 35) + ([1] * 15) + ([0] * 15) + ([1] * 35)

    model = FastWoe(dependence_warning_threshold=0.7)
    model.fit_matrix(rows, target, feature_names=["f0", "f1"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = model.predict_proba_matrix(rows[:5])

    messages = [str(item.message) for item in caught]
    assert any("strong feature dependence" in message for message in messages)
    assert any("docs/validation/ASSUMPTIONS_AND_LIMITATIONS.md" in message for message in messages)

    diagnostics = model.get_assumption_diagnostics()
    assert diagnostics["at_risk"] is True
    assert diagnostics["dependence"]["at_risk"] is True
    assert diagnostics["dependence"]["max_cramers_v"] >= 0.7
    assert diagnostics["dependence"]["worst_pair"] == ["f0", "f1"]


def test_assumption_warning_for_ultra_sparse_categories() -> None:
    rows = [[f"id_{idx}", "stable"] for idx in range(80)] + [["shared", "stable"]] * 20
    target = ([0, 1] * 40) + ([0] * 10) + ([1] * 10)

    model = FastWoe(
        sparse_singleton_fraction_threshold=0.05,
        sparse_rare_fraction_threshold=0.1,
        sparse_unique_ratio_threshold=0.3,
    )
    model.fit_matrix(rows, target, feature_names=["merchant_id", "bucket"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = model.predict_ci_matrix(rows[:5], alpha=0.05)

    messages = [str(item.message) for item in caught]
    assert any("ultra-sparse categories" in message for message in messages)

    diagnostics = model.get_assumption_diagnostics()
    assert diagnostics["sparsity"]["at_risk"] is True
    assert diagnostics["sparsity"]["risky_features"][0]["feature"] == "merchant_id"


def test_assumption_warning_emitted_only_once_per_fit() -> None:
    rows = [[f"id_{idx}", "stable"] for idx in range(50)] + [["shared", "stable"]] * 50
    target = ([0, 1] * 25) + ([0] * 25) + ([1] * 25)

    model = FastWoe(
        sparse_singleton_fraction_threshold=0.05,
        sparse_rare_fraction_threshold=0.1,
        sparse_unique_ratio_threshold=0.3,
    )
    model.fit_matrix(rows, target, feature_names=["merchant_id", "bucket"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = model.predict_proba_matrix(rows[:8])
        _ = model.predict_ci_matrix(rows[:8], alpha=0.05)
        _ = model.predict_proba_matrix(rows[:8])

    messages = [str(item.message) for item in caught]
    assumption_messages = [message for message in messages if "assumption-risk warning" in message]
    assert len(assumption_messages) == 1


def test_assumption_warning_can_be_disabled() -> None:
    rows = [[f"id_{idx}", "stable"] for idx in range(30)] + [["shared", "stable"]] * 30
    target = ([0, 1] * 15) + ([0] * 15) + ([1] * 15)

    model = FastWoe(warn_on_assumption_risk=False)
    model.fit_matrix(rows, target, feature_names=["merchant_id", "bucket"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = model.predict_proba_matrix(rows[:5])

    messages = [str(item.message) for item in caught]
    assert not any("assumption-risk warning" in message for message in messages)
