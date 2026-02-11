from __future__ import annotations

import math
from collections import Counter

import pytest

fastwoe_mod = pytest.importorskip("fastwoe")
FastWoe = fastwoe_mod.FastWoe


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: float) -> float:
    p = min(max(probability, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))


def test_predict_proba_matches_prior_plus_row_woe_sum() -> None:
    rows = [
        ["online", ">90"],
        ["online", ">90"],
        ["online", "<=90"],
        ["branch", "<=90"],
        ["branch", "<=90"],
        ["branch", ">90"],
        ["mobile", "<=90"],
        ["mobile", "<=90"],
        ["mobile", ">90"],
        ["mobile", ">90"],
    ]
    target = [1, 1, 0, 0, 0, 1, 0, 0, 1, 0]

    model = FastWoe()
    model.fit_matrix(rows, target, feature_names=["channel", "credit_usage"])

    transformed = model.transform_matrix(rows)
    predicted = model.predict_proba_matrix(rows)

    base_log_odds = _logit(sum(target) / len(target))
    reconstructed = [_sigmoid(base_log_odds + sum(row_woe)) for row_woe in transformed]

    assert predicted == pytest.approx(reconstructed, abs=1e-12, rel=1e-12)


def test_mapping_respects_woe_formula_and_sign_semantics() -> None:
    categories = ["A", "A", "A", "B", "B", "C", "C", "C", "C", "C"]
    target = [1, 1, 0, 0, 0, 1, 0, 0, 0, 0]

    model = FastWoe()
    model.fit(categories, target)
    mapping = model.get_mapping()

    smoothing = 0.5
    total_events = sum(target)
    total_non_events = len(target) - total_events
    num_categories = len(mapping)
    event_denom = total_events + smoothing * num_categories
    non_event_denom = total_non_events + smoothing * num_categories
    overall_event_rate = total_events / len(target)

    by_category = Counter(categories)
    events_by_category = Counter(cat for cat, y in zip(categories, target) if y == 1)

    for row in mapping:
        expected_woe = math.log(
            ((row.event_count + smoothing) / event_denom)
            / ((row.non_event_count + smoothing) / non_event_denom)
        )
        assert row.woe == pytest.approx(expected_woe, abs=1e-12, rel=1e-12)

        category_event_rate = events_by_category[row.category] / by_category[row.category]
        if category_event_rate > overall_event_rate:
            assert row.woe > 0.0
        elif category_event_rate < overall_event_rate:
            assert row.woe < 0.0


def test_single_feature_api_matches_single_column_matrix_api() -> None:
    categories = ["A", "A", "B", "C", "A", "B", "D", "D", "C", "A"]
    target = [1, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    query = ["A", "B", "C", "D", "Z"]

    model_1d = FastWoe(smoothing=0.5, default_woe=0.0)
    model_1d.fit(categories, target)
    transform_1d = model_1d.transform(query)
    proba_1d = model_1d.predict_proba(query)
    mapping_1d = {row.category: row for row in model_1d.get_mapping()}

    rows = [[value] for value in categories]
    query_rows = [[value] for value in query]
    model_matrix = FastWoe(smoothing=0.5, default_woe=0.0)
    model_matrix.fit_matrix(rows, target, feature_names=["f0"])
    transform_matrix = [row[0] for row in model_matrix.transform_matrix(query_rows)]
    proba_matrix = model_matrix.predict_proba_matrix(query_rows)
    mapping_matrix = {row.category: row for row in model_matrix.get_feature_mapping("f0")}

    assert transform_1d == pytest.approx(transform_matrix, abs=1e-12, rel=1e-12)
    assert proba_1d == pytest.approx(proba_matrix, abs=1e-12, rel=1e-12)
    assert sorted(mapping_1d) == sorted(mapping_matrix)

    for category in sorted(mapping_1d):
        left = mapping_1d[category]
        right = mapping_matrix[category]
        assert left.event_count == right.event_count
        assert left.non_event_count == right.non_event_count
        assert left.woe == pytest.approx(right.woe, abs=1e-12, rel=1e-12)
        assert left.woe_se == pytest.approx(right.woe_se, abs=1e-12, rel=1e-12)
