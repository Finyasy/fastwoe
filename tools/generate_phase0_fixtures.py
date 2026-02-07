"""Generate deterministic Phase 0 parity fixtures.

Run this after installing the local package:
  python tools/generate_phase0_fixtures.py
"""

import argparse
import json
from pathlib import Path

from fastwoe import FastWoe, WoePreprocessor

DEFAULT_OUTPUT = Path("tests/fixtures/parity/phase0_v1.json")


def build_fixture() -> dict:
    binary_rows = [
        ["A", "x"],
        ["A", "y"],
        ["B", "x"],
        ["C", "z"],
        ["B", "y"],
        ["C", "x"],
    ]
    binary_targets = [1, 0, 0, 1, 0, 1]
    feature_names = ["cat", "bucket"]

    binary_model = FastWoe()
    binary_model.fit_matrix(binary_rows, binary_targets, feature_names=feature_names)

    multiclass_rows = binary_rows
    multiclass_labels = ["c0", "c1", "c2", "c0", "c1", "c2"]
    multiclass_model = FastWoe()
    multiclass_model.fit_matrix_multiclass(
        multiclass_rows, multiclass_labels, feature_names=feature_names
    )

    preprocessor_cases = _build_preprocessor_cases()
    credit_scoring_fixture = _build_credit_scoring_fixture()

    return {
        "fixture_version": "phase0_v1",
        "binary": {
            "rows": binary_rows,
            "targets": binary_targets,
            "feature_names": feature_names,
            "feature_names_out": binary_model.get_feature_names(),
            "transform_matrix": binary_model.transform_matrix(binary_rows),
            "predict_proba_matrix": binary_model.predict_proba_matrix(binary_rows),
            "predict_ci_matrix_alpha_0_05": binary_model.predict_ci_matrix(binary_rows, alpha=0.05),
            "iv_analysis_alpha_0_05": _iv_rows_to_dict(binary_model.get_iv_analysis(alpha=0.05)),
        },
        "multiclass": {
            "rows": multiclass_rows,
            "labels": multiclass_labels,
            "feature_names": feature_names,
            "class_labels_out": multiclass_model.get_class_labels(),
            "feature_names_multiclass_out": multiclass_model.get_feature_names_multiclass(),
            "transform_matrix_multiclass": multiclass_model.transform_matrix_multiclass(
                multiclass_rows
            ),
            "predict_proba_matrix_multiclass": multiclass_model.predict_proba_matrix_multiclass(
                multiclass_rows
            ),
            "predict_ci_matrix_multiclass_alpha_0_05": (
                multiclass_model.predict_ci_matrix_multiclass(multiclass_rows, alpha=0.05)
            ),
            "predict_proba_matrix_class_c0": multiclass_model.predict_proba_matrix_class(
                multiclass_rows, "c0"
            ),
            "predict_ci_matrix_class_c0_alpha_0_05": multiclass_model.predict_ci_matrix_class(
                multiclass_rows,
                "c0",
                alpha=0.05,
            ),
            "iv_analysis_class_c0_alpha_0_05": _iv_rows_to_dict(
                multiclass_model.get_iv_analysis_multiclass("c0", alpha=0.05)
            ),
        },
        "preprocessor": preprocessor_cases,
        "credit_scoring": credit_scoring_fixture,
    }


def _build_preprocessor_cases() -> dict:
    quantile_rows = [
        [1000.0, "A"],
        [1100.0, "A"],
        [1200.0, "B"],
        [1300.0, "C"],
        [1400.0, None],
        [None, "D"],
        [1600.0, "E"],
    ]
    quantile_params = {
        "top_p": 0.7,
        "min_count": 1,
        "n_bins": 3,
        "binning_method": "quantile",
    }
    quantile_fit_kwargs = {"numerical_features": [0], "cat_features": [1]}
    quantile_pre = WoePreprocessor(**quantile_params)
    quantile_transformed = quantile_pre.fit_transform(quantile_rows, **quantile_fit_kwargs)
    quantile_summary = quantile_pre.get_reduction_summary()

    kmeans_rows = [[0.0], [0.2], [0.3], [10.0], [10.1], [10.3], [20.0], [20.1], [None]]
    kmeans_params = {"n_bins": 3, "binning_method": "kmeans"}
    kmeans_fit_kwargs = {"numerical_features": [0]}
    kmeans_pre = WoePreprocessor(**kmeans_params)
    kmeans_transformed = kmeans_pre.fit_transform(kmeans_rows, **kmeans_fit_kwargs)
    kmeans_summary = kmeans_pre.get_reduction_summary()

    tree_rows = [[1.0], [2.0], [3.0], [100.0], [110.0], [120.0]]
    tree_target = [0, 0, 0, 1, 1, 1]
    tree_params = {"n_bins": 2, "binning_method": "tree"}
    tree_fit_kwargs = {"numerical_features": [0], "target": tree_target}
    tree_pre = WoePreprocessor(**tree_params)
    tree_transformed = tree_pre.fit_transform(tree_rows, **tree_fit_kwargs)
    tree_summary = tree_pre.get_reduction_summary()

    monotonic_rows = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    monotonic_target = [0, 0, 1, 1, 0, 0, 1, 1]
    monotonic_params = {"n_bins": 4, "binning_method": "quantile"}
    monotonic_fit_kwargs = {
        "numerical_features": [0],
        "target": monotonic_target,
        "monotonic_constraints": "increasing",
    }
    monotonic_pre = WoePreprocessor(**monotonic_params)
    monotonic_transformed = monotonic_pre.fit_transform(
        monotonic_rows, **monotonic_fit_kwargs
    )
    monotonic_summary = monotonic_pre.get_reduction_summary()

    return {
        "quantile_mixed": {
            "rows": quantile_rows,
            "params": quantile_params,
            "fit_kwargs": quantile_fit_kwargs,
            "transformed": quantile_transformed,
            "summary": quantile_summary,
        },
        "kmeans_numeric": {
            "rows": kmeans_rows,
            "params": kmeans_params,
            "fit_kwargs": kmeans_fit_kwargs,
            "transformed": kmeans_transformed,
            "summary": kmeans_summary,
        },
        "tree_numeric": {
            "rows": tree_rows,
            "params": tree_params,
            "fit_kwargs": tree_fit_kwargs,
            "transformed": tree_transformed,
            "summary": tree_summary,
        },
        "monotonic_increasing": {
            "rows": monotonic_rows,
            "params": monotonic_params,
            "fit_kwargs": monotonic_fit_kwargs,
            "transformed": monotonic_transformed,
            "summary": monotonic_summary,
        },
    }


def _build_credit_scoring_fixture() -> dict:
    # Deterministic synthetic counts that mirror the worked credit-usage example:
    # >90% usage has a much higher default likelihood than <=90%.
    usage_rows = (
        [["<=90"]] * 81
        + [["<=90"]] * 2
        + [[">90"]] * 9
        + [[">90"]] * 8
    )
    usage_targets = ([0] * 81) + ([1] * 2) + ([0] * 9) + ([1] * 8)
    feature_names = ["credit_usage_group"]

    model = FastWoe()
    model.fit_matrix(usage_rows, usage_targets, feature_names=feature_names)

    # Raw (unsmoothed) worked-example values for roadmap and validation checks.
    prior_odds = (10 / 100) / (90 / 100)
    factor = (8 / 17) / (10 / 100)
    woe_log = _safe_log(factor)
    posterior_odds = prior_odds * factor
    posterior_prob = posterior_odds / (1.0 + posterior_odds)

    query_rows = [["<=90"], [">90"], ["__unknown__"]]

    return {
        "rows": usage_rows,
        "targets": usage_targets,
        "feature_names": feature_names,
        "query_rows": query_rows,
        "mapping": _mapping_rows_to_dict(model.get_feature_mapping(feature_names[0])),
        "query_transform_matrix": model.transform_matrix(query_rows),
        "query_predict_proba_matrix": model.predict_proba_matrix(query_rows),
        "query_predict_ci_matrix_alpha_0_05": model.predict_ci_matrix(query_rows, alpha=0.05),
        "worked_example_raw": {
            "prior_odds": prior_odds,
            "factor": factor,
            "woe_log": woe_log,
            "posterior_odds": posterior_odds,
            "posterior_prob": posterior_prob,
            "counts": {
                "total": 100,
                "non_default_total": 90,
                "default_total": 10,
                "gt_90_non_default": 9,
                "gt_90_default": 8,
                "lte_90_non_default": 81,
                "lte_90_default": 2,
            },
        },
    }


def _safe_log(value: float) -> float:
    if value <= 0.0:
        raise ValueError("log input must be positive.")
    import math

    return math.log(value)


def _iv_rows_to_dict(rows: list[object]) -> list[dict]:
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


def _mapping_rows_to_dict(rows: list[object]) -> list[dict]:
    return [
        {
            "category": row.category,
            "event_count": row.event_count,
            "non_event_count": row.non_event_count,
            "woe": row.woe,
            "woe_se": row.woe_se,
        }
        for row in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 0 parity fixture JSON.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for fixture JSON.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture()
    output_path.write_text(json.dumps(fixture, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote fixture: {output_path}")


if __name__ == "__main__":
    main()
