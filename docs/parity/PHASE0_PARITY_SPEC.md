# Phase 0 Parity Specification

This document operationalizes the immediate next actions from
`RUST_PYO3_ACTUALIZATION_REPORT.md`:

- Confirm API parity level.
- Select MVP release cut.
- Approve benchmark datasets and thresholds.
- Start fixture extraction and parity checks.

## 1) Parity Mode
Chosen parity mode: `pragmatic`.

Rules:
- Strict parity for API method names, required arguments, return shapes, and semantic behavior.
- Numeric parity uses tolerance-based validation for floating outputs.
- Deterministic ordering is required for feature names and mapping tables.

## 2) MVP Release Cut
Chosen MVP cut: `binary + multiclass + confidence intervals` (already implemented).

Out of MVP (remaining):
- Production FAISS packaging/CI support across platforms (optional milestone).
- Optional FAISS numerical binning remains Python-backed; Rust-core now covers
  quantile/uniform/kmeans/tree numerical binning and monotonic constraints.

## 3) Numeric Tolerances
Default tolerances for parity tests:

- Probabilities: `abs_tol=1e-9`, `rel_tol=1e-9`
- WOE transforms: `abs_tol=1e-9`, `rel_tol=1e-9`
- Confidence intervals: `abs_tol=1e-8`, `rel_tol=1e-8`

If future backend changes require relaxed tolerance, update this file and fixture version.

## 4) API Contract Scope (Phase 0)
In-scope methods for parity fixtures:

- Binary:
- `fit`, `transform`, `fit_transform`, `predict_proba`, `predict_ci`
- `fit_matrix`, `transform_matrix`, `fit_transform_matrix`, `predict_proba_matrix`, `predict_ci_matrix`
- `get_feature_names`, `get_feature_mapping`

- Multiclass:
- `fit_matrix_multiclass`, `predict_proba_matrix_multiclass`
- `predict_proba_matrix_class`
- `predict_ci_matrix_multiclass`, `predict_ci_matrix_class`
- `transform_matrix_multiclass`, `get_feature_names_multiclass`
- `get_class_labels`, `get_feature_mapping_multiclass`

- Preprocessor:
- `fit`, `transform`, `fit_transform`, `get_reduction_summary`
- Numeric binning scenarios for `quantile`, `kmeans`, `tree`
- Monotonic numerical constraints (`increasing`)

## 5) Acceptance Checklist
- Fixture generation script produces deterministic fixture JSON.
- Parity contract tests pass against current Rust+PyO3 package.
- Bench dataset/threshold document exists and is reviewed.
- CI can run parity tests as part of Python test stage.
