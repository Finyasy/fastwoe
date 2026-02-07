# Assumptions And Limitations

This document captures model assumptions that must be validated before
production use in regulated workflows such as credit scoring.

## 1) Core Modeling Assumptions

- `predict_proba` and `predict_ci` rely on a Naive Bayes-style additive WOE view.
- The probabilistic interpretation assumes feature evidence is not strongly
  dependent.
- Very granular categories with very low counts can inflate uncertainty.

## 2) Required Validation Gates

- Verify Bayes consistency on representative datasets:
  - `posterior_odds = prior_odds * factor`
  - `logit(posterior) = logit(prior) + sum(WOE_i)`
- Verify probability reconstruction parity:
  - `predict_proba ~= sigmoid(logit(prior) + sum(WOE_i))`
- Verify WOE directionality:
  - Positive WOE increases event odds.
  - Negative WOE decreases event odds.
- Verify multiclass probability contract:
  - each `predict_proba_matrix_multiclass` row sums to `1` within tolerance.
  - class-specific APIs (`predict_proba_matrix_class`, `predict_ci_matrix_class`)
    remain consistent with full multiclass outputs.
- Verify sparse-bin stability:
  - `predict_ci` bounds stay finite and valid.
  - IV uncertainty outputs (`iv_se`, `iv_ci_lower`, `iv_ci_upper`) stay stable.
- Verify monotonic compliance for credit-scoring constraints:
  - increasing/decreasing directions hold after binning and in downstream scores.

## 3) FAISS Operational Notes

- FAISS is optional and should not block core `kmeans`/`tree` paths.
- Environments with FAISS/NumPy ABI mismatch can raise import errors.
- Release validation must include:
  - FAISS optional-path smoke tests where FAISS is available.
  - non-FAISS fallback behavior validation (`kmeans` and `tree` workflows).
- Rust-core FAISS integration remains deferred until benchmark thresholds are met.

## 4) Practical Guidance

- Use release builds for operational benchmarks:
  `python -m maturin develop --release --manifest-path crates/fastwoe-py/Cargo.toml`
- FastWoe now surfaces assumption-risk warnings on probability/CI APIs after fit
  when strong feature dependence or ultra-sparse categories are detected.
- You can inspect diagnostics programmatically:
  `model.get_assumption_diagnostics()`
- Warnings are only emitted when training data has at least
  `min_rows_for_assumption_warnings` rows (default `50`) to avoid noisy small-sample alerts.
- You can disable runtime warnings when needed:
  `FastWoe(warn_on_assumption_risk=False)`
- Gate releases with parity + invariants + stability tests:
  `PYTHONPATH=python pytest -q tests/test_phase0_parity_contract.py tests/test_invariants.py`
- Validate final performance and calibration on your real credit-scoring data
  before promoting to production.
