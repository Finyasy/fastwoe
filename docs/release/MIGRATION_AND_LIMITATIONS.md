# Migration And Known Limitations

This note documents migration guidance from legacy Python-only usage to the
Rust + PyO3 implementation, plus current limitations that are intentional.

## Migration Guidance

1. Install a wheel build that includes `fastwoe_rs` (or build locally with `maturin`).
2. Keep using the same top-level imports:
   - `from fastwoe import FastWoe, WoePreprocessor`
3. Prefer matrix APIs for multi-feature workloads:
   - `fit_matrix`, `transform_matrix`, `predict_proba_matrix`, `predict_ci_matrix`
4. For large inputs, pass NumPy arrays or pandas DataFrames directly to reduce
   Python-side conversion overhead.
5. Pin numerical-parity tests in downstream systems with tolerance-based checks
   (`abs_tol`/`rel_tol`) instead of exact bitwise equality.

## Compatibility Notes

- Main binary and multiclass APIs are preserved.
- Confidence interval APIs are available for binary and multiclass paths.
- `WoePreprocessor` supports list/NumPy/pandas inputs and returns matching
  container types where possible.
- Rust-backed preprocessing is enabled automatically when the extension is
  available.

## Known Limitations

- Tree-based binning currently supports binary targets only.
- FAISS binning remains optional and Python-path dependent (`faiss` must be
  installed).
- Confidence intervals are normal-approximation based and can widen in sparse
  bins; production thresholds should be validated on domain data.
- Very high-cardinality text features can still be memory-intensive before
  reduction; use feature selection and `top_p`/`max_categories` tuning.

## Operational Recommendations

- Build/install in release mode for production:
  `python -m maturin develop --release --manifest-path crates/fastwoe-py/Cargo.toml`
- Run the parity + invariant suite before promotion:
  `PYTHONPATH=python pytest -q tests/test_phase0_parity_contract.py tests/test_invariants.py`
- Track benchmark smoke throughput in CI and gate merges when thresholds regress.
