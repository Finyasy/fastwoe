# FastWoe 0.1.2 Release Notes

Release date: February 7, 2026

## New Features

- Added assumption-risk diagnostics in `FastWoe` for:
  - strong feature dependence (Cramer's V based screening)
  - ultra-sparse categorical regimes (singleton/rare/unique-ratio checks)
- Added runtime warning surfacing on probability/CI APIs when risk is detected:
  - `predict_proba*`
  - `predict_ci*`
- Added `FastWoe.get_assumption_diagnostics()` for programmatic inspection.
- Added explicit warning controls:
  - `warn_on_assumption_risk`
  - threshold knobs for dependence/sparsity detection
  - `min_rows_for_assumption_warnings` (default `50`)
- Added deterministic credit-scoring Phase 0 parity fixture coverage.
- Expanded validation test suite for:
  - Bayes/WOE/logit consistency
  - monotonic credit-scoring compliance
  - sparse-bin inference stability
  - multiclass probability/CI contract behavior

## Breaking Changes

- No hard API breaks in existing fit/transform/predict methods.
- New warnings may appear during probability/CI calls on risk-prone datasets.
  This does not change outputs but may require warning filtering in strict
  pipelines.

## Migration Guidance

- If you have strict warning policies, either:
  - configure warning filters for `UserWarning` messages from FastWoe, or
  - disable this behavior with `FastWoe(warn_on_assumption_risk=False)`.
- For monitoring/observability, persist and inspect:
  `model.get_assumption_diagnostics()`.
- Keep using release builds for production:
  `python -m maturin develop --release --manifest-path crates/fastwoe-py/Cargo.toml`.

## Known Limitations

- FAISS remains optional and is not yet promoted to Rust-core implementation.
- Tree binning remains binary-target focused.
- CI/normal-approximation inference still requires domain validation on real
  credit-scoring data before production rollout.

## Validation Summary For This Release

- Quality gates:
  - `cargo fmt --all --check`
  - `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  - `cargo test --workspace --all-features`
  - `PYTHONPATH=python pytest -q tests/test_phase0_parity_contract.py`
- Local CI-equivalent flow:
  - `bash scripts/repro_ci_local.sh fastwoe-faiss`
- Benchmark and threshold gates:
  - criterion benchmark threshold checks
  - latency + memory threshold gates (`kmeans`, `tree`)
  - FAISS soft regression gates (`latency`, `memory`) in `fastwoe-faiss` env
