# FastWOE Rust + PyO3 Actualization Report

## 1) Objective
Build a Rust core library for FastWOE and expose it to Python via PyO3, preserving the current user-facing behavior while improving performance, reliability, and maintainability.

## 2) Synthesized Requirements From `woe.md`
The Rust implementation should preserve these functional pillars:

- Binary and multiclass WOE encoding (one-vs-rest for multiclass).
- Inference APIs: probability prediction and confidence intervals.
- Statistical outputs: WOE standard errors, IV standard errors, IV confidence intervals, feature-level summaries.
- Categorical preprocessing for high-cardinality features.
- Numerical binning modes:
- `kbins`-style binning.
- tree-based supervised binning.
- FAISS KMeans-style binning (initially optional/deferred if integration cost is high).
- Monotonic constraints support for numerical features.
- Robust handling of missing values, rare categories, and unknown categories.
- Python ergonomics aligned with current API expectations.

## 3) Product Scope and Delivery Strategy
Use staged delivery to reduce risk while preserving compatibility.

### Phase 0: Specification and Baseline
- Freeze Python API contract from current behavior.
- Define numeric tolerance targets between Python baseline and Rust outputs.
- Identify what must be strict parity vs acceptable approximation.

Exit criteria:
- Written parity spec and acceptance checklist.

### Phase 1: Rust Core (Binary WOE First)
- Implement Rust data model and core WOE computations for binary targets.
- Implement smoothing, event/non-event counts, WOE, IV, and standard errors.
- Implement missing/unknown category handling and deterministic ordering.

Exit criteria:
- Deterministic outputs with parity tests passing for binary workflows.

### Phase 2: PyO3 Bindings and Python API
- Expose Rust core through PyO3 classes and methods.
- Match Python-friendly input/output behavior (DataFrame/array-like support as feasible).
- Preserve method names and return schemas expected by users.

Exit criteria:
- End-to-end fit/transform/predict flows usable from Python with wheels installable locally.

### Phase 3: Multiclass + Statistical Inference
- Add one-vs-rest multiclass encoding and class-specific probability APIs.
- Add confidence interval and IV analysis endpoints in Rust and surface via PyO3.

Exit criteria:
- Multiclass and inference parity tests green against agreed tolerances.

### Phase 4: Advanced Features
- Add numerical binning variants (`kbins`, tree, optional FAISS-like path).
- Add monotonic constraints workflow.
- Optimize hot paths and memory layout.

Exit criteria:
- Performance and correctness targets met for advanced feature set.

### Phase 5: Packaging and Release
- Build manylinux/macOS/windows wheels via maturin CI.
- Publish pre-release and stable release with migration notes.

Exit criteria:
- Reproducible builds, test matrix green, release checklist complete.

## 4) Proposed Technical Architecture

### Rust Workspace Structure
- `fastwoe-core`: pure Rust algorithms and statistics (no Python dependency).
- `fastwoe-py`: PyO3 bindings crate that wraps `fastwoe-core`.
- Optional later crate(s) for external integrations (for example FAISS bridge if needed).

### Data and API Boundary
- Keep heavy computation in Rust.
- Keep Python object marshaling in thin binding layer.
- Minimize per-row Python callbacks; prefer vectorized batch transfer.

### Error Model
- Rust domain errors mapped into clear Python exceptions (`ValueError`, `RuntimeError`, etc.).
- Validation failures caught early in bindings before core compute.

## 5) API Parity Plan (Python-Facing)
Target parity with the currently documented behavior:

- `fit`, `transform`, `fit_transform`.
- `predict_proba`, class-specific probability helpers for multiclass.
- Confidence interval endpoints.
- Mapping/statistics endpoints (`get_mapping`, feature stats, IV analysis).
- Preprocessor behavior for high-cardinality category reduction.

Potential intentional differences to decide early:
- Strictness of input typing and coercion.
- Output container defaults (NumPy arrays vs pandas DataFrames).
- NaN/null canonical handling for speed and consistency.

## 6) Verification and Quality Gates

### Test Layers
- Unit tests in Rust core for formulas and edge cases.
- Property-based tests for invariants (monotonicity constraints, probability bounds, stability).
- Python integration tests validating API parity and error semantics.
- Regression fixtures from current implementation outputs.

### Acceptance Metrics
- Statistical parity within defined tolerances.
- Runtime speedup target on representative datasets (define benchmark suite).
- Memory usage targets and no major regressions.
- Cross-platform wheel installation and smoke tests.

## 7) Tooling and Build System
- Use `maturin` for wheel building and PyPI publishing path.
- CI matrix:
- Python versions aligned with project policy.
- OS: Linux/macOS/Windows.
- Rust stable toolchain (plus pinned minimum supported Rust version).
- Include benchmarks in CI (or scheduled workflow) for trend tracking.

## 8) Risks and Mitigations

- Risk: Hidden behavior differences from pandas/sklearn internals.
- Mitigation: Freeze behavioral fixtures and compare outputs systematically.

- Risk: Multiclass + CI/statistical methods create subtle numeric drift.
- Mitigation: Define tolerance envelopes and deterministic compute order.

- Risk: FAISS parity complexity for Python packaging.
- Mitigation: Treat FAISS path as optional milestone; release core binning first.

- Risk: API breakage for existing users.
- Mitigation: Compatibility layer, deprecation warnings, migration guide.

## 9) Recommended Delivery Timeline (High-Level)
- Week 1-2: parity spec, fixtures, Rust core scaffolding.
- Week 3-4: binary WOE core + tests.
- Week 5: PyO3 bindings + packaging skeleton.
- Week 6-7: multiclass + inference stats.
- Week 8: advanced binning/constraints prioritization.
- Week 9: hardening, benchmarks, RC release.

## 10) Definition of Done
Project is considered actualized when all conditions are met:

- Core APIs available in Python through PyO3.
- Binary and multiclass workflows validated.
- Statistical outputs (WOE/IV uncertainty metrics) exposed and tested.
- Packaging produces installable wheels across target platforms.
- Documentation includes migration and known limitations.

## 11) Immediate Next Actions
- [x] Keep extension-backed parity checks in CI for the Rust numeric preprocessor path.
- [x] Decide whether FAISS remains an optional Python path or gets a Rust-core implementation.
- [x] Expand preprocessing benchmarks (latency + throughput) for quantile/kmeans/tree binning.
- [x] Complete release hardening (wheel matrix, smoke tests, migration notes).

## 12) Implementation Progress Update
- Parity mode and MVP scope were formalized in `docs/parity/PHASE0_PARITY_SPEC.md`.
- Benchmark dataset/threshold baseline is documented in `docs/performance/BENCHMARK_DATASETS_AND_THRESHOLDS.md`.
- Phase 4 advanced-feature progress includes supervised `tree` numerical binning in `WoePreprocessor` for binary targets.
- KBins-style `kmeans` numerical binning was added to `WoePreprocessor`.
- Optional FAISS-backed numerical binning path was added to `WoePreprocessor` when `faiss` is installed.
- CI now includes a Linux FAISS optional-path validation job.
- High-cardinality categorical reduction logic now has a Rust-core (`PreprocessorCore`) path via PyO3.
- Numerical binning now has a Rust-core path (`NumericBinnerCore`) exposed via PyO3 (`RustNumericBinner`) for `quantile`, `uniform`, `kmeans`, and `tree`.
- The preprocessor Rust bridge now uses numeric-native marshaling (`Option[f64]`) for numeric features and selected-column marshaling for categorical reduction to reduce string-conversion overhead on NumPy/pandas inputs.
- Monotonic-constraint edge enforcement for numerical bins now runs in Rust core when the Rust backend is available.
- Python parity tests now include Rust-vs-Python checks for numeric preprocessor paths (quantile, kmeans, tree, monotonic).
- Phase 0 fixture parity coverage now includes deterministic preprocessor cases (quantile, kmeans, tree, monotonic) in `tests/fixtures/parity/phase0_v1.json`.
- Baseline monotonic-constraint workflow was added in `WoePreprocessor` for numerical binning.
- Regression coverage was added for out-of-range numeric bin assignment.
- Deterministic invariant/property-style tests were added for probability bounds, CI validity, and monotonic stability.
- Performance hardening now includes preprocessor benchmark groups (categorical and numeric) with CI smoke thresholds for numeric quantile transform throughput.
- CI now verifies Rust extension preprocessor backends are present and includes invariant tests in the quality gate suite.
- Wheels CI now performs install/import/fit smoke tests on Linux/macOS/Windows before uploading artifacts.
- Migration and known limitations documentation is now tracked in `docs/release/MIGRATION_AND_LIMITATIONS.md`.
- FAISS decision benchmarking harness was added at `tools/benchmark_faiss_decision.py`, with current baseline output captured in `docs/performance/FAISS_DECISION_BENCHMARK.md`.
- FAISS decision is currently to keep FAISS as an optional Python path; measured benchmarks did not justify Rust-core FAISS integration.
- Scheduled CI now runs a FAISS decision benchmark job with a soft regression gate (`tools/check_faiss_regression.py`) to fail only on major degradation.
- Rust numeric hot-path optimization improved preprocessor performance (`fit_kmeans`, `transform_quantile`) via reduced allocations and faster 1D k-means assignment.
- CI benchmark smoke now enforces thresholds for Rust preprocessor `kmeans`/`tree` and Python end-to-end latency thresholds for `kmeans`/`tree`.
- Local CI-equivalent reproduction is now scripted (`scripts/repro_ci_local.sh`) and validated without pip index fetches when using a prepared conda environment.
- Benchmark policy now includes end-to-end preprocessor memory deltas with CI threshold checks for `kmeans`/`tree`, plus scheduled `faiss` memory monitoring.
- Memory thresholds were tightened on February 7, 2026, and scheduled benchmarks now include a FAISS-vs-kmeans memory ratio soft gate (`tools/check_faiss_memory_regression.py`).
- Validation-roadmap test modules are now implemented for Bayes/WOE consistency, credit-scoring monotonic compliance, sparse-bin inference stability, and multiclass probability/CI contract checks.
- CI `quality-and-parity` now executes the new validation-roadmap test modules on every push and pull request.
- Phase 0 parity fixtures now include a deterministic credit-scoring worked example snapshot (counts, Bayes-factor arithmetic, and query-level Rust-model parity outputs).
- Validation assumption guardrails are now documented in `docs/validation/ASSUMPTIONS_AND_LIMITATIONS.md`.
- FastWoe now includes assumption-risk diagnostics for strong feature dependence and ultra-sparse categories, with runtime warnings surfaced on probability/CI APIs and opt-out support.
- Release checklist now includes explicit FAISS optional-path and non-FAISS fallback validation steps.

## 13) FAISS Decision Outcome (2026-02-07)
Decision source: `docs/performance/FAISS_DECISION_BENCHMARK.md`

Measured on representative synthetic numeric workloads (`10_000` and `100_000` rows, `4` numeric features, `8` bins):
- Preprocess best (`10_000`): `kmeans 32.126 ms` vs `faiss 47.869 ms`
- Preprocess best (`100_000`): `kmeans 453.994 ms` vs `faiss 493.762 ms`
- End-to-end best (`10_000`): `kmeans 49.710 ms` vs `faiss 58.275 ms`
- End-to-end best (`100_000`): `kmeans 616.789 ms` vs `faiss 650.255 ms`

Decision:
- Keep FAISS as an optional Python path.
- Do not implement Rust-core FAISS yet.
- Re-evaluate only when FAISS shows at least `20%` preprocess gain and `10%` end-to-end gain against `kmeans` across tested sizes.

## 14) Developer setup (maturin + PyO3 extension)

To build and test the Rust-backed Python extension locally:

### Install maturin

Use the active Python interpreter (recommended: do this inside your conda/venv):

```bash
python -m pip install --upgrade pip
python -m pip install maturin
```

Verify:

```bash
python -m maturin --version
```

With conda:

```bash
conda activate <your-env>
python -m pip install maturin
```

### Build and install the PyO3 extension

From the **repository root**:

```bash
python -m maturin develop --release --manifest-path crates/fastwoe-py/Cargo.toml
```

This compiles the Rust crate and installs the `fastwoe` package (with `fastwoe_rs` extension) into the current environment.

### Run parity and statistical tests

After the extension is installed:

```bash
PYTHONPATH=python pytest -q tests/test_phase0_parity_contract.py tests/test_statistical_accuracy.py
```

For the full test suite:

```bash
PYTHONPATH=python pytest -q tests/
```

### Troubleshooting

- **Missing maturin:** Install as above; ensure the same Python you use for `pytest` has maturin and the developed package.
- **`python -m maturin` reports `Unable to find maturin script`:** In mixed conda/venv setups, add `$CONDA_PREFIX/bin` to PATH and run `maturin` directly, or use `scripts/repro_ci_local.sh`.
- **Rust not found:** Install the Rust toolchain (`rustup`) and ensure `cargo` is on your PATH. The project uses `rust-toolchain.toml` (stable).
- **Parity/stat tests fail:** Ensure you ran `maturin develop --release` (release build). Debug builds are slower and are not the validated configuration.

## 15) Local CI-Equivalent Reproduction (Verified)

Verified on **February 7, 2026** with:
- release wheel build via maturin
- wheel install into a local venv
- `tests/test_phase0_parity_contract.py`, `tests/test_preprocessor.py`, `tests/test_invariants.py`
- end-to-end latency thresholds for `kmeans` and `tree`

Command:

```bash
bash scripts/repro_ci_local.sh fastwoe-faiss
```

## 16) Validation Signals Extracted From `images/` + `woe.md` (Roadmap Addendum)

The following source artifacts were reviewed for validation-critical requirements:
- `images/alan.png`
- `images/default.png`
- `images/theory.png`
- `images/fastwoe1.png`
- `images/woe1.png`
- `woe.md`

### Core Mathematical Validation Requirements
- Validate Bayes factor identity: `posterior_odds = prior_odds * factor`.
- Validate additive log-odds identity: `logit(posterior) = logit(prior) + sum(WOE_i)`.
- Validate probability reconstruction parity: `predict_proba` must match `sigmoid(logit(prior) + sum(WOE_i))` within tolerance.
- Validate WOE sign semantics: positive WOE increases event/default odds and negative WOE decreases event/default odds.
- Track optional explainability output: add deciban-style explanation support (`10 * log10(factor)`) as a roadmap item.

### Credit-Scoring Validation Requirements
- Add explicit credit-scoring fixtures mirroring the worked odds-factor-WOE examples from `images/default.png`.
- Treat monotonic constraints as compliance gates: increasing/decreasing constraints must hold after fit/transform and remain visible in summaries.
- Enforce small-sample robustness checks: confidence intervals and standard errors must remain valid for sparse bins and rare categories.
- Require numeric stability for IV uncertainty outputs: `iv_se`, `iv_ci_lower`, and `iv_ci_upper` must be non-contradictory.

### Inference and Documentation Guardrails
- Keep and strengthen the inference caveat from `woe.md`: `predict_proba` and `predict_ci` rely on naive-independence assumptions.
- Add explicit user guidance for correlated features and ultra-granular categories.
- Add a release validation checklist item confirming that assumptions are documented in API docs and release notes.

### Multiclass and API Contract Requirements
- Enforce one-vs-rest probability mass checks: each `predict_proba(X)` row must sum to `1` within tolerance.
- Enforce class API consistency: `predict_proba_class` and `predict_ci_class` must remain consistent with full multiclass outputs.
- Update Phase 3 acceptance criteria to include multiclass CI parity, not only multiclass probability parity.

### FAISS Operational Validation Requirements
- Keep FAISS optional (no Rust-core FAISS implementation yet), aligned with current benchmark decision.
- Add a packaging compatibility gate from `woe.md` troubleshooting guidance.
- Validate FAISS import behavior against NumPy compatibility constraints in a dedicated environment matrix.
- Preserve a fallback path to `kmeans`/`tree` when FAISS is unavailable or incompatible.

## 17) Roadmap Actualization Tasks Derived From This Addendum
- Add `tests/test_bayes_woe_consistency.py` for odds/logit/probability reconstruction parity.
- Add `tests/test_credit_scoring_monotonic_compliance.py` for monotonic direction and reproducibility.
- Add `tests/test_sparse_bin_inference_stability.py` for CI/SE behavior under rare-count regimes.
- Add `tests/test_multiclass_probability_contract.py` for simplex and class-specific API consistency.
- Add `docs/validation/ASSUMPTIONS_AND_LIMITATIONS.md` capturing independence assumptions and failure modes.
- Extend release checklist to include FAISS import-compat smoke checks and fallback-behavior validation.
