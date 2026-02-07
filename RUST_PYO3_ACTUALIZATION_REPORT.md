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
- Confirm required API parity level (strict vs pragmatic).
- Select MVP cut for first release (`binary + basic stats` or `binary + multiclass`).
- Approve benchmark datasets and tolerance thresholds.
- Begin Phase 0 parity spec document and fixture extraction.

## 12) Implementation Progress Update
- Parity mode and MVP scope were formalized in `docs/parity/PHASE0_PARITY_SPEC.md`.
- Benchmark dataset/threshold baseline is documented in `docs/performance/BENCHMARK_DATASETS_AND_THRESHOLDS.md`.
- Phase 4 advanced-feature progress includes supervised `tree` numerical binning in `WoePreprocessor` for binary targets.
- Baseline monotonic-constraint workflow was added in `WoePreprocessor` for numerical binning.
- Regression coverage was added for out-of-range numeric bin assignment.

## 13) Developer setup (maturin + PyO3 extension)

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
- **Rust not found:** Install the Rust toolchain (`rustup`) and ensure `cargo` is on your PATH. The project uses `rust-toolchain.toml` (stable).
- **Parity/stat tests fail:** Ensure you ran `maturin develop --release` (release build). Debug builds are slower and are not the validated configuration.
