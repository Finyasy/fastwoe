# Release Checklist

This checklist implements Phase 5 (Packaging and Release) from
`RUST_PYO3_ACTUALIZATION_REPORT.md`.

## 1) Quality Gates
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace --all-features`
- `python -m pytest -q tests/test_phase0_parity_contract.py`

## 2) Build Artifacts
- Build local release wheel:
  `python -m maturin build --release --manifest-path crates/fastwoe-py/Cargo.toml`
- Confirm wheel exists under `target/wheels/`.

## 3) Benchmark Gate
- Run benchmark smoke:
  `cargo bench -p fastwoe-core --bench woe_simulation -- binary_transform/transform_matrix/10000 --sample-size 10`
- Compare result against `docs/performance/BENCHMARK_DATASETS_AND_THRESHOLDS.md`.
- Run end-to-end preprocessor latency + memory threshold checks:
  `python tools/check_preprocessor_latency_thresholds.py ...`
  `python tools/check_preprocessor_memory_thresholds.py ...`
- For FAISS benchmark environments, run memory ratio soft gate:
  `python tools/check_faiss_memory_regression.py ...`
- Validate FAISS optional path and fallback behavior:
  `PYTHONPATH=python pytest -q tests/test_preprocessor.py -k "faiss_binning_optional or faiss_binning_executes_when_faiss_available"`
  `PYTHONPATH=python pytest -q tests/test_preprocessor.py -k "numeric_kmeans_binning or numeric_tree_binning_uses_target_signal"`

## 4) Versioning and Changelog
- Bump version in `pyproject.toml` and workspace package metadata if needed.
- Add release notes with:
- New features
- Breaking changes
- Migration guidance
- Known limitations
- Update `docs/release/MIGRATION_AND_LIMITATIONS.md` for this release.

## 5) CI Validation
- Ensure `CI` workflow is green for push/PR.
- Ensure the optional `FAISS Optional Path` CI job is green (Linux).
- Trigger `Wheels` workflow and verify artifacts for Linux/macOS/Windows.
- Confirm wheel smoke tests pass on Linux/macOS/Windows jobs.
- Confirm `Source distribution (sdist)` job is green in `Wheels`.
- Confirm scheduled/manual benchmark workflow runs and uploads artifacts.
- Confirm new validation docs are reflected in release notes:
  `docs/validation/ASSUMPTIONS_AND_LIMITATIONS.md`
- Reproduce CI-critical local flow before release:
  `bash scripts/repro_ci_local.sh fastwoe-faiss`

## 6) Publish Setup (One-Time)
- Configure PyPI Trusted Publisher for this repository/workflow:
- Owner/repo: `Finyasy/fastwoe`
- Workflow: `.github/workflows/wheels.yml`
- Optional API-token fallback:
- GitHub secret `PYPI_API_TOKEN` (for PyPI)
- GitHub secret `TEST_PYPI_API_TOKEN` (for TestPyPI)
- Configure TestPyPI Trusted Publisher with the same repository/workflow when using OIDC.

## 7) Publish Readiness
- Tag release (`vX.Y.Z`) only when all gates pass.
- Push tag to publish to PyPI automatically:
  `git tag vX.Y.Z && git push origin vX.Y.Z`
- For dry-runs, run `Wheels` manually with `publish_to=none`.
- For TestPyPI publish, run `Wheels` manually with `publish_to=testpypi`.
- For manual PyPI publish (no new tag), run `Wheels` manually with `publish_to=pypi`.
- Verify published files on:
- PyPI: `https://pypi.org/project/fastwoe-rs/`
- TestPyPI: `https://test.pypi.org/project/fastwoe-rs/`
- Validate release performance on a real credit-scoring dataset:
  `python tools/benchmark_real_dataset.py --input-csv /path/to/credit.csv --target-col <target> --methods kmeans tree --threshold kmeans:<pre_ms>:<e2e_ms> --threshold tree:<pre_ms>:<e2e_ms>`
