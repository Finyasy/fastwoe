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

## 4) Versioning and Changelog
- Bump version in `pyproject.toml` and workspace package metadata if needed.
- Add release notes with:
- New features
- Breaking changes
- Migration guidance
- Known limitations

## 5) CI Validation
- Ensure `CI` workflow is green for push/PR.
- Trigger `Wheels` workflow and verify artifacts for Linux/macOS/Windows.
- Confirm scheduled/manual benchmark workflow runs and uploads artifacts.

## 6) Publish Readiness
- Tag release (`vX.Y.Z`) only when all gates pass.
- Publish wheels from CI artifacts or release pipeline.
