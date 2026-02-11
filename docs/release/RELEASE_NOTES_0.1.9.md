# FastWoe 0.1.9 Release Notes

Release date: February 11, 2026

## New Features

- Added readable Python string representations for PyO3 row objects:
  - `WoeRow`
  - `IvRow`
  - `ReductionSummaryRow`
- Updated IV row display text to use generic confidence-interval labeling
  (`CI=[lower, upper]`) instead of a hardcoded `95%CI` label.
- Added regression tests for row-object representations to keep notebook output
  stable.

## Quality And Safety Improvements

- Added strict binary-target validation in Python wrappers to prevent silent
  coercion of non-binary values.
- Added duplicate `feature_names` validation in both Python and Rust paths.
- Added pytest path pinning to prefer local source-tree imports during
  repository test runs.

## Breaking Changes

- No API surface break in fit/transform/predict methods.
- Input validation is stricter for:
  - non-binary targets
  - duplicate `feature_names`
  Workloads relying on previous implicit coercion now fail fast with explicit
  validation errors.

## Migration Guidance

- Ensure binary labels are in `{0, 1}` (or equivalent string/bool forms).
- Ensure `feature_names` are unique for matrix APIs.
- If validating local changes before publishing, run tests from repository root
  and verify the active environment imports the intended package build.

## Known Limitations

- FAISS remains optional and may fall back to `kmeans` when unavailable or ABI
  incompatible.
- Tree binning remains binary-target focused.
- Confidence intervals and IV uncertainty are approximation-based and should be
  validated on real production datasets.

## Validation Summary For This Release

- `pytest -q`
- `cargo test --all-features`
- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- wheel build and metadata checks (`maturin build`, `twine check`)
