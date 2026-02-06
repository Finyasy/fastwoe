# Benchmark Datasets and Thresholds

This document formalizes the benchmark approval step from
`RUST_PYO3_ACTUALIZATION_REPORT.md`.

## 1) Datasets
Use deterministic synthetic workloads:

- Binary matrix:
- row counts: `1_000`, `10_000`, `50_000`, `100_000`, `250_000`
- feature counts: `2`, `5`, `10`, `20`, `50`
- category cardinality per column: `15` or `20`

- Multiclass matrix:
- row counts: `1_000`, `10_000`, `50_000`
- feature counts: `5`
- class count: `5`

## 2) Bench Commands
- Core simulation suite:
  `cargo bench -p fastwoe-core --bench woe_simulation`
- Targeted transform benchmark:
  `cargo bench -p fastwoe-core --bench woe_simulation -- binary_transform/transform_matrix/10000 --sample-size 10`

## 3) Baseline Thresholds (Initial)
Initial acceptance thresholds on Apple Silicon (local reference):

- `binary_transform/transform_matrix/10000`: `>= 17.0M elems/s`
- `binary_transform/transform_matrix/100000`: `>= 16.0M elems/s`

These are starting baselines for regression detection, not hard cross-machine SLAs.

## 4) Regression Policy
- Any benchmark regression over `10%` against last accepted baseline requires review.
- If a slower result is accepted, update this document with rationale and new baseline.
