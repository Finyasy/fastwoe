# Benchmark Datasets and Thresholds

This document formalizes the benchmark approval step from
`RUST_PYO3_ACTUALIZATION_REPORT.md`.

## 1) Datasets
Use deterministic synthetic workloads:

- Binary matrix:
- row counts: `1_000`, `10_000`, `50_000`, `100_000`, `250_000`
- feature counts: `2`, `5`, `10`, `20`, `50`
- category cardinality per column: `15` or `20`

- Preprocessor categorical:
- row counts: `10_000`, `100_000`
- feature counts: `4`
- category cardinality per column: `300`

- Preprocessor numeric:
- row counts: `10_000`, `100_000`
- feature counts: `3` (criterion smoke) and `4` (FAISS decision benchmark)
- methods: `quantile` transform smoke, `kmeans`/`tree` (scheduled benchmark), `faiss` (optional benchmark when installed)

- Multiclass matrix:
- row counts: `1_000`, `10_000`, `50_000`
- feature counts: `5`
- class count: `5`

## 2) Bench Commands
- Core simulation suite:
  `cargo bench -p fastwoe-core --bench woe_simulation`
- Targeted transform benchmark:
  `cargo bench -p fastwoe-core --bench woe_simulation -- binary_transform/transform_matrix/10000 --sample-size 10`
- Targeted preprocessor numeric benchmark:
  `cargo bench -p fastwoe-core --bench woe_simulation -- preprocessor_numeric/transform_quantile/10000 --sample-size 10`
- Targeted preprocessor `kmeans` benchmark:
  `cargo bench -p fastwoe-core --bench woe_simulation -- preprocessor_numeric/fit_kmeans/10000 --sample-size 10`
- Targeted preprocessor `tree` benchmark:
  `cargo bench -p fastwoe-core --bench woe_simulation -- preprocessor_numeric/fit_tree/10000 --sample-size 10`
- End-to-end preprocessor latency benchmark (`kmeans`/`tree`):
  `python tools/benchmark_faiss_decision.py --methods kmeans tree --sizes 10000 --warmup 1 --repeats 3 --output benchmark-artifacts/`
- End-to-end preprocessor memory benchmark (`kmeans`/`tree`):
  `python tools/benchmark_preprocessor_memory.py --methods kmeans tree --sizes 10000 --output benchmark-artifacts/`
- FAISS decision benchmark (Python end-to-end):
  `python tools/benchmark_faiss_decision.py --methods kmeans tree faiss --sizes 10000 100000 --output docs/performance/FAISS_DECISION_BENCHMARK.md`
- FAISS soft regression gate:
  `python tools/check_faiss_regression.py --report docs/performance/FAISS_DECISION_BENCHMARK.md --sizes 10000 100000 --max-preprocess-ratio 2.0 --max-e2e-ratio 1.6`
- End-to-end latency threshold gate:
  `python tools/check_preprocessor_latency_thresholds.py --report benchmark-artifacts/FAISS_DECISION_BENCHMARK.md --threshold kmeans:10000:120:180 --threshold tree:10000:120:160`
- End-to-end memory threshold gate:
  `python tools/check_preprocessor_memory_thresholds.py --report benchmark-artifacts/PREPROCESSOR_MEMORY_BENCHMARK.md --threshold kmeans:10000:150:190 --threshold kmeans:100000:180:220 --threshold tree:10000:150:190 --threshold tree:100000:180:220 --threshold faiss:10000:180:220 --threshold faiss:100000:200:240`
- FAISS memory soft regression gate:
  `python tools/check_faiss_memory_regression.py --report benchmark-artifacts/PREPROCESSOR_MEMORY_BENCHMARK.md --sizes 10000 100000 --max-pre-delta-ratio 1.5 --max-e2e-delta-ratio 1.5`

## 3) Baseline Thresholds (Initial)
Initial acceptance thresholds on Apple Silicon (local reference):

- `binary_transform/transform_matrix/10000`: `>= 17.0M elems/s`
- `binary_transform/transform_matrix/100000`: `>= 16.0M elems/s`
- `preprocessor_numeric/transform_quantile/10000`: `>= 8.0M elems/s`
- `preprocessor_numeric/fit_kmeans/10000`: `>= 12.0M elems/s`
- `preprocessor_numeric/fit_tree/10000`: `>= 15.0M elems/s`
- End-to-end latency (`kmeans`, 10k): preprocess `<= 120 ms`, e2e `<= 180 ms`
- End-to-end latency (`tree`, 10k): preprocess `<= 120 ms`, e2e `<= 160 ms`
- End-to-end memory delta (`kmeans`, 10k): preprocess `<= 150 MB`, e2e `<= 190 MB`
- End-to-end memory delta (`kmeans`, 100k): preprocess `<= 180 MB`, e2e `<= 220 MB`
- End-to-end memory delta (`tree`, 10k): preprocess `<= 150 MB`, e2e `<= 190 MB`
- End-to-end memory delta (`tree`, 100k): preprocess `<= 180 MB`, e2e `<= 220 MB`
- End-to-end memory delta (`faiss`, 10k, scheduled benchmark): preprocess `<= 180 MB`, e2e `<= 220 MB`
- End-to-end memory delta (`faiss`, 100k, scheduled benchmark): preprocess `<= 200 MB`, e2e `<= 240 MB`

These are starting baselines for regression detection, not hard cross-machine SLAs.
Thresholds were tightened on **February 7, 2026** after the first green CI-equivalent validation cycle.

### CI Push-Safe Thresholds
To reduce false negatives on shared GitHub runners, `CI / Benchmark Smoke` uses
more conservative push-time thresholds:
- `binary_transform/transform_matrix/10000`: `>= 12.0M elems/s`
- `preprocessor_numeric/transform_quantile/10000`: `>= 6.0M elems/s`
- `preprocessor_numeric/fit_kmeans/10000`: `>= 8.0M elems/s`
- `preprocessor_numeric/fit_tree/10000`: `>= 10.0M elems/s`
- End-to-end latency (`kmeans`, 10k): preprocess `<= 250 ms`, e2e `<= 350 ms`
- End-to-end latency (`tree`, 10k): preprocess `<= 250 ms`, e2e `<= 330 ms`

Stricter thresholds remain enforced in scheduled benchmark workflows.

## 4) FAISS Integration Decision Rule
- Keep FAISS optional unless benchmarks show clear gain on representative workloads.
- Promote FAISS to Rust-core implementation only when both are true versus `kmeans`:
- preprocessing latency gain is at least `20%` across tested sizes.
- end-to-end latency gain (preprocess + fit + predict) is at least `10%` across tested sizes.

## 5) Regression Policy
- Any benchmark regression over `10%` against last accepted baseline requires review.
- If a slower result is accepted, update this document with rationale and new baseline.
- FAISS benchmark uses a soft gate in scheduled CI: fail only on major degradation
  (faiss/kmeans preprocess ratio > `2.0` or end-to-end ratio > `1.6` for monitored sizes).
- FAISS memory benchmark uses a soft gate in scheduled CI: fail when
  faiss/kmeans memory-delta ratio exceeds `1.5` for preprocess or end-to-end at monitored sizes.
