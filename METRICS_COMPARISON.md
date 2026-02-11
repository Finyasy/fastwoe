# METRICS_COMPARISON

_Generated: 2026-02-11 18:48:11 UTC_

## Scope
- Compare `pip install fastwoe` vs `pip install fastwoe-rs` on large simulated categorical datasets.
- Use only shared API surface across both libraries: `fit`, `transform`, `predict_proba` (1D categorical).

## Isolation (No Project Pollution)
- Benchmarks were executed in isolated virtualenvs under `/tmp/fastwoe_pkg_compare`.
- No project dependencies, lockfiles, or source organization were changed for this comparison.
- Project workspace only received this report file.

## Environment
- Python: `3.9.6`
- Platform: `macOS-26.2-arm64-arm-64bit`
- Package installs:
  - `fastwoe==0.1.6`
  - `fastwoe-rs==0.1.10`

## Benchmark Configuration
- Dataset sizes (rows): `[100000, 300000, 700000]`
- Cardinality: `500` categories
- Warmup: `1` run
- Measured repeats: `3` runs (median reported)
- Synthetic data: random categorical strings `c{0..N}` with binary targets.

## Raw Results

| Library | Rows | Fit (s) | Transform (s) | Predict (s) | Total (s) | Throughput (rows/s) |
|---|---:|---:|---:|---:|---:|---:|
| fastwoe | 100,000 | 0.069352 | 0.015308 | 0.018541 | 0.102797 | 972787 |
| fastwoe | 300,000 | 0.150499 | 0.043557 | 0.054897 | 0.248953 | 1205045 |
| fastwoe | 700,000 | 0.312807 | 0.101621 | 0.128201 | 0.543144 | 1288792 |
| fastwoe-rs | 100,000 | 0.052712 | 0.013588 | 0.013559 | 0.079883 | 1251826 |
| fastwoe-rs | 300,000 | 0.166977 | 0.040988 | 0.041363 | 0.248697 | 1206285 |
| fastwoe-rs | 700,000 | 0.412300 | 0.096100 | 0.095437 | 0.606193 | 1154748 |

## Head-to-Head

| Rows | Faster Library | Total Time Delta | Throughput Delta |
|---:|---|---:|---:|
| 100,000 | fastwoe-rs | 22.29% | +28.68% (`rs` vs `fastwoe`) |
| 300,000 | fastwoe-rs | 0.10% | +0.10% (`rs` vs `fastwoe`) |
| 700,000 | fastwoe | 10.40% | -10.40% (`rs` vs `fastwoe`) |

## Aggregate Summary (This Scenario)
- Weighted throughput across all tested sizes:
  - `fastwoe`: **1,229,194 rows/s**
  - `fastwoe-rs`: **1,176,756 rows/s**
- Relative: `fastwoe-rs` vs `fastwoe` = **-4.27%** throughput.

## Interpretation
- On this 1D shared API benchmark, performance is mixed by size:
  - `fastwoe-rs` leads at 100k rows and is nearly tied at 300k.
  - `fastwoe` leads at 700k rows in this run.
- Net effect for these three sizes is a small aggregate advantage for `fastwoe` in this specific scenario.
- This does **not** benchmark multi-feature/multiclass APIs because `fastwoe` (0.1.6) does not expose matrix methods.
- `fastwoe-rs` provides broader API coverage (matrix + multiclass + assumption diagnostics), so choose based on both performance and feature needs.

## Repro Command
```bash
python3 -m venv /tmp/fastwoe_pkg_compare/.venv_fastwoe
python3 -m venv /tmp/fastwoe_pkg_compare/.venv_fastwoe_rs
/tmp/fastwoe_pkg_compare/.venv_fastwoe/bin/python -m pip install numpy pandas fastwoe
/tmp/fastwoe_pkg_compare/.venv_fastwoe_rs/bin/python -m pip install numpy pandas fastwoe-rs
/tmp/fastwoe_pkg_compare/.venv_fastwoe/bin/python /tmp/fastwoe_pkg_compare/bench_compare_common_1d.py --label fastwoe --sizes 100000 300000 700000 --n-categories 500 --warmup 1 --repeats 3 --output /tmp/fastwoe_pkg_compare/result_fastwoe.json
/tmp/fastwoe_pkg_compare/.venv_fastwoe_rs/bin/python /tmp/fastwoe_pkg_compare/bench_compare_common_1d.py --label fastwoe-rs --sizes 100000 300000 700000 --n-categories 500 --warmup 1 --repeats 3 --output /tmp/fastwoe_pkg_compare/result_fastwoe_rs.json
```

## Extended Parameter Benchmarks (New)

_Added: 2026-02-11 22:03:20 UTC_

These additional benchmarks vary parameters beyond the baseline size-only run to evaluate whether the optimization remains effective under different data regimes.

Parameters varied:
- Categorical cardinality (`n_categories`: 50, 500, 5000)
- Class balance (`pos_rate`: 0.10, 0.30, 0.50)
- Unknown category share in inference (`unknown_ratio`: 0.30)
- Larger row volume (`rows`: 700,000)

| Scenario | Rows | Categories | Pos Rate | Unknown Ratio | fastwoe Total (s) | fastwoe-rs Total (s) | Winner | rs Throughput Delta |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `low_card_balanced` | 200,000 | 50 | 0.50 | 0.00 | 0.143544 | 0.139179 | fastwoe-rs | +3.14% |
| `high_card_balanced` | 200,000 | 5,000 | 0.50 | 0.00 | 0.429835 | 0.143537 | fastwoe-rs | +199.46% |
| `mid_card_imbalanced` | 200,000 | 500 | 0.10 | 0.00 | 0.173241 | 0.134917 | fastwoe-rs | +28.41% |
| `mid_card_unknown_30pct` | 200,000 | 500 | 0.30 | 0.30 | 0.174917 | 0.137462 | fastwoe-rs | +27.25% |
| `larger_rows_mid_card` | 700,000 | 500 | 0.30 | 0.00 | 0.707110 | 0.688778 | fastwoe-rs | +2.66% |

### Extended Grid Summary
- Scenario wins: `fastwoe-rs` **5 / 5**, `fastwoe` **0 / 5**.
- Weighted throughput across this grid:
  - `fastwoe`: **921,010 rows/s**
  - `fastwoe-rs` (optimized local wheel): **1,205,911 rows/s**
- Relative (`rs` vs `fastwoe`): **+30.93%** throughput on this parameter grid.

### Observations
- The optimization remains robust across cardinality and imbalance settings.
- Biggest gain appears at high-cardinality fit (`n_categories=5000`), where rs fit time is much lower.
- Unknown-category inference path (`unknown_ratio=0.30`) still shows rs ahead.
- For low-cardinality data, totals are close; rs gains come mainly from faster transform/predict while fit can be comparable or slightly slower.

### Repro (Extended Grid)
```bash
/tmp/fastwoe_pkg_compare/.venv_fastwoe/bin/python /tmp/fastwoe_pkg_compare/bench_scenario_grid.py --label fastwoe --warmup 1 --repeats 3 --seed 42 --output /tmp/fastwoe_pkg_compare/result_fastwoe_grid.json
/tmp/fastwoe_pkg_compare/.venv_fastwoe_rs/bin/python /tmp/fastwoe_pkg_compare/bench_scenario_grid.py --label fastwoe-rs-local-wheel --warmup 1 --repeats 3 --seed 42 --output /tmp/fastwoe_pkg_compare/result_fastwoe_rs_grid.json
```
