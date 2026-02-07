# Scripts

## simulation_benchmark.py

Runs a **simulation test** to see if the library is fast at scale and prints recommendations.

**Prereqs:** Install the project so the Rust extension is available:

```bash
# From repo root
maturin develop --release
# or
pip install -e .
```

**Run:**

```bash
python scripts/simulation_benchmark.py
```

Measures throughput (rows/s) for:

- Binary: `fit_matrix`, `transform_matrix`, `predict_proba_matrix`, `predict_ci_matrix` at 1k–250k rows
- Multiclass: fit and predict at 1k–50k rows
- Pandas input overhead (DataFrame → internal list conversion)

Then prints **recommendations** (data shape, batching, release builds, etc.).

## Rust benchmarks (Criterion)

From repo root:

```bash
cargo bench --package fastwoe-core
```

Requires Rust toolchain (`rustup`). Benchmarks run in the Rust layer only (no Python/PyO3 overhead).

## repro_ci_local.sh

Runs the CI-critical local flow without pip index fetches when a prepared conda env already
has required packages installed.

Usage:

```bash
bash scripts/repro_ci_local.sh fastwoe-faiss
```

It performs:
- release wheel build/install
- parity + preprocessor + invariants tests
- end-to-end latency threshold checks (`kmeans`, `tree`)
- end-to-end memory threshold checks (`kmeans`, `tree`)
