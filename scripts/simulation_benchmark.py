#!/usr/bin/env python3
"""
Simulation benchmark for fastwoe: measures fit/transform/predict throughput
at various scales and prints recommendations.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

# Prefer local package: python/ is the package dir for maturin
root = Path(__file__).resolve().parent.parent
python_dir = root / "python"
if python_dir.exists():
    sys.path.insert(0, str(python_dir))
try:
    from fastwoe import FastWoe
except ImportError as e:
    print("Import error:", e, file=sys.stderr)
    print("Install the package first: pip install -e .  or  maturin develop", file=sys.stderr)
    sys.exit(1)



def make_matrix(n_rows: int, n_cols: int, cardinality: int) -> list[list[str]]:
    return [
        [f"cat_{c}_{random.randint(0, cardinality - 1)}" for c in range(n_cols)]
        for _ in range(n_rows)
    ]


def make_targets(n_rows: int, event_rate: float = 0.3) -> list[int]:
    return [1 if random.random() < event_rate else 0 for _ in range(n_rows)]


def make_multiclass_labels(n_rows: int, n_classes: int) -> list[str]:
    return [f"class_{random.randint(0, n_classes - 1)}" for _ in range(n_rows)]


def time_it(f, *args, n_warmup: int = 2, n_run: int = 5, **kwargs):
    for _ in range(n_warmup):
        f(*args, **kwargs)
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        f(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return min(times), sum(times) / len(times)


def run_binary_benchmarks():
    print("=" * 60)
    print("BINARY WOE – Simulation benchmark (Rust-backed fastwoe)")
    print("=" * 60)

    sizes = [1_000, 10_000, 50_000, 100_000, 250_000]
    n_cols = 5
    cardinality = 20

    for n_rows in sizes:
        rows = make_matrix(n_rows, n_cols, cardinality)
        targets = make_targets(n_rows, 0.3)
        model = FastWoe(smoothing=0.5, default_woe=0.0)

        try:
            t_fit, _ = time_it(model.fit_matrix, rows, targets)
        except AttributeError as e:
            if "fit_matrix" in str(e):
                print(
                    "\n  Rust backend is missing fit_matrix (old wheel?). Rebuild: maturin develop --release",
                    file=sys.stderr,
                )
            raise
        t_transform, _ = time_it(model.transform_matrix, rows)
        t_proba, _ = time_it(model.predict_proba_matrix, rows)
        t_ci, _ = time_it(model.predict_ci_matrix, rows, 0.05)

        fit_throughput = n_rows / t_fit if t_fit > 0 else 0
        transform_throughput = n_rows / t_transform if t_transform > 0 else 0
        proba_throughput = n_rows / t_proba if t_proba > 0 else 0

        print(f"\n  n_rows={n_rows:,}  n_cols={n_cols}")
        print(f"    fit_matrix:        {t_fit*1000:8.2f} ms  ({fit_throughput:,.0f} rows/s)")
        print(f"    transform_matrix:  {t_transform*1000:8.2f} ms  ({transform_throughput:,.0f} rows/s)")
        print(f"    predict_proba:     {t_proba*1000:8.2f} ms  ({proba_throughput:,.0f} rows/s)")
        print(f"    predict_ci (α=0.05): {t_ci*1000:8.2f} ms")


def run_multiclass_benchmarks():
    print("\n" + "=" * 60)
    print("MULTICLASS WOE – Simulation benchmark")
    print("=" * 60)

    sizes = [1_000, 10_000, 50_000]
    n_cols = 5
    n_classes = 5
    cardinality = 20

    for n_rows in sizes:
        rows = make_matrix(n_rows, n_cols, cardinality)
        labels = make_multiclass_labels(n_rows, n_classes)
        model = FastWoe(smoothing=0.5, default_woe=0.0)

        t_fit, _ = time_it(model.fit_matrix_multiclass, rows, labels)
        t_proba, _ = time_it(model.predict_proba_matrix_multiclass, rows)

        fit_throughput = n_rows / t_fit if t_fit > 0 else 0
        proba_throughput = n_rows / t_proba if t_proba > 0 else 0

        print(f"\n  n_rows={n_rows:,}  n_cols={n_cols}  n_classes={n_classes}")
        print(f"    fit_matrix_multiclass:     {t_fit*1000:8.2f} ms  ({fit_throughput:,.0f} rows/s)")
        print(f"    predict_proba_matrix:      {t_proba*1000:8.2f} ms  ({proba_throughput:,.0f} rows/s)")


def run_pandas_overhead_check():
    print("\n" + "=" * 60)
    print("PANDAS INPUT – Overhead (DataFrame → list[list[str]])")
    print("=" * 60)

    try:
        import pandas as pd
    except ImportError:
        print("  pandas not installed, skipping.")
        return

    n_rows = 50_000
    n_cols = 5
    cardinality = 20
    rows_list = make_matrix(n_rows, n_cols, cardinality)
    df = pd.DataFrame(rows_list, columns=[f"f{i}" for i in range(n_cols)])
    targets = make_targets(n_rows, 0.3)

    # Time with list input
    model = FastWoe(0.5, 0.0)
    t_list, _ = time_it(model.fit_matrix, rows_list, targets)
    t_list_proba, _ = time_it(model.predict_proba_matrix, rows_list)

    # Time with DataFrame input (goes through _to_2d_str in model.py)
    model2 = FastWoe(0.5, 0.0)
    t_df, _ = time_it(model2.fit_matrix, df, targets)
    t_df_proba, _ = time_it(model2.predict_proba_matrix, df)

    print(f"\n  n_rows={n_rows:,}  n_cols={n_cols}")
    print(f"    fit_matrix:     list {t_list*1000:.2f} ms   vs   DataFrame {t_df*1000:.2f} ms   (overhead: {(t_df/t_list - 1)*100:.1f}%)")
    print(f"    predict_proba:  list {t_list_proba*1000:.2f} ms   vs   DataFrame {t_df_proba*1000:.2f} ms   (overhead: {(t_df_proba/t_list_proba - 1)*100:.1f}%)")


def print_recommendations():
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("""
1. Throughput
   - Rust core is built for speed; expect hundreds of thousands to low millions
     of rows/s for transform/predict on typical hardware. Fit is heavier (hash
     aggregation per column) but still scales well.

2. Data shape
   - More columns → more hash lookups per row. If you have many high-cardinality
     categoricals, consider batching or reducing cardinality (e.g. rare bin).

3. Python ↔ Rust boundary
   - Passing list[list[str]] from Python allocates and copies. For maximum
     speed, batch large tables and avoid repeated tiny calls. NumPy/pandas
     buffers (zero-copy where possible) would reduce overhead (see your
     roadmap).

4. Multiclass
   - Multiclass fits one binary WOE model per class; predict runs all models
     and softmax. Scale is roughly linear in number of classes.

5. Confidence intervals
   - predict_ci adds a small cost (normal_ppf + arithmetic per row). Usually
     negligible compared to transform + proba.

6. Release builds
   - Always benchmark with `maturin build --release` and install the
     resulting wheel; debug builds are much slower.
""")


def main():
    random.seed(42)
    run_binary_benchmarks()
    run_multiclass_benchmarks()
    run_pandas_overhead_check()
    print_recommendations()
    print("Done.\n")


if __name__ == "__main__":
    main()
