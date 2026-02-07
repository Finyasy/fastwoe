#!/usr/bin/env python3
"""Benchmark preprocessor binning methods to guide FAISS integration decisions."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastwoe import FastWoe, WoePreprocessor


@dataclass
class BenchResult:
    method: str
    rows: int
    best_seconds: float
    median_seconds: float
    rows_per_second: float
    e2e_best_seconds: float


def _faiss_import_status() -> tuple[bool, str]:
    if importlib.util.find_spec("faiss") is None:
        return False, "faiss module not found"
    try:
        # Some broken faiss installs emit ABI errors to stderr during import.
        # Silence that noise and return a structured status instead.
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module("faiss")
    except Exception as exc:  # pragma: no cover - depends on environment packaging.
        return False, f"faiss import failed: {exc}"
    return True, "faiss import ok"


def _make_numeric_rows(rows: int, n_numeric_features: int) -> list[list[float]]:
    rng = random.Random(42)
    out: list[list[float]] = []
    for i in range(rows):
        row: list[float] = []
        for j in range(n_numeric_features):
            base = ((i * (j + 3)) % 4093) * (0.03 + j * 0.01)
            row.append(base + rng.random() * (1.0 + j))
        out.append(row)
    return out


def _make_binary_target(rows: int) -> list[int]:
    rng = random.Random(7)
    out: list[int] = []
    for i in range(rows):
        p = 0.2 + 0.6 * ((i % 12) / 12.0)
        out.append(1 if rng.random() < p else 0)
    if not any(out):
        out[0] = 1
    if all(out):
        out[0] = 0
    return out


def _time_many(fn: Any, *args: Any, warmup: int, repeats: int, **kwargs: Any) -> list[float]:
    for _ in range(warmup):
        fn(*args, **kwargs)
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        out.append(time.perf_counter() - t0)
    return out


def _benchmark_method(
    *,
    method: str,
    rows: list[list[float]],
    y: list[int],
    n_bins: int,
    warmup: int,
    repeats: int,
) -> BenchResult:
    kwargs = {"numerical_features": list(range(len(rows[0])))}
    if method == "tree":
        kwargs["target"] = y

    # Benchmark preprocessing fit_transform.
    pre = WoePreprocessor(n_bins=n_bins, binning_method=method)
    pre_times = _time_many(pre.fit_transform, rows, warmup=warmup, repeats=repeats, **kwargs)
    best = min(pre_times)
    median = statistics.median(pre_times)

    # Benchmark end-to-end path (preprocess + model fit + predict).
    feature_names = [f"f{i}" for i in range(len(rows[0]))]

    def run_e2e() -> None:
        pre_local = WoePreprocessor(n_bins=n_bins, binning_method=method)
        transformed = pre_local.fit_transform(rows, **kwargs)
        model = FastWoe()
        model.fit_matrix(transformed, y, feature_names=feature_names)
        _ = model.predict_proba_matrix(transformed)

    e2e_times = _time_many(run_e2e, warmup=warmup, repeats=repeats)
    e2e_best = min(e2e_times)

    return BenchResult(
        method=method,
        rows=len(rows),
        best_seconds=best,
        median_seconds=median,
        rows_per_second=(len(rows) / best) if best > 0 else 0.0,
        e2e_best_seconds=e2e_best,
    )


def _format_seconds(seconds: float) -> str:
    return f"{seconds * 1000.0:.3f} ms"


def _decision_text(results: dict[tuple[str, int], BenchResult], faiss_available: bool) -> str:
    if not faiss_available:
        return (
            "FAISS not available in this environment, so FAISS-vs-kmeans speedup could not "
            "be measured. Keep FAISS optional until the benchmark is run where faiss is installed."
        )

    comparisons: list[tuple[int, float, float]] = []
    for (method, rows), result in results.items():
        if method != "kmeans":
            continue
        faiss_key = ("faiss", rows)
        if faiss_key not in results:
            continue
        faiss_result = results[faiss_key]
        preprocess_gain = (result.best_seconds - faiss_result.best_seconds) / result.best_seconds
        e2e_gain = (
            result.e2e_best_seconds - faiss_result.e2e_best_seconds
        ) / result.e2e_best_seconds
        comparisons.append((rows, preprocess_gain, e2e_gain))

    if not comparisons:
        return "No comparable kmeans/faiss results available."

    min_pre_gain = min(c[1] for c in comparisons)
    min_e2e_gain = min(c[2] for c in comparisons)
    if min_pre_gain >= 0.20 and min_e2e_gain >= 0.10:
        return (
            "Recommend implementing Rust-core FAISS integration: minimum observed gain is "
            f"{min_pre_gain * 100:.1f}% (preprocess) and {min_e2e_gain * 100:.1f}% (end-to-end)."
        )
    return (
        "Do not implement Rust-core FAISS yet: gains do not clear the threshold "
        "(>=20% preprocess and >=10% end-to-end)."
    )


def _render_markdown(
    *,
    results: dict[tuple[str, int], BenchResult],
    methods: list[str],
    sizes: list[int],
    n_bins: int,
    n_features: int,
    warmup: int,
    repeats: int,
    faiss_available: bool,
    faiss_status: str,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# FAISS Decision Benchmark",
        "",
        f"- Generated (UTC): `{now}`",
        f"- Methods requested: `{methods}`",
        f"- Dataset sizes: `{sizes}`",
        f"- Numeric features: `{n_features}`",
        f"- n_bins: `{n_bins}`",
        f"- Warmup: `{warmup}`",
        f"- Repeats: `{repeats}`",
        f"- faiss available: `{faiss_available}`",
        f"- faiss status: `{faiss_status}`",
        "",
        "## Results",
        "",
        "| method | rows | preprocess best | preprocess median | preprocess rows/s | e2e best |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for method in methods:
        for size in sizes:
            key = (method, size)
            if key not in results:
                lines.append(f"| {method} | {size} | skipped | skipped | skipped | skipped |")
                continue
            res = results[key]
            lines.append(
                "| "
                f"{res.method} | "
                f"{res.rows} | "
                f"{_format_seconds(res.best_seconds)} | "
                f"{_format_seconds(res.median_seconds)} | "
                f"{res.rows_per_second:,.0f} | "
                f"{_format_seconds(res.e2e_best_seconds)} |"
            )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            _decision_text(results, faiss_available),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark kmeans/tree/faiss preprocessing to guide FAISS integration."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["kmeans", "tree", "faiss"],
        help="Methods to benchmark.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10_000, 100_000],
        help="Row counts to benchmark.",
    )
    parser.add_argument("--n-bins", type=int, default=8, help="Number of numeric bins.")
    parser.add_argument(
        "--n-features",
        type=int,
        default=4,
        help="Number of numeric features in synthetic data.",
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs per benchmark.")
    parser.add_argument("--repeats", type=int, default=5, help="Measured runs per benchmark.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/performance/FAISS_DECISION_BENCHMARK.md"),
        help="Path to write markdown report.",
    )
    args = parser.parse_args()

    available_methods = set(args.methods)
    faiss_available, faiss_status = _faiss_import_status()
    if "faiss" in available_methods and not faiss_available:
        print(f"faiss unavailable; faiss benchmarks will be skipped ({faiss_status}).")

    results: dict[tuple[str, int], BenchResult] = {}
    for size in args.sizes:
        rows = _make_numeric_rows(size, args.n_features)
        y = _make_binary_target(size)
        for method in args.methods:
            if method == "faiss" and not faiss_available:
                continue
            print(f"benchmarking method={method} rows={size} ...")
            result = _benchmark_method(
                method=method,
                rows=rows,
                y=y,
                n_bins=args.n_bins,
                warmup=args.warmup,
                repeats=args.repeats,
            )
            results[(method, size)] = result

    report = _render_markdown(
        results=results,
        methods=args.methods,
        sizes=args.sizes,
        n_bins=args.n_bins,
        n_features=args.n_features,
        warmup=args.warmup,
        repeats=args.repeats,
        faiss_available=faiss_available,
        faiss_status=faiss_status,
    )
    output_path = args.output
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_path = output_path / "FAISS_DECISION_BENCHMARK.md"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote report: {output_path}")


if __name__ == "__main__":
    main()
