#!/usr/bin/env python3
"""Benchmark preprocessing and end-to-end latency on a real CSV dataset."""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from fastwoe import FastWoe, WoePreprocessor


@dataclass
class BenchResult:
    method: str
    rows: int
    preprocess_best_ms: float
    preprocess_median_ms: float
    e2e_best_ms: float


def _time_many(fn: Any, *args: Any, warmup: int, repeats: int, **kwargs: Any) -> list[float]:
    for _ in range(warmup):
        fn(*args, **kwargs)
    out: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def _run_single(
    *,
    method: str,
    x: pd.DataFrame,
    y: list[int],
    n_bins: int,
    warmup: int,
    repeats: int,
) -> BenchResult:
    numeric_cols = [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
    cat_cols = [c for c in x.columns if c not in numeric_cols]

    kwargs: dict[str, Any] = {
        "numerical_features": numeric_cols if numeric_cols else None,
        "cat_features": cat_cols if cat_cols else None,
    }
    if method == "tree":
        kwargs["target"] = y

    def run_preprocess() -> Any:
        pre = WoePreprocessor(n_bins=n_bins, binning_method=method)
        return pre.fit_transform(x, **kwargs)

    pre_times = _time_many(run_preprocess, warmup=warmup, repeats=repeats)

    def run_e2e() -> None:
        pre = WoePreprocessor(n_bins=n_bins, binning_method=method)
        transformed = pre.fit_transform(x, **kwargs)
        model = FastWoe()
        model.fit_matrix(transformed, y, feature_names=list(x.columns))
        _ = model.predict_proba_matrix(transformed)

    e2e_times = _time_many(run_e2e, warmup=warmup, repeats=repeats)

    return BenchResult(
        method=method,
        rows=len(x),
        preprocess_best_ms=min(pre_times),
        preprocess_median_ms=statistics.median(pre_times),
        e2e_best_ms=min(e2e_times),
    )


def _parse_threshold(raw: str) -> tuple[str, float, float]:
    # method:max_pre_ms:max_e2e_ms
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid threshold '{raw}', expected method:max_pre_ms:max_e2e_ms")
    return parts[0], float(parts[1]), float(parts[2])


def _render_markdown(
    *,
    dataset_path: Path,
    target_col: str,
    results: list[BenchResult],
    n_bins: int,
    warmup: int,
    repeats: int,
) -> str:
    lines = [
        "# Real Dataset Benchmark",
        "",
        f"- dataset: `{dataset_path}`",
        f"- target column: `{target_col}`",
        f"- n_bins: `{n_bins}`",
        f"- warmup: `{warmup}`",
        f"- repeats: `{repeats}`",
        "",
        "| method | rows | preprocess best | preprocess median | e2e best |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result.method} | {result.rows} | "
            f"{result.preprocess_best_ms:.3f} ms | {result.preprocess_median_ms:.3f} ms | "
            f"{result.e2e_best_ms:.3f} ms |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fastwoe preprocessing/end-to-end on a real CSV dataset."
    )
    parser.add_argument("--input-csv", required=True, type=Path, help="Input CSV file.")
    parser.add_argument("--target-col", required=True, help="Target column name.")
    parser.add_argument(
        "--methods", nargs="+", default=["kmeans", "tree"], help="Binning methods to benchmark."
    )
    parser.add_argument("--n-bins", type=int, default=8, help="Number of bins.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs per method.")
    parser.add_argument("--repeats", type=int, default=3, help="Measured runs per method.")
    parser.add_argument(
        "--threshold",
        action="append",
        default=[],
        help="Threshold spec: method:max_pre_ms:max_e2e_ms",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-artifacts/REAL_DATASET_BENCHMARK.md"),
        help="Output markdown path (or directory).",
    )
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"input csv not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if args.target_col not in df.columns:
        raise ValueError(f"target column '{args.target_col}' not found in CSV")

    y_series = pd.to_numeric(df[args.target_col], errors="raise")
    y = [int(v) for v in y_series.tolist()]
    x = df.drop(columns=[args.target_col])
    if x.empty:
        raise ValueError("no feature columns left after dropping target column")

    results: list[BenchResult] = []
    for method in args.methods:
        print(f"benchmarking method={method} rows={len(x)} ...")
        results.append(
            _run_single(
                method=method,
                x=x,
                y=y,
                n_bins=args.n_bins,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

    by_method = {r.method: r for r in results}
    failures: list[str] = []
    for raw in args.threshold:
        method, max_pre_ms, max_e2e_ms = _parse_threshold(raw)
        if method not in by_method:
            failures.append(f"missing result row for method={method}")
            continue
        row = by_method[method]
        print(
            f"{method}: pre_ms={row.preprocess_best_ms:.3f}/{max_pre_ms:.3f} "
            f"e2e_ms={row.e2e_best_ms:.3f}/{max_e2e_ms:.3f}"
        )
        if row.preprocess_best_ms > max_pre_ms:
            failures.append(
                f"{method} preprocess too slow: {row.preprocess_best_ms:.3f} > {max_pre_ms:.3f} ms"
            )
        if row.e2e_best_ms > max_e2e_ms:
            failures.append(f"{method} e2e too slow: {row.e2e_best_ms:.3f} > {max_e2e_ms:.3f} ms")

    output_path = args.output
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_path = output_path / "REAL_DATASET_BENCHMARK.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_markdown(
            dataset_path=args.input_csv,
            target_col=args.target_col,
            results=results,
            n_bins=args.n_bins,
            warmup=args.warmup,
            repeats=args.repeats,
        ),
        encoding="utf-8",
    )
    print(f"wrote report: {output_path}")

    if failures:
        for failure in failures:
            print(f"threshold-failure: {failure}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
