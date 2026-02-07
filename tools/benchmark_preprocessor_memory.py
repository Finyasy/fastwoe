#!/usr/bin/env python3
"""Benchmark preprocessor memory usage for kmeans/tree/faiss methods."""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import importlib.util
import io
import json
import random
import resource
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastwoe import FastWoe, WoePreprocessor


@dataclass
class MemoryResult:
    method: str
    rows: int
    preprocess_peak_mb: float
    preprocess_delta_mb: float
    e2e_peak_mb: float
    e2e_delta_mb: float


def _max_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KB, macOS reports bytes.
    if sys.platform == "darwin":
        return raw / (1024.0 * 1024.0)
    return raw / 1024.0


def _faiss_import_status() -> tuple[bool, str]:
    if importlib.util.find_spec("faiss") is None:
        return False, "faiss module not found"
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module("faiss")
    except Exception as exc:  # pragma: no cover - environment-dependent.
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


def _run_preprocess_memory(
    *,
    method: str,
    rows: list[list[float]],
    y: list[int],
    n_bins: int,
) -> tuple[float, float]:
    kwargs: dict[str, Any] = {"numerical_features": list(range(len(rows[0])))}
    if method == "tree":
        kwargs["target"] = y
    gc.collect()
    before = _max_rss_mb()
    pre = WoePreprocessor(n_bins=n_bins, binning_method=method)
    _ = pre.fit_transform(rows, **kwargs)
    gc.collect()
    after = _max_rss_mb()
    return after, max(0.0, after - before)


def _run_e2e_memory(
    *,
    method: str,
    rows: list[list[float]],
    y: list[int],
    n_bins: int,
) -> tuple[float, float]:
    kwargs: dict[str, Any] = {"numerical_features": list(range(len(rows[0])))}
    if method == "tree":
        kwargs["target"] = y
    feature_names = [f"f{i}" for i in range(len(rows[0]))]
    gc.collect()
    before = _max_rss_mb()
    pre = WoePreprocessor(n_bins=n_bins, binning_method=method)
    transformed = pre.fit_transform(rows, **kwargs)
    model = FastWoe()
    model.fit_matrix(transformed, y, feature_names=feature_names)
    _ = model.predict_proba_matrix(transformed)
    gc.collect()
    after = _max_rss_mb()
    return after, max(0.0, after - before)


def _run_worker(args: argparse.Namespace) -> int:
    rows = _make_numeric_rows(args.rows, args.n_features)
    y = _make_binary_target(args.rows)
    if args.worker_mode == "preprocess":
        peak_mb, delta_mb = _run_preprocess_memory(
            method=args.method, rows=rows, y=y, n_bins=args.n_bins
        )
    else:
        peak_mb, delta_mb = _run_e2e_memory(
            method=args.method, rows=rows, y=y, n_bins=args.n_bins
        )
    print(json.dumps({"peak_mb": peak_mb, "delta_mb": delta_mb}))
    return 0


def _run_worker_subprocess(
    *,
    script_path: Path,
    method: str,
    rows: int,
    n_bins: int,
    n_features: int,
    worker_mode: str,
) -> tuple[float, float]:
    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        "--worker-mode",
        worker_mode,
        "--method",
        method,
        "--rows",
        str(rows),
        "--n-bins",
        str(n_bins),
        "--n-features",
        str(n_features),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    payload = json.loads(proc.stdout.strip())
    return float(payload["peak_mb"]), float(payload["delta_mb"])


def _format_mb(value: float) -> str:
    return f"{value:.3f} MB"


def _render_markdown(
    *,
    results: dict[tuple[str, int], MemoryResult],
    methods: list[str],
    sizes: list[int],
    n_bins: int,
    n_features: int,
    faiss_available: bool,
    faiss_status: str,
) -> str:
    now = datetime.now(timezone.utc).isoformat()
    lines = [
        "# Preprocessor Memory Benchmark",
        "",
        f"- Generated (UTC): `{now}`",
        f"- Methods requested: `{methods}`",
        f"- Dataset sizes: `{sizes}`",
        f"- Numeric features: `{n_features}`",
        f"- n_bins: `{n_bins}`",
        f"- faiss available: `{faiss_available}`",
        f"- faiss status: `{faiss_status}`",
        "",
        "## Results",
        "",
        "| method | rows | preprocess peak RSS | preprocess delta RSS "
        "| e2e peak RSS | e2e delta RSS |",
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
                f"{_format_mb(res.preprocess_peak_mb)} | "
                f"{_format_mb(res.preprocess_delta_mb)} | "
                f"{_format_mb(res.e2e_peak_mb)} | "
                f"{_format_mb(res.e2e_delta_mb)} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark memory usage for kmeans/tree/faiss preprocessor methods."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-mode",
        choices=("preprocess", "e2e"),
        default="preprocess",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--method", default="kmeans", help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, default=10000, help=argparse.SUPPRESS)
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/performance/PREPROCESSOR_MEMORY_BENCHMARK.md"),
        help="Path to write markdown report.",
    )
    args = parser.parse_args()

    if args.worker:
        raise SystemExit(_run_worker(args))

    faiss_available, faiss_status = _faiss_import_status()
    if "faiss" in set(args.methods) and not faiss_available:
        print(f"faiss unavailable; faiss memory benchmarks will be skipped ({faiss_status}).")

    results: dict[tuple[str, int], MemoryResult] = {}
    script_path = Path(__file__)
    for size in args.sizes:
        for method in args.methods:
            if method == "faiss" and not faiss_available:
                continue
            print(f"memory benchmarking method={method} rows={size} ...")
            pre_peak_mb, pre_delta_mb = _run_worker_subprocess(
                script_path=script_path,
                method=method,
                rows=size,
                n_bins=args.n_bins,
                n_features=args.n_features,
                worker_mode="preprocess",
            )
            e2e_peak_mb, e2e_delta_mb = _run_worker_subprocess(
                script_path=script_path,
                method=method,
                rows=size,
                n_bins=args.n_bins,
                n_features=args.n_features,
                worker_mode="e2e",
            )
            results[(method, size)] = MemoryResult(
                method=method,
                rows=size,
                preprocess_peak_mb=pre_peak_mb,
                preprocess_delta_mb=pre_delta_mb,
                e2e_peak_mb=e2e_peak_mb,
                e2e_delta_mb=e2e_delta_mb,
            )

    report = _render_markdown(
        results=results,
        methods=args.methods,
        sizes=args.sizes,
        n_bins=args.n_bins,
        n_features=args.n_features,
        faiss_available=faiss_available,
        faiss_status=faiss_status,
    )
    output_path = args.output
    if (output_path.exists() and output_path.is_dir()) or output_path.suffix == "":
        output_path = output_path / "PREPROCESSOR_MEMORY_BENCHMARK.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"wrote report: {output_path}")


if __name__ == "__main__":
    main()
