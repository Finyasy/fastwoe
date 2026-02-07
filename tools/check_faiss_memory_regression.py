#!/usr/bin/env python3
"""Validate FAISS memory ratios against kmeans from memory benchmark report."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^\|\s*(?P<method>[a-zA-Z0-9_]+)\s*\|\s*(?P<rows>\d+)\s*\|\s*"
    r"(?P<pre_peak>[0-9.]+)\s*MB\s*\|\s*(?P<pre_delta>[0-9.]+)\s*MB\s*\|\s*"
    r"(?P<e2e_peak>[0-9.]+)\s*MB\s*\|\s*(?P<e2e_delta>[0-9.]+)\s*MB\s*\|$"
)


def _parse_report(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    content = path.read_text(encoding="utf-8")
    out: dict[tuple[str, int], dict[str, float]] = {}

    for line in content.splitlines():
        match = ROW_RE.match(line.strip())
        if match is None:
            continue
        method = match.group("method")
        rows = int(match.group("rows"))
        out[(method, rows)] = {
            "pre_delta_mb": float(match.group("pre_delta")),
            "e2e_delta_mb": float(match.group("e2e_delta")),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check FAISS memory regression ratios against kmeans in benchmark report."
    )
    parser.add_argument("--report", required=True, type=Path, help="Report markdown file path.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10_000, 100_000],
        help="Dataset sizes that must exist in report.",
    )
    parser.add_argument(
        "--max-pre-delta-ratio",
        type=float,
        default=1.5,
        help="Maximum allowed faiss/kmeans preprocess memory-delta ratio.",
    )
    parser.add_argument(
        "--max-e2e-delta-ratio",
        type=float,
        default=1.5,
        help="Maximum allowed faiss/kmeans end-to-end memory-delta ratio.",
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"report not found: {args.report}")

    parsed = _parse_report(args.report)
    failures: list[str] = []

    for size in args.sizes:
        k_key = ("kmeans", size)
        f_key = ("faiss", size)
        if k_key not in parsed:
            failures.append(f"missing kmeans row for size={size}")
            continue
        if f_key not in parsed:
            failures.append(f"missing faiss row for size={size}")
            continue

        kmeans = parsed[k_key]
        faiss = parsed[f_key]
        if kmeans["pre_delta_mb"] <= 0.0 or kmeans["e2e_delta_mb"] <= 0.0:
            failures.append(f"invalid kmeans memory baseline at size={size}")
            continue

        pre_ratio = faiss["pre_delta_mb"] / kmeans["pre_delta_mb"]
        e2e_ratio = faiss["e2e_delta_mb"] / kmeans["e2e_delta_mb"]
        print(
            f"size={size} pre_delta_ratio={pre_ratio:.3f} "
            f"e2e_delta_ratio={e2e_ratio:.3f} "
            f"(thresholds: pre<={args.max_pre_delta_ratio}, e2e<={args.max_e2e_delta_ratio})"
        )

        if pre_ratio > args.max_pre_delta_ratio:
            failures.append(
                "size="
                f"{size} preprocess memory ratio too high: {pre_ratio:.3f} > "
                f"{args.max_pre_delta_ratio}"
            )
        if e2e_ratio > args.max_e2e_delta_ratio:
            failures.append(
                "size="
                f"{size} end-to-end memory ratio too high: {e2e_ratio:.3f} > "
                f"{args.max_e2e_delta_ratio}"
            )

    if failures:
        print("FAISS memory regression check failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        raise SystemExit(1)

    print("FAISS memory regression check passed.")


if __name__ == "__main__":
    main()
