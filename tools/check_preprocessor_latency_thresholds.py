#!/usr/bin/env python3
"""Check preprocess and end-to-end latency thresholds from benchmark markdown output."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^\|\s*(?P<method>[a-zA-Z0-9_]+)\s*\|\s*(?P<rows>\d+)\s*\|\s*"
    r"(?P<pre_ms>\d+(?:\.\d+)?) ms\s*\|\s*(?P<median_ms>\d+(?:\.\d+)?) ms\s*\|\s*"
    r"(?P<rows_ps>[0-9,]+)\s*\|\s*(?P<e2e_ms>\d+(?:\.\d+)?) ms\s*\|$"
)


def _parse_rows(path: Path) -> dict[tuple[str, int], dict[str, float]]:
    out: dict[tuple[str, int], dict[str, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ROW_RE.match(line.strip())
        if match is None:
            continue
        method = match.group("method")
        rows = int(match.group("rows"))
        out[(method, rows)] = {
            "pre_ms": float(match.group("pre_ms")),
            "e2e_ms": float(match.group("e2e_ms")),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark markdown against latency thresholds."
    )
    parser.add_argument("--report", required=True, type=Path, help="Benchmark markdown path.")
    parser.add_argument(
        "--threshold",
        action="append",
        required=True,
        help=(
            "Threshold spec 'method:size:max_pre_ms:max_e2e_ms'. "
            "Example: kmeans:10000:250:350"
        ),
    )
    args = parser.parse_args()

    if not args.report.exists():
        raise FileNotFoundError(f"report not found: {args.report}")

    parsed = _parse_rows(args.report)
    failures: list[str] = []

    for spec in args.threshold:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"Invalid --threshold spec '{spec}'. Expected method:size:max_pre_ms:max_e2e_ms"
            )

        method = parts[0]
        size = int(parts[1])
        max_pre_ms = float(parts[2])
        max_e2e_ms = float(parts[3])
        key = (method, size)

        if key not in parsed:
            failures.append(f"missing benchmark row for {method}:{size}")
            continue

        row = parsed[key]
        pre_ms = row["pre_ms"]
        e2e_ms = row["e2e_ms"]
        print(
            f"{method}:{size} pre_ms={pre_ms:.3f}/{max_pre_ms:.3f} "
            f"e2e_ms={e2e_ms:.3f}/{max_e2e_ms:.3f}"
        )

        if pre_ms > max_pre_ms:
            failures.append(
                f"{method}:{size} preprocess latency too high: {pre_ms:.3f} > {max_pre_ms:.3f}"
            )
        if e2e_ms > max_e2e_ms:
            failures.append(
                f"{method}:{size} end-to-end latency too high: {e2e_ms:.3f} > {max_e2e_ms:.3f}"
            )

    if failures:
        print("Latency threshold check failed:", file=sys.stderr)
        for item in failures:
            print(f"- {item}", file=sys.stderr)
        raise SystemExit(1)

    print("Latency threshold check passed.")


if __name__ == "__main__":
    main()
