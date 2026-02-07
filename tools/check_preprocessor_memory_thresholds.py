#!/usr/bin/env python3
"""Check preprocessor memory thresholds from benchmark markdown output."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROW_RE = re.compile(
    r"^\|\s*(?P<method>[a-z0-9_]+)\s*\|\s*(?P<rows>\d+)\s*\|"
    r"\s*(?P<pre_peak>[0-9.]+)\s*MB\s*\|"
    r"\s*(?P<pre_delta>[0-9.]+)\s*MB\s*\|"
    r"\s*(?P<e2e_peak>[0-9.]+)\s*MB\s*\|"
    r"\s*(?P<e2e_delta>[0-9.]+)\s*MB\s*\|$"
)


def _parse_report(report_path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    results: dict[tuple[str, int], tuple[float, float]] = {}
    for raw_line in report_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        match = ROW_RE.match(line)
        if not match:
            continue
        method = match.group("method")
        rows = int(match.group("rows"))
        pre_delta = float(match.group("pre_delta"))
        e2e_delta = float(match.group("e2e_delta"))
        results[(method, rows)] = (pre_delta, e2e_delta)
    return results


def _parse_threshold(raw: str) -> tuple[str, int, float, float]:
    # method:rows:max_pre_delta_mb:max_e2e_delta_mb
    parts = raw.split(":")
    if len(parts) != 4:
        raise ValueError(
            f"Invalid threshold '{raw}'. Expected method:rows:max_pre_delta_mb:max_e2e_delta_mb"
        )
    method = parts[0].strip()
    rows = int(parts[1])
    max_pre_delta_mb = float(parts[2])
    max_e2e_delta_mb = float(parts[3])
    return method, rows, max_pre_delta_mb, max_e2e_delta_mb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate benchmark markdown against memory thresholds."
    )
    parser.add_argument(
        "--report", required=True, type=Path, help="Benchmark markdown report path."
    )
    parser.add_argument(
        "--threshold",
        action="append",
        required=True,
        help="Threshold spec: method:rows:max_pre_delta_mb:max_e2e_delta_mb",
    )
    args = parser.parse_args()

    values = _parse_report(args.report)
    failures: list[str] = []

    for threshold in args.threshold:
        method, rows, max_pre_delta_mb, max_e2e_delta_mb = _parse_threshold(threshold)
        key = (method, rows)
        if key not in values:
            failures.append(f"Missing benchmark row for method={method} rows={rows}.")
            continue
        pre_delta_mb, e2e_delta_mb = values[key]
        print(
            f"{method}:{rows} pre_delta_mb={pre_delta_mb:.3f}/{max_pre_delta_mb:.3f} "
            f"e2e_delta_mb={e2e_delta_mb:.3f}/{max_e2e_delta_mb:.3f}"
        )
        if pre_delta_mb > max_pre_delta_mb:
            failures.append(
                f"{method}:{rows} preprocess memory delta too high: "
                f"{pre_delta_mb:.3f} > {max_pre_delta_mb:.3f} MB"
            )
        if e2e_delta_mb > max_e2e_delta_mb:
            failures.append(
                f"{method}:{rows} end-to-end memory delta too high: "
                f"{e2e_delta_mb:.3f} > {max_e2e_delta_mb:.3f} MB"
            )

    if failures:
        print("Memory threshold check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        raise SystemExit(1)

    print("Memory threshold check passed.")


if __name__ == "__main__":
    main()
