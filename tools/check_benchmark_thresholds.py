"""Check Criterion throughput output against minimum thresholds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

THROUGHPUT_RE = re.compile(r"thrpt:\s*\[\s*([0-9]+(?:\.[0-9]+)?)\s*([KMG]?elem/s)")

UNIT_MULTIPLIER: dict[str, float] = {
    "elem/s": 1.0,
    "Kelem/s": 1_000.0,
    "Melem/s": 1_000_000.0,
    "Gelem/s": 1_000_000_000.0,
}


def _to_elems_per_sec(value: float, unit: str) -> float:
    if unit not in UNIT_MULTIPLIER:
        raise ValueError(f"Unsupported throughput unit: {unit}")
    return value * UNIT_MULTIPLIER[unit]


def _extract_lower_throughput(lines: list[str], target: str) -> float:
    for idx, line in enumerate(lines):
        if line.strip() != target:
            continue
        for lookahead in lines[idx + 1 : idx + 40]:
            if "change:" in lookahead:
                break
            match = THROUGHPUT_RE.search(lookahead)
            if match:
                value = float(match.group(1))
                unit = match.group(2)
                return _to_elems_per_sec(value, unit)
    raise ValueError(f"Target throughput block not found or unparsable: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Criterion benchmark throughput.")
    parser.add_argument("--file", required=True, help="Path to Criterion output log.")
    parser.add_argument("--target", required=True, help="Criterion benchmark target identifier.")
    parser.add_argument(
        "--min-elems-per-sec",
        required=True,
        type=float,
        help="Minimum acceptable lower-bound throughput in elements/sec.",
    )
    args = parser.parse_args()

    log_path = Path(args.file)
    content = log_path.read_text(encoding="utf-8")
    throughput = _extract_lower_throughput(content.splitlines(), args.target)
    threshold = float(args.min_elems_per_sec)

    print(
        f"target={args.target} throughput={throughput:.2f} elem/s "
        f"threshold={threshold:.2f} elem/s"
    )
    if throughput < threshold:
        raise SystemExit(
            f"Benchmark threshold failed for {args.target}: "
            f"{throughput:.2f} < {threshold:.2f} elem/s"
        )


if __name__ == "__main__":
    main()

