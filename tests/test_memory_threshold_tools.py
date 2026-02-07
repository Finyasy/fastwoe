from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module(path: str):
    module_path = Path(path)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_memory_threshold_parser_extracts_rows(tmp_path: Path) -> None:
    module = _load_module("tools/check_preprocessor_memory_thresholds.py")
    report = tmp_path / "PREPROCESSOR_MEMORY_BENCHMARK.md"
    report.write_text(
        "\n".join(
            [
                "| method | rows | preprocess peak RSS | preprocess delta RSS "
                "| e2e peak RSS | e2e delta RSS |",
                "|---|---:|---:|---:|---:|---:|",
                "| kmeans | 10000 | 110.000 MB | 16.000 MB | 123.000 MB | 25.000 MB |",
                "| tree | 10000 | 108.000 MB | 18.000 MB | 121.000 MB | 27.000 MB |",
            ]
        ),
        encoding="utf-8",
    )
    parsed = module._parse_report(report)
    assert parsed[("kmeans", 10000)] == (16.0, 25.0)
    assert parsed[("tree", 10000)] == (18.0, 27.0)


def test_memory_threshold_cli_passes(tmp_path: Path) -> None:
    report = tmp_path / "PREPROCESSOR_MEMORY_BENCHMARK.md"
    report.write_text(
        "\n".join(
            [
                "| method | rows | preprocess peak RSS | preprocess delta RSS "
                "| e2e peak RSS | e2e delta RSS |",
                "|---|---:|---:|---:|---:|---:|",
                "| kmeans | 10000 | 110.000 MB | 16.000 MB | 123.000 MB | 25.000 MB |",
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_preprocessor_memory_thresholds.py",
            "--report",
            str(report),
            "--threshold",
            "kmeans:10000:20:30",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Memory threshold check passed." in proc.stdout


def test_memory_threshold_cli_fails_on_exceeding_delta(tmp_path: Path) -> None:
    report = tmp_path / "PREPROCESSOR_MEMORY_BENCHMARK.md"
    report.write_text(
        "\n".join(
            [
                "| method | rows | preprocess peak RSS | preprocess delta RSS "
                "| e2e peak RSS | e2e delta RSS |",
                "|---|---:|---:|---:|---:|---:|",
                "| tree | 10000 | 108.000 MB | 18.000 MB | 121.000 MB | 27.000 MB |",
            ]
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_preprocessor_memory_thresholds.py",
            "--report",
            str(report),
            "--threshold",
            "tree:10000:10:26",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "preprocess memory delta too high" in proc.stderr
    assert "end-to-end memory delta too high" in proc.stderr
