from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_report(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "| method | rows | preprocess peak RSS | preprocess delta RSS "
                "| e2e peak RSS | e2e delta RSS |",
                "|---|---:|---:|---:|---:|---:|",
                "| kmeans | 10000 | 80.125 MB | 60.797 MB | 80.125 MB | 61.172 MB |",
                "| kmeans | 100000 | 174.625 MB | 134.812 MB | 180.984 MB | 141.344 MB |",
                "| faiss | 10000 | 88.906 MB | 69.406 MB | 90.781 MB | 71.750 MB |",
                "| faiss | 100000 | 176.938 MB | 137.141 MB | 190.109 MB | 150.500 MB |",
            ]
        ),
        encoding="utf-8",
    )


def test_faiss_memory_regression_cli_passes(tmp_path: Path) -> None:
    report = tmp_path / "PREPROCESSOR_MEMORY_BENCHMARK.md"
    _write_report(report)
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_faiss_memory_regression.py",
            "--report",
            str(report),
            "--sizes",
            "10000",
            "100000",
            "--max-pre-delta-ratio",
            "1.5",
            "--max-e2e-delta-ratio",
            "1.5",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "FAISS memory regression check passed." in proc.stdout


def test_faiss_memory_regression_cli_fails_when_ratio_exceeds(tmp_path: Path) -> None:
    report = tmp_path / "PREPROCESSOR_MEMORY_BENCHMARK.md"
    _write_report(report)
    proc = subprocess.run(
        [
            sys.executable,
            "tools/check_faiss_memory_regression.py",
            "--report",
            str(report),
            "--sizes",
            "10000",
            "--max-pre-delta-ratio",
            "1.1",
            "--max-e2e-delta-ratio",
            "1.1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "preprocess memory ratio too high" in proc.stderr
    assert "end-to-end memory ratio too high" in proc.stderr
