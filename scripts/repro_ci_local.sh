#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-fastwoe-faiss}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv-ci-local"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this script (conda command not found)." >&2
  exit 1
fi

echo "[repro] Creating local CI venv from conda env: ${ENV_NAME}"
rm -rf "${VENV_DIR}"
conda run -n "${ENV_NAME}" python -m venv "${VENV_DIR}" --system-site-packages

CONDA_BIN="$(
  conda run -n "${ENV_NAME}" python -c \
    "import pathlib, sys; print(pathlib.Path(sys.executable).parent)" | tail -n 1
)"
export PATH="${HOME}/.cargo/bin:${CONDA_BIN}:${PATH}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "cargo not found on PATH. Install Rust with rustup and ensure ~/.cargo/bin is available." >&2
  exit 1
fi

if ! command -v maturin >/dev/null 2>&1; then
  echo "maturin not found on PATH. Install it in conda env '${ENV_NAME}' or add it to PATH." >&2
  exit 1
fi

cd "${REPO_ROOT}"

echo "[repro] Building wheel"
maturin build --release --manifest-path crates/fastwoe-py/Cargo.toml --out dist -i "${VENV_DIR}/bin/python"

echo "[repro] Installing wheel into local CI venv"
"${VENV_DIR}/bin/python" -m pip install --force-reinstall --no-deps dist/*.whl

echo "[repro] Running parity + preprocessor + invariants tests"
"${VENV_DIR}/bin/python" -m pytest -q \
  tests/test_phase0_parity_contract.py \
  tests/test_preprocessor.py \
  tests/test_invariants.py

echo "[repro] Running end-to-end latency benchmark and threshold gate"
"${VENV_DIR}/bin/python" tools/benchmark_faiss_decision.py \
  --methods kmeans tree \
  --sizes 10000 \
  --warmup 1 \
  --repeats 3 \
  --output benchmark-artifacts/
"${VENV_DIR}/bin/python" tools/check_preprocessor_latency_thresholds.py \
  --report benchmark-artifacts/FAISS_DECISION_BENCHMARK.md \
  --threshold kmeans:10000:120:180 \
  --threshold tree:10000:120:160

echo "[repro] Running end-to-end memory benchmark and threshold gate"
"${VENV_DIR}/bin/python" tools/benchmark_preprocessor_memory.py \
  --methods kmeans tree \
  --sizes 10000 \
  --output benchmark-artifacts/
"${VENV_DIR}/bin/python" tools/check_preprocessor_memory_thresholds.py \
  --report benchmark-artifacts/PREPROCESSOR_MEMORY_BENCHMARK.md \
  --threshold kmeans:10000:150:190 \
  --threshold tree:10000:150:190

echo "[repro] Local CI-equivalent flow completed successfully."
