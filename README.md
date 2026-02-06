# fastwoe
Fast Weight of Evidence (WOE) Encoding and Inference

This repository is scaffolded as a Rust workspace with PyO3 bindings for Python.

## Workspace
- `crates/fastwoe-core`: pure Rust WOE/statistics engine.
- `crates/fastwoe-py`: PyO3 extension module (`fastwoe_rs`).

## Prerequisites
1. Install Rust (stable) with rustup.
2. Install Python 3.9+.
3. Install maturin:
   `python -m pip install maturin`

## Local Development
1. Rust checks:
   `cargo fmt --all`
   `cargo clippy --all-targets --all-features -D warnings`
   `cargo test --all-features`
2. Build/install Python extension in active environment:
   `maturin develop --manifest-path crates/fastwoe-py/Cargo.toml`

## Python Tooling
Ruff and Python dev settings are configured in `pyproject.toml`.

## Quick Python Usage
```python
from fastwoe import FastWoe

model = FastWoe(smoothing=0.5, default_woe=0.0)
categories = ["A", "A", "B", "C"]
target = [1, 0, 0, 1]

model.fit(categories, target)
woe_values = model.transform(["A", "B", "Z"])
proba = model.predict_proba(["A", "B", "Z"])
mapping = model.get_mapping()
```

## Multi-Feature API (Categorical Matrix)
```python
from fastwoe import FastWoe

model = FastWoe()
rows = [
    ["A", "x"],
    ["A", "y"],
    ["B", "x"],
    ["C", "z"],
]
target = [1, 0, 0, 1]

model.fit_matrix(rows, target, feature_names=["cat", "bucket"])
X_woe = model.transform_matrix(rows)
proba = model.predict_proba_matrix(rows)
cat_mapping = model.get_feature_mapping("cat")
```

## Multiclass One-vs-Rest API
```python
from fastwoe import FastWoe

model = FastWoe(smoothing=0.5, default_woe=0.0)

rows = [
    ["A", "x"],
    ["A", "y"],
    ["B", "x"],
    ["C", "z"],
    ["B", "y"],
]
labels = ["c0", "c1", "c2", "c0", "c1"]

model.fit_matrix_multiclass(rows, labels, feature_names=["cat", "bucket"])
all_probs = model.predict_proba_matrix_multiclass(rows)  # shape: (n_rows, n_classes)
c1_probs = model.predict_proba_matrix_class(rows, "c1")
classes = model.get_class_labels()

# Feature mapping for a specific class (one-vs-rest)
cat_mapping_for_c0 = model.get_feature_mapping_multiclass("c0", "cat")
```
