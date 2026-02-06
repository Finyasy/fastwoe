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

`FastWoe` accepts Python lists, NumPy arrays, pandas Series, and pandas DataFrames.

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
X_woe_multi = model.transform_matrix_multiclass(rows)
woe_feature_names = model.get_feature_names_multiclass()

# Feature mapping for a specific class (one-vs-rest)
cat_mapping_for_c0 = model.get_feature_mapping_multiclass("c0", "cat")
```

## Confidence Intervals
```python
from fastwoe import FastWoe

model = FastWoe()
model.fit(["A", "B", "A"], [1, 0, 1])
ci = model.predict_ci(["A", "Z"], alpha=0.05)
# [(prediction, lower_ci, upper_ci), ...]

# Matrix APIs
rows = [["A", "x"], ["B", "y"]]
model.fit_matrix(rows, [1, 0], feature_names=["cat", "bucket"])
ci_matrix = model.predict_ci_matrix(rows, alpha=0.05)

# Multiclass APIs
model.fit_matrix_multiclass(rows, ["c0", "c1"], feature_names=["cat", "bucket"])
ci_multi = model.predict_ci_matrix_multiclass(rows, alpha=0.05)
ci_c0 = model.predict_ci_matrix_class(rows, "c0", alpha=0.05)
```

## IV Analysis (Credit-Scoring Focus)
```python
from fastwoe import FastWoe

rows = [["A", "x"], ["A", "y"], ["B", "x"], ["C", "z"]]
target = [1, 0, 0, 1]

model = FastWoe()
model.fit_matrix(rows, target, feature_names=["cat", "bucket"])

# Per-feature Information Value with standard error + CI
iv_rows = model.get_iv_analysis(alpha=0.05)
iv_cat_only = model.get_iv_analysis(feature_name="cat", alpha=0.05)

# DataFrame output for reporting pipelines
iv_df = model.get_iv_analysis(as_frame=True)

# Multiclass one-vs-rest IV analysis for a specific class label
model.fit_matrix_multiclass(rows, ["c0", "c1", "c2", "c0"], feature_names=["cat", "bucket"])
iv_c0 = model.get_iv_analysis_multiclass("c0", alpha=0.05)
```

## Pandas Output Mode
```python
import pandas as pd
from fastwoe import FastWoe

X = pd.DataFrame({"cat": ["A", "B"], "bucket": ["x", "y"]})
y = [1, 0]

model = FastWoe()
model.fit_matrix(X, y, feature_names=X.columns)

X_woe_df = model.transform_matrix(X, as_frame=True)
ci_df = model.predict_ci_matrix(X, as_frame=True)
proba_multi_df = model.predict_proba_matrix_multiclass(X, as_frame=True)
```

## Performance Guidance
- Build extension wheels in optimized mode:
  `python -m maturin build --release --manifest-path crates/fastwoe-py/Cargo.toml`
- Run core performance benchmarks:
  `cargo bench -p fastwoe-core --bench woe_simulation`
- Release profile is tuned for runtime speed (`lto=fat`, `codegen-units=1`, stripped symbols).

## CI and Release
- CI workflow: `.github/workflows/ci.yml`
- Wheels workflow: `.github/workflows/wheels.yml`
- Benchmark workflow: `.github/workflows/benchmarks.yml`
- Release checklist: `docs/release/RELEASE_CHECKLIST.md`
