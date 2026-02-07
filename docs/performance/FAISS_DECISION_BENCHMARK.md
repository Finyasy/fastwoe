# FAISS Decision Benchmark

- Generated (UTC): `2026-02-07T09:58:08.222539+00:00`
- Methods requested: `['kmeans', 'tree', 'faiss']`
- Dataset sizes: `[10000, 100000]`
- Numeric features: `4`
- n_bins: `8`
- Warmup: `2`
- Repeats: `5`
- faiss available: `True`
- faiss status: `faiss import ok`

## Results

| method | rows | preprocess best | preprocess median | preprocess rows/s | e2e best |
|---|---:|---:|---:|---:|---:|
| kmeans | 10000 | 32.126 ms | 37.581 ms | 311,270 | 49.710 ms |
| kmeans | 100000 | 453.994 ms | 459.760 ms | 220,267 | 616.789 ms |
| tree | 10000 | 27.901 ms | 34.423 ms | 358,412 | 45.171 ms |
| tree | 100000 | 411.808 ms | 427.478 ms | 242,831 | 564.883 ms |
| faiss | 10000 | 47.869 ms | 48.463 ms | 208,903 | 58.275 ms |
| faiss | 100000 | 493.762 ms | 501.890 ms | 202,527 | 650.255 ms |

## Decision

Do not implement Rust-core FAISS yet: gains do not clear the threshold (>=20% preprocess and >=10% end-to-end).
