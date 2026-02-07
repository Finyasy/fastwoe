# Preprocessor Memory Benchmark

- Generated (UTC): `2026-02-07T10:52:35.573206+00:00`
- Methods requested: `['kmeans', 'tree', 'faiss']`
- Dataset sizes: `[10000, 100000]`
- Numeric features: `4`
- n_bins: `8`
- faiss available: `True`
- faiss status: `faiss import ok`

## Results

| method | rows | preprocess peak RSS | preprocess delta RSS | e2e peak RSS | e2e delta RSS |
|---|---:|---:|---:|---:|---:|
| kmeans | 10000 | 80.125 MB | 60.797 MB | 80.125 MB | 61.172 MB |
| kmeans | 100000 | 174.625 MB | 134.812 MB | 180.984 MB | 141.344 MB |
| tree | 10000 | 79.500 MB | 60.234 MB | 83.438 MB | 64.203 MB |
| tree | 100000 | 177.453 MB | 137.312 MB | 181.297 MB | 141.203 MB |
| faiss | 10000 | 88.906 MB | 69.406 MB | 90.781 MB | 71.750 MB |
| faiss | 100000 | 176.938 MB | 137.141 MB | 190.109 MB | 150.500 MB |
