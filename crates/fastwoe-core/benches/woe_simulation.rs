//! Simulation benchmarks for FastWOE: fit, transform, predict at scale.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastwoe_core::{
    BinaryTabularWoeModel, MulticlassTabularWoeModel, NumericBinnerCore, PreprocessorCore,
};

fn make_categories(n_rows: usize, n_cols: usize, cardinality_per_col: usize) -> Vec<Vec<String>> {
    (0..n_rows)
        .map(|row| {
            (0..n_cols)
                .map(|col| format!("cat_{}_{}", col, row % cardinality_per_col))
                .collect()
        })
        .collect()
}

fn make_binary_targets(n_rows: usize, event_rate: f64) -> Vec<u8> {
    (0..n_rows)
        .map(|i| {
            if (i % 100) < (event_rate * 100.0) as usize {
                1
            } else {
                0
            }
        })
        .collect()
}

fn make_multiclass_labels(n_rows: usize, n_classes: usize) -> Vec<String> {
    (0..n_rows)
        .map(|i| format!("class_{}", i % n_classes))
        .collect()
}

fn make_numeric_matrix(
    n_rows: usize,
    n_cols: usize,
    max_value_mod: usize,
    missing_every: usize,
) -> Vec<Vec<Option<f64>>> {
    (0..n_rows)
        .map(|row| {
            (0..n_cols)
                .map(|col| {
                    if missing_every > 0 && (row + col) % missing_every == 0 {
                        None
                    } else {
                        Some(((row % max_value_mod) as f64) + (col as f64 * 0.01))
                    }
                })
                .collect()
        })
        .collect()
}

fn numeric_feature_names(n_cols: usize) -> Vec<String> {
    (0..n_cols).map(|i| format!("num_{i}")).collect()
}

fn numeric_feature_indices(n_cols: usize) -> Vec<usize> {
    (0..n_cols).collect()
}

fn bench_binary_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_fit");
    for size in [1_000, 10_000, 50_000, 100_000, 250_000] {
        let rows = make_categories(size, 5, 20);
        let targets = make_binary_targets(size, 0.3);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("fit_matrix", size),
            &(rows, targets),
            |b, (rows, targets)| {
                b.iter(|| {
                    let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
                    model
                        .fit_matrix(black_box(rows), black_box(targets), None)
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_binary_transform(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_transform");
    for size in [1_000, 10_000, 50_000, 100_000, 250_000] {
        let rows = make_categories(size, 5, 20);
        let targets = make_binary_targets(size, 0.3);
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
        model.fit_matrix(&rows, &targets, None).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("transform_matrix", size),
            &rows,
            |b, rows| {
                b.iter(|| model.transform_matrix(black_box(rows)).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_binary_predict_proba(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_predict_proba");
    for size in [1_000, 10_000, 50_000, 100_000, 250_000] {
        let rows = make_categories(size, 5, 20);
        let targets = make_binary_targets(size, 0.3);
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
        model.fit_matrix(&rows, &targets, None).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("predict_proba_matrix", size),
            &rows,
            |b, rows| {
                b.iter(|| model.predict_proba_matrix(black_box(rows)).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_binary_predict_ci(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_predict_ci");
    for size in [1_000, 10_000, 50_000, 100_000] {
        let rows = make_categories(size, 5, 20);
        let targets = make_binary_targets(size, 0.3);
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
        model.fit_matrix(&rows, &targets, None).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("predict_ci_matrix", size),
            &rows,
            |b, rows| {
                b.iter(|| model.predict_ci_matrix(black_box(rows), 0.05).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_fit_transform_combined(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_fit_transform");
    for size in [1_000, 10_000, 50_000, 100_000] {
        let rows = make_categories(size, 5, 20);
        let targets = make_binary_targets(size, 0.3);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("fit_transform_matrix", size),
            &(rows, targets),
            |b, (rows, targets)| {
                b.iter(|| {
                    let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
                    model
                        .fit_transform_matrix(black_box(rows), black_box(targets), None)
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn bench_multiclass_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiclass_fit");
    for size in [1_000, 10_000, 50_000] {
        let rows = make_categories(size, 5, 20);
        let labels = make_multiclass_labels(size, 5);
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("fit_matrix", size),
            &(rows, labels),
            |b, (rows, labels)| {
                b.iter(|| {
                    let mut model = MulticlassTabularWoeModel::new();
                    model
                        .fit_matrix(black_box(rows), black_box(labels), None, 0.5, 0.0)
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_multiclass_predict_proba(c: &mut Criterion) {
    let mut group = c.benchmark_group("multiclass_predict_proba");
    for size in [1_000, 10_000, 50_000] {
        let rows = make_categories(size, 5, 20);
        let labels = make_multiclass_labels(size, 5);
        let mut model = MulticlassTabularWoeModel::new();
        model.fit_matrix(&rows, &labels, None, 0.5, 0.0).unwrap();

        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("predict_proba_matrix", size),
            &rows,
            |b, rows| {
                b.iter(|| model.predict_proba_matrix(black_box(rows)).unwrap());
            },
        );
    }
    group.finish();
}

fn bench_scaling_columns(c: &mut Criterion) {
    const ROWS: usize = 20_000;
    let mut group = c.benchmark_group("binary_scaling_columns");
    for n_cols in [2, 5, 10, 20, 50] {
        let rows = make_categories(ROWS, n_cols, 15);
        let targets = make_binary_targets(ROWS, 0.3);
        group.throughput(Throughput::Elements((ROWS * n_cols) as u64));
        group.bench_with_input(
            BenchmarkId::new("fit_matrix", n_cols),
            &(rows, targets),
            |b, (rows, targets)| {
                b.iter(|| {
                    let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
                    model
                        .fit_matrix(black_box(rows), black_box(targets), None)
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_preprocessor_categorical(c: &mut Criterion) {
    const N_COLS: usize = 4;
    let mut group = c.benchmark_group("preprocessor_categorical");
    for size in [10_000, 100_000] {
        let rows = make_categories(size, N_COLS, 300);
        let feature_names = (0..N_COLS)
            .map(|i| format!("cat_{i}"))
            .collect::<Vec<String>>();
        let selected_idxs = (0..N_COLS).collect::<Vec<usize>>();

        group.throughput(Throughput::Elements((size * N_COLS) as u64));
        group.bench_with_input(
            BenchmarkId::new("fit", size),
            &(rows.clone(), feature_names.clone(), selected_idxs.clone()),
            |b, (rows, feature_names, selected_idxs)| {
                b.iter(|| {
                    let mut pre = PreprocessorCore::new(
                        Some(50),
                        0.95,
                        2,
                        "__other__".to_string(),
                        "__missing__".to_string(),
                    )
                    .unwrap();
                    pre.fit(
                        black_box(rows),
                        black_box(feature_names),
                        black_box(selected_idxs),
                    )
                    .unwrap();
                });
            },
        );

        let mut fitted = PreprocessorCore::new(
            Some(50),
            0.95,
            2,
            "__other__".to_string(),
            "__missing__".to_string(),
        )
        .unwrap();
        fitted.fit(&rows, &feature_names, &selected_idxs).unwrap();
        group.bench_with_input(BenchmarkId::new("transform", size), &rows, |b, rows| {
            b.iter(|| fitted.transform(black_box(rows)).unwrap());
        });
    }
    group.finish();
}

fn bench_preprocessor_numeric(c: &mut Criterion) {
    const N_COLS: usize = 3;
    let mut group = c.benchmark_group("preprocessor_numeric");
    for size in [10_000, 100_000] {
        let rows = make_numeric_matrix(size, N_COLS, 20_000, 113);
        let feature_names = numeric_feature_names(N_COLS);
        let numeric_idxs = numeric_feature_indices(N_COLS);

        group.throughput(Throughput::Elements((size * N_COLS) as u64));
        group.bench_with_input(
            BenchmarkId::new("fit_quantile", size),
            &(rows.clone(), feature_names.clone(), numeric_idxs.clone()),
            |b, (rows, feature_names, numeric_idxs)| {
                b.iter(|| {
                    let mut binner =
                        NumericBinnerCore::new(8, "quantile", "__missing__".to_string()).unwrap();
                    binner
                        .fit(
                            black_box(rows),
                            black_box(feature_names),
                            black_box(numeric_idxs),
                            None,
                            None,
                        )
                        .unwrap();
                });
            },
        );

        let mut fitted_quantile =
            NumericBinnerCore::new(8, "quantile", "__missing__".to_string()).unwrap();
        fitted_quantile
            .fit(&rows, &feature_names, &numeric_idxs, None, None)
            .unwrap();
        group.bench_with_input(
            BenchmarkId::new("transform_quantile", size),
            &rows,
            |b, rows| {
                b.iter(|| fitted_quantile.transform(black_box(rows)).unwrap());
            },
        );
    }

    // kmeans is slower; use a medium-size workload for smoke visibility.
    let size = 10_000usize;
    let rows = make_numeric_matrix(size, N_COLS, 20_000, 113);
    let feature_names = numeric_feature_names(N_COLS);
    let numeric_idxs = numeric_feature_indices(N_COLS);
    group.throughput(Throughput::Elements((size * N_COLS) as u64));
    group.bench_with_input(
        BenchmarkId::new("fit_kmeans", size),
        &(rows, feature_names, numeric_idxs),
        |b, (rows, feature_names, numeric_idxs)| {
            b.iter(|| {
                let mut binner =
                    NumericBinnerCore::new(8, "kmeans", "__missing__".to_string()).unwrap();
                binner
                    .fit(
                        black_box(rows),
                        black_box(feature_names),
                        black_box(numeric_idxs),
                        None,
                        None,
                    )
                    .unwrap();
            });
        },
    );

    let rows = make_numeric_matrix(size, N_COLS, 20_000, 113);
    let feature_names = numeric_feature_names(N_COLS);
    let numeric_idxs = numeric_feature_indices(N_COLS);
    let mut fitted_kmeans = NumericBinnerCore::new(8, "kmeans", "__missing__".to_string()).unwrap();
    fitted_kmeans
        .fit(&rows, &feature_names, &numeric_idxs, None, None)
        .unwrap();
    group.bench_with_input(
        BenchmarkId::new("transform_kmeans", size),
        &rows,
        |b, rows| {
            b.iter(|| fitted_kmeans.transform(black_box(rows)).unwrap());
        },
    );

    let rows = make_numeric_matrix(size, N_COLS, 20_000, 113);
    let feature_names = numeric_feature_names(N_COLS);
    let numeric_idxs = numeric_feature_indices(N_COLS);
    let targets = make_binary_targets(size, 0.3);
    group.bench_with_input(
        BenchmarkId::new("fit_tree", size),
        &(rows, feature_names, numeric_idxs, targets),
        |b, (rows, feature_names, numeric_idxs, targets)| {
            b.iter(|| {
                let mut binner =
                    NumericBinnerCore::new(8, "tree", "__missing__".to_string()).unwrap();
                binner
                    .fit(
                        black_box(rows),
                        black_box(feature_names),
                        black_box(numeric_idxs),
                        Some(black_box(targets)),
                        None,
                    )
                    .unwrap();
            });
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_binary_fit,
    bench_binary_transform,
    bench_binary_predict_proba,
    bench_binary_predict_ci,
    bench_fit_transform_combined,
    bench_multiclass_fit,
    bench_multiclass_predict_proba,
    bench_scaling_columns,
    bench_preprocessor_categorical,
    bench_preprocessor_numeric,
);
criterion_main!(benches);
