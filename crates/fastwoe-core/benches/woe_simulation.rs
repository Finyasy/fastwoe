//! Simulation benchmarks for FastWOE: fit, transform, predict at scale.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use fastwoe_core::{BinaryTabularWoeModel, MulticlassTabularWoeModel};

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
);
criterion_main!(benches);
