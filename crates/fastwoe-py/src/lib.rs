use fastwoe_core::{
    compute_binary_woe, BinaryTabularWoeModel, MulticlassTabularWoeModel, NumericBinnerCore,
    PredictionCi, PreprocessorCore, PreprocessorSummaryRow,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug)]
pub struct WoeRow {
    #[pyo3(get)]
    pub category: String,
    #[pyo3(get)]
    pub event_count: usize,
    #[pyo3(get)]
    pub non_event_count: usize,
    #[pyo3(get)]
    pub woe: f64,
    #[pyo3(get)]
    pub woe_se: f64,
}

#[pyclass]
#[derive(Debug)]
pub struct IvRow {
    #[pyo3(get)]
    pub feature: String,
    #[pyo3(get)]
    pub iv: f64,
    #[pyo3(get)]
    pub iv_se: f64,
    #[pyo3(get)]
    pub iv_ci_lower: f64,
    #[pyo3(get)]
    pub iv_ci_upper: f64,
    #[pyo3(get)]
    pub iv_significance: String,
}

#[pyclass]
#[derive(Debug)]
pub struct ReductionSummaryRow {
    #[pyo3(get)]
    pub feature: String,
    #[pyo3(get)]
    pub original_unique: usize,
    #[pyo3(get)]
    pub reduced_unique: usize,
    #[pyo3(get)]
    pub coverage: f64,
}

#[pyclass]
#[derive(Debug)]
pub struct RustPreprocessor {
    inner: PreprocessorCore,
}

#[pymethods]
impl RustPreprocessor {
    #[new]
    #[pyo3(signature = (max_categories=None, top_p=1.0, min_count=1, other_token="__other__".to_string(), missing_token="__missing__".to_string()))]
    fn new(
        max_categories: Option<usize>,
        top_p: f64,
        min_count: usize,
        other_token: String,
        missing_token: String,
    ) -> PyResult<Self> {
        let inner =
            PreprocessorCore::new(max_categories, top_p, min_count, other_token, missing_token)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn fit(
        &mut self,
        rows: Vec<Vec<String>>,
        feature_names: Vec<String>,
        cat_feature_indices: Vec<usize>,
    ) -> PyResult<()> {
        self.inner
            .fit(&rows, &feature_names, &cat_feature_indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transform(&self, rows: Vec<Vec<String>>) -> PyResult<Vec<Vec<String>>> {
        self.inner
            .transform(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn fit_transform(
        &mut self,
        rows: Vec<Vec<String>>,
        feature_names: Vec<String>,
        cat_feature_indices: Vec<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        self.inner
            .fit_transform(&rows, &feature_names, &cat_feature_indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_reduction_summary(&self) -> PyResult<Vec<ReductionSummaryRow>> {
        self.inner
            .summary_rows()
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(to_reduction_rows)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct RustNumericBinner {
    inner: NumericBinnerCore,
}

#[pymethods]
impl RustNumericBinner {
    #[new]
    #[pyo3(signature = (n_bins=5, binning_method="quantile".to_string(), missing_token="__missing__".to_string()))]
    fn new(n_bins: usize, binning_method: String, missing_token: String) -> PyResult<Self> {
        let inner = NumericBinnerCore::new(n_bins, &binning_method, missing_token)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (rows, feature_names, numeric_feature_indices, targets=None, monotonic_directions=None))]
    fn fit(
        &mut self,
        rows: Vec<Vec<Option<f64>>>,
        feature_names: Vec<String>,
        numeric_feature_indices: Vec<usize>,
        targets: Option<Vec<u8>>,
        monotonic_directions: Option<Vec<(usize, String)>>,
    ) -> PyResult<()> {
        self.inner
            .fit(
                &rows,
                &feature_names,
                &numeric_feature_indices,
                targets.as_deref(),
                monotonic_directions.as_deref(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transform(&self, rows: Vec<Vec<Option<f64>>>) -> PyResult<Vec<Vec<String>>> {
        self.inner
            .transform(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (rows, feature_names, numeric_feature_indices, targets=None, monotonic_directions=None))]
    fn fit_transform(
        &mut self,
        rows: Vec<Vec<Option<f64>>>,
        feature_names: Vec<String>,
        numeric_feature_indices: Vec<usize>,
        targets: Option<Vec<u8>>,
        monotonic_directions: Option<Vec<(usize, String)>>,
    ) -> PyResult<Vec<Vec<String>>> {
        self.inner
            .fit_transform(
                &rows,
                &feature_names,
                &numeric_feature_indices,
                targets.as_deref(),
                monotonic_directions.as_deref(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_reduction_summary(&self) -> PyResult<Vec<ReductionSummaryRow>> {
        self.inner
            .summary_rows()
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(to_reduction_rows)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct FastWoe {
    smoothing: f64,
    default_woe: f64,
    model: BinaryTabularWoeModel,
    multiclass_model: MulticlassTabularWoeModel,
}

#[pymethods]
impl FastWoe {
    #[new]
    #[pyo3(signature = (smoothing=0.5, default_woe=0.0))]
    fn new(smoothing: f64, default_woe: f64) -> Self {
        Self {
            smoothing,
            default_woe,
            model: BinaryTabularWoeModel::new(smoothing, default_woe),
            multiclass_model: MulticlassTabularWoeModel::new(),
        }
    }

    fn fit(&mut self, categories: Vec<String>, targets: Vec<u8>) -> PyResult<()> {
        let rows = to_single_feature_rows(&categories);
        let feature_names = vec![single_feature_name().to_string()];
        self.model
            .fit_matrix(&rows, &targets, Some(&feature_names))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transform(&self, categories: Vec<String>) -> PyResult<Vec<f64>> {
        let rows = to_single_feature_rows(&categories);
        self.model
            .transform_matrix(&rows)
            .map(|matrix| matrix.into_iter().map(|row| row[0]).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn fit_transform(&mut self, categories: Vec<String>, targets: Vec<u8>) -> PyResult<Vec<f64>> {
        let rows = to_single_feature_rows(&categories);
        let feature_names = vec![single_feature_name().to_string()];
        self.model
            .fit_transform_matrix(&rows, &targets, Some(&feature_names))
            .map(|matrix| matrix.into_iter().map(|row| row[0]).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict_proba(&self, categories: Vec<String>) -> PyResult<Vec<f64>> {
        let rows = to_single_feature_rows(&categories);
        self.model
            .predict_proba_matrix(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (categories, alpha=0.05))]
    fn predict_ci(&self, categories: Vec<String>, alpha: f64) -> PyResult<Vec<(f64, f64, f64)>> {
        let rows = to_single_feature_rows(&categories);
        self.model
            .predict_ci_matrix(&rows, alpha)
            .map(ci_to_tuples)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_mapping(&self) -> PyResult<Vec<WoeRow>> {
        self.model
            .feature_mapping(single_feature_name())
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(to_woe_rows)
    }

    #[pyo3(signature = (rows, targets, feature_names=None))]
    fn fit_matrix(
        &mut self,
        rows: Vec<Vec<String>>,
        targets: Vec<u8>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        self.model
            .fit_matrix(&rows, &targets, feature_names.as_deref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transform_matrix(&self, rows: Vec<Vec<String>>) -> PyResult<Vec<Vec<f64>>> {
        self.model
            .transform_matrix(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (rows, targets, feature_names=None))]
    fn fit_transform_matrix(
        &mut self,
        rows: Vec<Vec<String>>,
        targets: Vec<u8>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        self.model
            .fit_transform_matrix(&rows, &targets, feature_names.as_deref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict_proba_matrix(&self, rows: Vec<Vec<String>>) -> PyResult<Vec<f64>> {
        self.model
            .predict_proba_matrix(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (rows, alpha=0.05))]
    fn predict_ci_matrix(
        &self,
        rows: Vec<Vec<String>>,
        alpha: f64,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        self.model
            .predict_ci_matrix(&rows, alpha)
            .map(ci_to_tuples)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_feature_names(&self) -> PyResult<Vec<String>> {
        self.model
            .feature_names()
            .map(|names| names.to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_feature_mapping(&self, feature_name: String) -> PyResult<Vec<WoeRow>> {
        self.model
            .feature_mapping(&feature_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(to_woe_rows)
    }

    #[pyo3(signature = (feature_name=None, alpha=0.05))]
    fn get_iv_analysis(&self, feature_name: Option<String>, alpha: f64) -> PyResult<Vec<IvRow>> {
        if let Some(name) = feature_name {
            self.model
                .iv_analysis_feature_with_alpha(&name, alpha)
                .map(|row| vec![to_iv_row(row)])
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            self.model
                .iv_analysis_with_alpha(alpha)
                .map(|rows| rows.into_iter().map(to_iv_row).collect())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }

    fn fit_multiclass(
        &mut self,
        categories: Vec<String>,
        class_labels: Vec<String>,
    ) -> PyResult<()> {
        let rows = to_single_feature_rows(&categories);
        let feature_names = vec![single_feature_name().to_string()];
        self.multiclass_model
            .fit_matrix(
                &rows,
                &class_labels,
                Some(&feature_names),
                self.smoothing,
                self.default_woe,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict_proba_multiclass(&self, categories: Vec<String>) -> PyResult<Vec<Vec<f64>>> {
        let rows = to_single_feature_rows(&categories);
        self.multiclass_model
            .predict_proba_matrix(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (categories, alpha=0.05))]
    fn predict_ci_multiclass(
        &self,
        categories: Vec<String>,
        alpha: f64,
    ) -> PyResult<Vec<Vec<(f64, f64, f64)>>> {
        let rows = to_single_feature_rows(&categories);
        self.multiclass_model
            .predict_ci_matrix(&rows, alpha)
            .map(|out| out.into_iter().map(ci_to_tuples).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict_proba_class(
        &self,
        categories: Vec<String>,
        class_label: String,
    ) -> PyResult<Vec<f64>> {
        let rows = to_single_feature_rows(&categories);
        self.multiclass_model
            .predict_proba_class(&rows, &class_label)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (categories, class_label, alpha=0.05))]
    fn predict_ci_class(
        &self,
        categories: Vec<String>,
        class_label: String,
        alpha: f64,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        let rows = to_single_feature_rows(&categories);
        self.multiclass_model
            .predict_ci_class(&rows, &class_label, alpha)
            .map(ci_to_tuples)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_mapping_multiclass(&self, class_label: String) -> PyResult<Vec<WoeRow>> {
        self.multiclass_model
            .feature_mapping(&class_label, single_feature_name())
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(to_woe_rows)
    }

    #[pyo3(signature = (rows, class_labels, feature_names=None))]
    fn fit_matrix_multiclass(
        &mut self,
        rows: Vec<Vec<String>>,
        class_labels: Vec<String>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<()> {
        self.multiclass_model
            .fit_matrix(
                &rows,
                &class_labels,
                feature_names.as_deref(),
                self.smoothing,
                self.default_woe,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict_proba_matrix_multiclass(&self, rows: Vec<Vec<String>>) -> PyResult<Vec<Vec<f64>>> {
        self.multiclass_model
            .predict_proba_matrix(&rows)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transform_matrix_multiclass(&self, rows: Vec<Vec<String>>) -> PyResult<Vec<Vec<f64>>> {
        let class_labels = self
            .multiclass_model
            .class_labels()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let per_class = class_labels
            .iter()
            .map(|class_label| {
                self.multiclass_model
                    .transform_matrix_for_class(&rows, class_label)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        if per_class.is_empty() {
            return Ok(Vec::new());
        }

        let n_rows = per_class[0].len();
        let n_features = per_class[0].first().map_or(0, Vec::len);
        let mut flattened = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let mut out_row = Vec::with_capacity(class_labels.len() * n_features);
            for class_matrix in &per_class {
                out_row.extend(class_matrix[row_idx].iter().copied());
            }
            flattened.push(out_row);
        }
        Ok(flattened)
    }

    #[pyo3(signature = (rows, alpha=0.05))]
    fn predict_ci_matrix_multiclass(
        &self,
        rows: Vec<Vec<String>>,
        alpha: f64,
    ) -> PyResult<Vec<Vec<(f64, f64, f64)>>> {
        self.multiclass_model
            .predict_ci_matrix(&rows, alpha)
            .map(|out| out.into_iter().map(ci_to_tuples).collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn predict_proba_matrix_class(
        &self,
        rows: Vec<Vec<String>>,
        class_label: String,
    ) -> PyResult<Vec<f64>> {
        self.multiclass_model
            .predict_proba_class(&rows, &class_label)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (rows, class_label, alpha=0.05))]
    fn predict_ci_matrix_class(
        &self,
        rows: Vec<Vec<String>>,
        class_label: String,
        alpha: f64,
    ) -> PyResult<Vec<(f64, f64, f64)>> {
        self.multiclass_model
            .predict_ci_class(&rows, &class_label, alpha)
            .map(ci_to_tuples)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_class_labels(&self) -> PyResult<Vec<String>> {
        self.multiclass_model
            .class_labels()
            .map(|labels| labels.to_vec())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn get_feature_names_multiclass(&self) -> PyResult<Vec<String>> {
        let class_labels = self
            .multiclass_model
            .class_labels()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let feature_names = self
            .multiclass_model
            .feature_names()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let mut out = Vec::with_capacity(class_labels.len() * feature_names.len());
        for class_label in class_labels {
            for feature_name in feature_names {
                out.push(format!("{feature_name}_class_{class_label}"));
            }
        }
        Ok(out)
    }

    fn get_feature_mapping_multiclass(
        &self,
        class_label: String,
        feature_name: String,
    ) -> PyResult<Vec<WoeRow>> {
        self.multiclass_model
            .feature_mapping(&class_label, &feature_name)
            .map_err(|e| PyValueError::new_err(e.to_string()))
            .map(to_woe_rows)
    }

    #[pyo3(signature = (class_label, feature_name=None, alpha=0.05))]
    fn get_iv_analysis_multiclass(
        &self,
        class_label: String,
        feature_name: Option<String>,
        alpha: f64,
    ) -> PyResult<Vec<IvRow>> {
        if let Some(name) = feature_name {
            self.multiclass_model
                .iv_analysis_class_feature_with_alpha(&class_label, &name, alpha)
                .map(|row| vec![to_iv_row(row)])
                .map_err(|e| PyValueError::new_err(e.to_string()))
        } else {
            self.multiclass_model
                .iv_analysis_class_with_alpha(&class_label, alpha)
                .map(|rows| rows.into_iter().map(to_iv_row).collect())
                .map_err(|e| PyValueError::new_err(e.to_string()))
        }
    }
}

#[pyfunction]
#[pyo3(signature = (categories, targets, smoothing=0.5))]
fn compute_binary_woe_py(
    categories: Vec<String>,
    targets: Vec<u8>,
    smoothing: f64,
) -> PyResult<Vec<WoeRow>> {
    let stats = compute_binary_woe(&categories, &targets, smoothing)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(stats
        .into_iter()
        .map(|s| WoeRow {
            category: s.category,
            event_count: s.event_count,
            non_event_count: s.non_event_count,
            woe: s.woe,
            woe_se: s.woe_se,
        })
        .collect())
}

#[pymodule]
fn fastwoe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastWoe>()?;
    m.add_class::<RustPreprocessor>()?;
    m.add_class::<RustNumericBinner>()?;
    m.add_class::<WoeRow>()?;
    m.add_class::<IvRow>()?;
    m.add_class::<ReductionSummaryRow>()?;
    m.add_function(wrap_pyfunction!(compute_binary_woe_py, m)?)?;
    Ok(())
}

fn single_feature_name() -> &'static str {
    "__single_feature__"
}

fn to_single_feature_rows(categories: &[String]) -> Vec<Vec<String>> {
    categories
        .iter()
        .map(|v| vec![v.clone()])
        .collect::<Vec<Vec<String>>>()
}

fn to_woe_rows(stats: &[fastwoe_core::CategoryStats]) -> Vec<WoeRow> {
    stats
        .iter()
        .map(|s| WoeRow {
            category: s.category.clone(),
            event_count: s.event_count,
            non_event_count: s.non_event_count,
            woe: s.woe,
            woe_se: s.woe_se,
        })
        .collect()
}

fn ci_to_tuples(values: Vec<PredictionCi>) -> Vec<(f64, f64, f64)> {
    values
        .into_iter()
        .map(|r| (r.prediction, r.lower_ci, r.upper_ci))
        .collect()
}

fn to_iv_row(value: fastwoe_core::IvFeatureStats) -> IvRow {
    IvRow {
        feature: value.feature,
        iv: value.iv,
        iv_se: value.iv_se,
        iv_ci_lower: value.iv_ci_lower,
        iv_ci_upper: value.iv_ci_upper,
        iv_significance: value.iv_significance,
    }
}

fn to_reduction_rows(values: &[PreprocessorSummaryRow]) -> Vec<ReductionSummaryRow> {
    values
        .iter()
        .map(|row| ReductionSummaryRow {
            feature: row.feature.clone(),
            original_unique: row.original_unique,
            reduced_unique: row.reduced_unique,
            coverage: row.coverage,
        })
        .collect()
}
