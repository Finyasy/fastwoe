use fastwoe_core::{
    compute_binary_woe, BinaryTabularWoeModel, MulticlassTabularWoeModel, PredictionCi,
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
        })
        .collect())
}

#[pymodule]
fn fastwoe_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<FastWoe>()?;
    m.add_class::<WoeRow>()?;
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
        })
        .collect()
}

fn ci_to_tuples(values: Vec<PredictionCi>) -> Vec<(f64, f64, f64)> {
    values
        .into_iter()
        .map(|r| (r.prediction, r.lower_ci, r.upper_ci))
        .collect()
}
