use std::collections::HashMap;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum WoeError {
    #[error("input categories and targets must have the same length")]
    LengthMismatch,
    #[error("target values must be binary (0 or 1)")]
    NonBinaryTarget,
    #[error("empty input is not allowed")]
    EmptyInput,
    #[error("model is not fitted")]
    NotFitted,
    #[error("matrix input must be non-empty and rectangular")]
    InvalidMatrixShape,
    #[error("feature_names length must match number of columns")]
    FeatureNameCountMismatch,
    #[error("unknown feature: {0}")]
    UnknownFeature(String),
    #[error("class labels must be non-empty")]
    EmptyClassLabels,
    #[error("model is not fitted for multiclass")]
    MulticlassNotFitted,
    #[error("unknown class label: {0}")]
    UnknownClassLabel(String),
    #[error("alpha must be between 0 and 1 (exclusive)")]
    InvalidAlpha,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryStats {
    pub category: String,
    pub event_count: usize,
    pub non_event_count: usize,
    pub woe: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictionCi {
    pub prediction: f64,
    pub lower_ci: f64,
    pub upper_ci: f64,
}

/// Compute WOE per category for a single categorical feature.
pub fn compute_binary_woe(
    categories: &[String],
    targets: &[u8],
    smoothing: f64,
) -> Result<Vec<CategoryStats>, WoeError> {
    if categories.is_empty() || targets.is_empty() {
        return Err(WoeError::EmptyInput);
    }
    if categories.len() != targets.len() {
        return Err(WoeError::LengthMismatch);
    }
    if targets.iter().any(|&v| v > 1) {
        return Err(WoeError::NonBinaryTarget);
    }

    let mut counts: HashMap<String, (usize, usize)> = HashMap::new();
    let mut total_events = 0usize;
    let mut total_non_events = 0usize;

    for (cat, y) in categories.iter().zip(targets.iter()) {
        let entry = counts.entry(cat.clone()).or_insert((0, 0));
        if *y == 1 {
            entry.0 += 1;
            total_events += 1;
        } else {
            entry.1 += 1;
            total_non_events += 1;
        }
    }

    let k = counts.len() as f64;
    let denom_events = total_events as f64 + smoothing * k;
    let denom_non_events = total_non_events as f64 + smoothing * k;

    let mut out = Vec::with_capacity(counts.len());
    for (category, (event_count, non_event_count)) in counts {
        let p_event = (event_count as f64 + smoothing) / denom_events;
        let p_non_event = (non_event_count as f64 + smoothing) / denom_non_events;
        let woe = (p_event / p_non_event).ln();

        out.push(CategoryStats {
            category,
            event_count,
            non_event_count,
            woe,
        });
    }
    out.sort_by(|a, b| a.category.cmp(&b.category));

    Ok(out)
}

#[derive(Debug, Clone)]
pub struct BinaryWoeModel {
    smoothing: f64,
    default_woe: f64,
    base_log_odds: Option<f64>,
    mapping: HashMap<String, f64>,
    stats: Vec<CategoryStats>,
}

impl BinaryWoeModel {
    pub fn new(smoothing: f64, default_woe: f64) -> Self {
        Self {
            smoothing,
            default_woe,
            base_log_odds: None,
            mapping: HashMap::new(),
            stats: Vec::new(),
        }
    }

    pub fn fit(&mut self, categories: &[String], targets: &[u8]) -> Result<(), WoeError> {
        let stats = compute_binary_woe(categories, targets, self.smoothing)?;
        let n = targets.len() as f64;
        let event_rate = targets.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let clipped_event_rate = event_rate.clamp(1e-12, 1.0 - 1e-12);
        self.base_log_odds = Some((clipped_event_rate / (1.0 - clipped_event_rate)).ln());
        self.mapping = stats
            .iter()
            .map(|row| (row.category.clone(), row.woe))
            .collect::<HashMap<_, _>>();
        self.stats = stats;
        Ok(())
    }

    pub fn transform(&self, categories: &[String]) -> Result<Vec<f64>, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }

        Ok(categories
            .iter()
            .map(|cat| self.mapping.get(cat).copied().unwrap_or(self.default_woe))
            .collect())
    }

    pub fn fit_transform(
        &mut self,
        categories: &[String],
        targets: &[u8],
    ) -> Result<Vec<f64>, WoeError> {
        self.fit(categories, targets)?;
        self.transform(categories)
    }

    pub fn predict_proba(&self, categories: &[String]) -> Result<Vec<f64>, WoeError> {
        let base = self.base_log_odds.ok_or(WoeError::NotFitted)?;
        let woe_values = self.transform(categories)?;
        Ok(woe_values
            .into_iter()
            .map(|woe| sigmoid(base + woe))
            .collect::<Vec<_>>())
    }

    pub fn mapping(&self) -> Result<&[CategoryStats], WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        Ok(&self.stats)
    }
}

#[derive(Debug, Clone)]
pub struct BinaryTabularWoeModel {
    smoothing: f64,
    default_woe: f64,
    base_log_odds: Option<f64>,
    feature_names: Vec<String>,
    feature_index: HashMap<String, usize>,
    feature_mappings: Vec<HashMap<String, f64>>,
    feature_stats: Vec<Vec<CategoryStats>>,
}

impl BinaryTabularWoeModel {
    pub fn new(smoothing: f64, default_woe: f64) -> Self {
        Self {
            smoothing,
            default_woe,
            base_log_odds: None,
            feature_names: Vec::new(),
            feature_index: HashMap::new(),
            feature_mappings: Vec::new(),
            feature_stats: Vec::new(),
        }
    }

    pub fn fit_matrix(
        &mut self,
        rows: &[Vec<String>],
        targets: &[u8],
        feature_names: Option<&[String]>,
    ) -> Result<(), WoeError> {
        validate_binary_targets(targets)?;
        let ncols = validate_matrix(rows, Some(targets.len()))?;
        self.feature_names = resolve_feature_names(feature_names, ncols)?;

        let n = targets.len() as f64;
        let event_rate = targets.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let clipped_event_rate = event_rate.clamp(1e-12, 1.0 - 1e-12);
        self.base_log_odds = Some((clipped_event_rate / (1.0 - clipped_event_rate)).ln());

        self.feature_index = self
            .feature_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect::<HashMap<_, _>>();
        self.feature_mappings.clear();
        self.feature_stats.clear();
        self.feature_mappings.reserve(self.feature_names.len());
        self.feature_stats.reserve(self.feature_names.len());

        for col_idx in 0..self.feature_names.len() {
            let categories = rows
                .iter()
                .map(|row| row[col_idx].clone())
                .collect::<Vec<String>>();
            let stats = compute_binary_woe(&categories, targets, self.smoothing)?;
            let mapping = stats
                .iter()
                .map(|s| (s.category.clone(), s.woe))
                .collect::<HashMap<_, _>>();
            self.feature_mappings.push(mapping);
            self.feature_stats.push(stats);
        }

        Ok(())
    }

    pub fn transform_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        validate_matrix(rows, None)?;

        Ok(rows
            .iter()
            .map(|row| {
                self.feature_mappings
                    .iter()
                    .enumerate()
                    .map(|(col_idx, mapping)| {
                        mapping
                            .get(&row[col_idx])
                            .copied()
                            .unwrap_or(self.default_woe)
                    })
                    .collect::<Vec<f64>>()
            })
            .collect())
    }

    pub fn fit_transform_matrix(
        &mut self,
        rows: &[Vec<String>],
        targets: &[u8],
        feature_names: Option<&[String]>,
    ) -> Result<Vec<Vec<f64>>, WoeError> {
        self.fit_matrix(rows, targets, feature_names)?;
        self.transform_matrix(rows)
    }

    pub fn predict_proba_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<f64>, WoeError> {
        let base = self.base_log_odds.ok_or(WoeError::NotFitted)?;
        let scores = self.decision_scores_matrix(rows)?;
        Ok(scores
            .into_iter()
            .map(|score| sigmoid(base + score))
            .collect())
    }

    pub fn predict_ci_matrix(
        &self,
        rows: &[Vec<String>],
        alpha: f64,
    ) -> Result<Vec<PredictionCi>, WoeError> {
        let probs = self.predict_proba_matrix(rows)?;
        compute_prediction_ci(&probs, alpha)
    }

    pub fn decision_scores_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<f64>, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        let transformed = self.transform_matrix(rows)?;
        Ok(transformed
            .into_iter()
            .map(|row| row.iter().sum::<f64>())
            .collect())
    }

    pub fn feature_names(&self) -> Result<&[String], WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        Ok(&self.feature_names)
    }

    pub fn feature_mapping(&self, feature_name: &str) -> Result<&[CategoryStats], WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        let idx = self
            .feature_index
            .get(feature_name)
            .copied()
            .ok_or_else(|| WoeError::UnknownFeature(feature_name.to_string()))?;
        self.feature_stats
            .get(idx)
            .map(|v| v.as_slice())
            .ok_or_else(|| WoeError::UnknownFeature(feature_name.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct MulticlassTabularWoeModel {
    class_labels: Vec<String>,
    class_index: HashMap<String, usize>,
    models: Vec<BinaryTabularWoeModel>,
}

impl MulticlassTabularWoeModel {
    pub fn new() -> Self {
        Self {
            class_labels: Vec::new(),
            class_index: HashMap::new(),
            models: Vec::new(),
        }
    }

    pub fn fit_matrix(
        &mut self,
        rows: &[Vec<String>],
        class_labels: &[String],
        feature_names: Option<&[String]>,
        smoothing: f64,
        default_woe: f64,
    ) -> Result<(), WoeError> {
        if class_labels.is_empty() {
            return Err(WoeError::EmptyClassLabels);
        }
        validate_matrix(rows, Some(class_labels.len()))?;
        let unique_classes = unique_strings(class_labels);

        self.class_labels = unique_classes.clone();
        self.class_index = self
            .class_labels
            .iter()
            .enumerate()
            .map(|(idx, c)| (c.clone(), idx))
            .collect::<HashMap<_, _>>();
        self.models.clear();
        self.models.reserve(self.class_labels.len());

        for class_label in unique_classes {
            let binary_targets = class_labels
                .iter()
                .map(|label| u8::from(label == &class_label))
                .collect::<Vec<u8>>();
            let mut model = BinaryTabularWoeModel::new(smoothing, default_woe);
            model.fit_matrix(rows, &binary_targets, feature_names)?;
            self.models.push(model);
        }
        Ok(())
    }

    pub fn predict_proba_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }

        let mut class_scores = Vec::with_capacity(self.class_labels.len());
        for (class_idx, class_label) in self.class_labels.iter().enumerate() {
            let model = self
                .models
                .get(class_idx)
                .ok_or_else(|| WoeError::UnknownClassLabel(class_label.clone()))?;
            let base = model.base_log_odds.ok_or(WoeError::NotFitted)?;
            let scores = model.decision_scores_matrix(rows)?;
            let logits = scores.into_iter().map(|s| base + s).collect::<Vec<f64>>();
            class_scores.push(logits);
        }

        let n_rows = class_scores.first().map_or(0, Vec::len);
        let mut probs = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let logits = class_scores
                .iter()
                .map(|v| v[row_idx])
                .collect::<Vec<f64>>();
            probs.push(softmax(&logits));
        }
        Ok(probs)
    }

    pub fn predict_proba_class(
        &self,
        rows: &[Vec<String>],
        class_label: &str,
    ) -> Result<Vec<f64>, WoeError> {
        let class_idx = self
            .class_index
            .get(class_label)
            .copied()
            .ok_or_else(|| WoeError::UnknownClassLabel(class_label.to_string()))?;
        let all_probs = self.predict_proba_matrix(rows)?;
        Ok(all_probs.into_iter().map(|row| row[class_idx]).collect())
    }

    pub fn transform_matrix_for_class(
        &self,
        rows: &[Vec<String>],
        class_label: &str,
    ) -> Result<Vec<Vec<f64>>, WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        let class_idx = self
            .class_index
            .get(class_label)
            .copied()
            .ok_or_else(|| WoeError::UnknownClassLabel(class_label.to_string()))?;
        let model = self
            .models
            .get(class_idx)
            .ok_or_else(|| WoeError::UnknownClassLabel(class_label.to_string()))?;
        model.transform_matrix(rows)
    }

    pub fn feature_names(&self) -> Result<&[String], WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        let model = self.models.first().ok_or(WoeError::MulticlassNotFitted)?;
        model.feature_names()
    }

    pub fn predict_ci_matrix(
        &self,
        rows: &[Vec<String>],
        alpha: f64,
    ) -> Result<Vec<Vec<PredictionCi>>, WoeError> {
        let probs = self.predict_proba_matrix(rows)?;
        probs
            .iter()
            .map(|row| compute_prediction_ci(row, alpha))
            .collect::<Result<Vec<_>, _>>()
    }

    pub fn predict_ci_class(
        &self,
        rows: &[Vec<String>],
        class_label: &str,
        alpha: f64,
    ) -> Result<Vec<PredictionCi>, WoeError> {
        let probs = self.predict_proba_class(rows, class_label)?;
        compute_prediction_ci(&probs, alpha)
    }

    pub fn class_labels(&self) -> Result<&[String], WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        Ok(&self.class_labels)
    }

    pub fn feature_mapping(
        &self,
        class_label: &str,
        feature_name: &str,
    ) -> Result<&[CategoryStats], WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        let class_idx = self
            .class_index
            .get(class_label)
            .copied()
            .ok_or_else(|| WoeError::UnknownClassLabel(class_label.to_string()))?;
        let model = self
            .models
            .get(class_idx)
            .ok_or_else(|| WoeError::UnknownClassLabel(class_label.to_string()))?;
        model.feature_mapping(feature_name)
    }
}

impl Default for MulticlassTabularWoeModel {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_binary_targets(targets: &[u8]) -> Result<(), WoeError> {
    if targets.is_empty() {
        return Err(WoeError::EmptyInput);
    }
    if targets.iter().any(|&v| v > 1) {
        return Err(WoeError::NonBinaryTarget);
    }
    Ok(())
}

fn validate_matrix(rows: &[Vec<String>], expected_rows: Option<usize>) -> Result<usize, WoeError> {
    if rows.is_empty() {
        return Err(WoeError::EmptyInput);
    }
    if let Some(n) = expected_rows {
        if rows.len() != n {
            return Err(WoeError::LengthMismatch);
        }
    }
    let ncols = rows[0].len();
    if ncols == 0 || rows.iter().any(|row| row.len() != ncols) {
        return Err(WoeError::InvalidMatrixShape);
    }
    Ok(ncols)
}

fn resolve_feature_names(
    feature_names: Option<&[String]>,
    ncols: usize,
) -> Result<Vec<String>, WoeError> {
    match feature_names {
        Some(names) if names.len() != ncols => Err(WoeError::FeatureNameCountMismatch),
        Some(names) => Ok(names.to_vec()),
        None => Ok((0..ncols).map(|i| format!("feature_{i}")).collect()),
    }
}

fn unique_strings(values: &[String]) -> Vec<String> {
    let mut out = values.to_vec();
    out.sort();
    out.dedup();
    out
}

fn compute_prediction_ci(probs: &[f64], alpha: f64) -> Result<Vec<PredictionCi>, WoeError> {
    if !(0.0..1.0).contains(&alpha) {
        return Err(WoeError::InvalidAlpha);
    }
    let z = normal_ppf(1.0 - alpha / 2.0);
    Ok(probs
        .iter()
        .map(|&p| {
            let var = (p * (1.0 - p)).max(0.0);
            let se = var.sqrt();
            let lower = (p - z * se).clamp(0.0, 1.0);
            let upper = (p + z * se).clamp(0.0, 1.0);
            PredictionCi {
                prediction: p,
                lower_ci: lower,
                upper_ci: upper,
            }
        })
        .collect())
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_logit = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps = logits
        .iter()
        .map(|v| (v - max_logit).exp())
        .collect::<Vec<f64>>();
    let sum = exps.iter().sum::<f64>();
    exps.into_iter().map(|e| e / sum).collect()
}

fn normal_ppf(p: f64) -> f64 {
    const A: [f64; 6] = [
        -39.696_830_286_653_76,
        220.946_098_424_520_5,
        -275.928_510_446_968_7,
        138.357_751_867_269,
        -30.664_798_066_147_16,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -54.476_098_798_224_06,
        161.585_836_858_040_9,
        -155.698_979_859_886_6,
        66.801_311_887_719_72,
        -13.280_681_552_885_72,
    ];
    const C: [f64; 6] = [
        -0.007_784_894_002_430_293,
        -0.322_396_458_041_136_5,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        0.007_784_695_709_041_462,
        0.322_467_129_070_039_8,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    let plow = 0.024_25;
    let phigh = 1.0 - plow;
    if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= phigh {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_woe_for_simple_data() {
        let cats = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let y = vec![1, 0, 1, 0];

        let stats = compute_binary_woe(&cats, &y, 0.5).expect("woe should compute");
        assert_eq!(stats.len(), 2);
        assert_eq!(stats[0].category, "A");
        assert_eq!(stats[1].category, "B");
    }

    #[test]
    fn model_fit_transform_predict_proba_works() {
        let cats = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "B".to_string(),
        ];
        let y = vec![1, 0, 1, 0];
        let mut model = BinaryWoeModel::new(0.5, 0.0);

        let transformed = model
            .fit_transform(&cats, &y)
            .expect("fit_transform should work");
        assert_eq!(transformed.len(), 4);

        let probs = model.predict_proba(&cats).expect("predict should work");
        assert_eq!(probs.len(), 4);
        assert!(probs.into_iter().all(|p| p > 0.0 && p < 1.0));
    }

    #[test]
    fn tabular_model_fit_transform_predict_works() {
        let rows = vec![
            vec!["A".to_string(), "X".to_string()],
            vec!["A".to_string(), "Y".to_string()],
            vec!["B".to_string(), "X".to_string()],
            vec!["B".to_string(), "Z".to_string()],
        ];
        let y = vec![1, 0, 0, 1];
        let feature_names = vec!["cat".to_string(), "bucket".to_string()];
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);

        let transformed = model
            .fit_transform_matrix(&rows, &y, Some(&feature_names))
            .expect("fit_transform_matrix should work");
        assert_eq!(transformed.len(), 4);
        assert_eq!(transformed[0].len(), 2);

        let probs = model
            .predict_proba_matrix(&rows)
            .expect("predict_proba_matrix should work");
        assert_eq!(probs.len(), 4);
        assert!(probs.into_iter().all(|p| p > 0.0 && p < 1.0));

        let mapping = model
            .feature_mapping("cat")
            .expect("feature mapping should exist");
        assert!(!mapping.is_empty());
    }

    #[test]
    fn multiclass_predict_proba_softmax_works() {
        let rows = vec![
            vec!["A".to_string(), "X".to_string()],
            vec!["A".to_string(), "Y".to_string()],
            vec!["B".to_string(), "X".to_string()],
            vec!["B".to_string(), "Z".to_string()],
            vec!["C".to_string(), "Y".to_string()],
        ];
        let y = vec![
            "class_0".to_string(),
            "class_1".to_string(),
            "class_2".to_string(),
            "class_0".to_string(),
            "class_1".to_string(),
        ];
        let feature_names = vec!["cat".to_string(), "bucket".to_string()];
        let mut model = MulticlassTabularWoeModel::new();
        model
            .fit_matrix(&rows, &y, Some(&feature_names), 0.5, 0.0)
            .expect("multiclass fit should work");

        let probs = model
            .predict_proba_matrix(&rows)
            .expect("multiclass predict should work");
        assert_eq!(probs.len(), rows.len());
        assert!(probs
            .iter()
            .all(|row| (row.iter().sum::<f64>() - 1.0).abs() < 1e-9));

        let class_0 = model
            .predict_proba_class(&rows, "class_0")
            .expect("class prediction should work");
        assert_eq!(class_0.len(), rows.len());
    }

    #[test]
    fn binary_predict_ci_works() {
        let rows = vec![
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["A".to_string()],
        ];
        let y = vec![1, 0, 1];
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
        model
            .fit_matrix(&rows, &y, None)
            .expect("fit should work for ci test");
        let ci = model
            .predict_ci_matrix(&rows, 0.05)
            .expect("ci should compute");
        assert_eq!(ci.len(), rows.len());
        assert!(ci
            .iter()
            .all(|r| r.lower_ci <= r.prediction && r.prediction <= r.upper_ci));
    }

    #[test]
    fn multiclass_predict_ci_class_works() {
        let rows = vec![
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["C".to_string()],
            vec!["A".to_string()],
        ];
        let y = vec![
            "class_0".to_string(),
            "class_1".to_string(),
            "class_2".to_string(),
            "class_0".to_string(),
        ];
        let mut model = MulticlassTabularWoeModel::new();
        model
            .fit_matrix(&rows, &y, None, 0.5, 0.0)
            .expect("fit should work");
        let ci = model
            .predict_ci_class(&rows, "class_0", 0.05)
            .expect("ci class should work");
        assert_eq!(ci.len(), rows.len());
    }

    #[test]
    fn multiclass_transform_for_class_works() {
        let rows = vec![
            vec!["A".to_string(), "X".to_string()],
            vec!["B".to_string(), "Y".to_string()],
            vec!["C".to_string(), "Z".to_string()],
        ];
        let y = vec![
            "class_0".to_string(),
            "class_1".to_string(),
            "class_0".to_string(),
        ];
        let feature_names = vec!["cat".to_string(), "bucket".to_string()];
        let mut model = MulticlassTabularWoeModel::new();
        model
            .fit_matrix(&rows, &y, Some(&feature_names), 0.5, 0.0)
            .expect("fit should work");
        let transformed = model
            .transform_matrix_for_class(&rows, "class_0")
            .expect("class transform should work");
        assert_eq!(transformed.len(), rows.len());
        assert_eq!(transformed[0].len(), feature_names.len());
    }
}
