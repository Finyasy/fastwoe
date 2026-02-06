use std::collections::BTreeMap;

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
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryStats {
    pub category: String,
    pub event_count: usize,
    pub non_event_count: usize,
    pub woe: f64,
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

    let mut counts: BTreeMap<String, (usize, usize)> = BTreeMap::new();
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

    Ok(out)
}

#[derive(Debug, Clone)]
pub struct BinaryWoeModel {
    smoothing: f64,
    default_woe: f64,
    base_log_odds: Option<f64>,
    mapping: BTreeMap<String, f64>,
    stats: Vec<CategoryStats>,
}

impl BinaryWoeModel {
    pub fn new(smoothing: f64, default_woe: f64) -> Self {
        Self {
            smoothing,
            default_woe,
            base_log_odds: None,
            mapping: BTreeMap::new(),
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
            .collect::<BTreeMap<_, _>>();
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
    feature_mappings: BTreeMap<String, BTreeMap<String, f64>>,
    feature_stats: BTreeMap<String, Vec<CategoryStats>>,
}

impl BinaryTabularWoeModel {
    pub fn new(smoothing: f64, default_woe: f64) -> Self {
        Self {
            smoothing,
            default_woe,
            base_log_odds: None,
            feature_names: Vec::new(),
            feature_mappings: BTreeMap::new(),
            feature_stats: BTreeMap::new(),
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

        self.feature_mappings.clear();
        self.feature_stats.clear();

        for (col_idx, feature_name) in self.feature_names.iter().enumerate() {
            let categories = rows
                .iter()
                .map(|row| row[col_idx].clone())
                .collect::<Vec<String>>();
            let stats = compute_binary_woe(&categories, targets, self.smoothing)?;
            let mapping = stats
                .iter()
                .map(|s| (s.category.clone(), s.woe))
                .collect::<BTreeMap<_, _>>();
            self.feature_mappings.insert(feature_name.clone(), mapping);
            self.feature_stats.insert(feature_name.clone(), stats);
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
                self.feature_names
                    .iter()
                    .enumerate()
                    .map(|(col_idx, feature_name)| {
                        self.feature_mappings
                            .get(feature_name)
                            .and_then(|map| map.get(&row[col_idx]))
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
        self.feature_stats
            .get(feature_name)
            .map(|v| v.as_slice())
            .ok_or_else(|| WoeError::UnknownFeature(feature_name.to_string()))
    }
}

#[derive(Debug, Clone)]
pub struct MulticlassTabularWoeModel {
    class_labels: Vec<String>,
    models: BTreeMap<String, BinaryTabularWoeModel>,
}

impl MulticlassTabularWoeModel {
    pub fn new() -> Self {
        Self {
            class_labels: Vec::new(),
            models: BTreeMap::new(),
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
        self.models.clear();

        for class_label in unique_classes {
            let binary_targets = class_labels
                .iter()
                .map(|label| u8::from(label == &class_label))
                .collect::<Vec<u8>>();
            let mut model = BinaryTabularWoeModel::new(smoothing, default_woe);
            model.fit_matrix(rows, &binary_targets, feature_names)?;
            self.models.insert(class_label, model);
        }
        Ok(())
    }

    pub fn predict_proba_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<Vec<f64>>, WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }

        let mut class_scores = Vec::with_capacity(self.class_labels.len());
        for class_label in &self.class_labels {
            let model = self
                .models
                .get(class_label)
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
            .class_labels
            .iter()
            .position(|c| c == class_label)
            .ok_or_else(|| WoeError::UnknownClassLabel(class_label.to_string()))?;
        let all_probs = self.predict_proba_matrix(rows)?;
        Ok(all_probs.into_iter().map(|row| row[class_idx]).collect())
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
        let model = self
            .models
            .get(class_label)
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
}
