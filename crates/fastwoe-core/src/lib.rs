use std::collections::{HashMap, HashSet};

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
    #[error("smoothing must be strictly positive")]
    InvalidSmoothing,
    #[error("max_categories must be positive when provided")]
    InvalidMaxCategories,
    #[error("top_p must be in (0, 1]")]
    InvalidTopP,
    #[error("min_count must be positive")]
    InvalidMinCount,
    #[error("n_bins must be greater than 1")]
    InvalidNBins,
    #[error("unsupported binning_method: {0}")]
    InvalidBinningMethod(String),
    #[error("target is required when binning_method='tree' and numerical_features are provided")]
    MissingTargetForBinning,
    #[error("invalid monotonic direction: {0}")]
    InvalidMonotonicDirection(String),
    #[error("feature index out of range: {0}")]
    FeatureIndexOutOfRange(usize),
}

#[derive(Debug, Clone, PartialEq)]
pub struct CategoryStats {
    pub category: String,
    pub event_count: usize,
    pub non_event_count: usize,
    pub woe: f64,
    pub woe_se: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IvFeatureStats {
    pub feature: String,
    pub iv: f64,
    pub iv_se: f64,
    pub iv_ci_lower: f64,
    pub iv_ci_upper: f64,
    pub iv_significance: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PreprocessorSummaryRow {
    pub feature: String,
    pub original_unique: usize,
    pub reduced_unique: usize,
    pub coverage: f64,
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
    if smoothing <= 0.0 {
        return Err(WoeError::InvalidSmoothing);
    }
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
        let woe_se = (1.0 / (event_count as f64 + smoothing)
            + 1.0 / (non_event_count as f64 + smoothing))
            .sqrt();

        out.push(CategoryStats {
            category,
            event_count,
            non_event_count,
            woe,
            woe_se,
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
pub struct PreprocessorCore {
    max_categories: Option<usize>,
    top_p: f64,
    min_count: usize,
    other_token: String,
    missing_token: String,
    fitted: bool,
    feature_names: Vec<String>,
    selected_idxs: Vec<usize>,
    allowed: HashMap<usize, HashSet<String>>,
    summary_rows: Vec<PreprocessorSummaryRow>,
}

impl PreprocessorCore {
    pub fn new(
        max_categories: Option<usize>,
        top_p: f64,
        min_count: usize,
        other_token: String,
        missing_token: String,
    ) -> Result<Self, WoeError> {
        if let Some(v) = max_categories {
            if v == 0 {
                return Err(WoeError::InvalidMaxCategories);
            }
        }
        if !(0.0 < top_p && top_p <= 1.0) {
            return Err(WoeError::InvalidTopP);
        }
        if min_count == 0 {
            return Err(WoeError::InvalidMinCount);
        }

        Ok(Self {
            max_categories,
            top_p,
            min_count,
            other_token,
            missing_token,
            fitted: false,
            feature_names: Vec::new(),
            selected_idxs: Vec::new(),
            allowed: HashMap::new(),
            summary_rows: Vec::new(),
        })
    }

    pub fn fit(
        &mut self,
        rows: &[Vec<String>],
        feature_names: &[String],
        selected_idxs: &[usize],
    ) -> Result<(), WoeError> {
        let ncols = validate_matrix(rows, None)?;
        if feature_names.len() != ncols {
            return Err(WoeError::FeatureNameCountMismatch);
        }

        let selected = stable_unique_usize(selected_idxs);
        for &idx in &selected {
            if idx >= ncols {
                return Err(WoeError::FeatureIndexOutOfRange(idx));
            }
        }

        self.feature_names = feature_names.to_vec();
        self.selected_idxs = selected;
        self.allowed.clear();
        self.summary_rows.clear();
        self.summary_rows.reserve(self.selected_idxs.len());

        for &idx in &self.selected_idxs {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for row in rows {
                let category = row[idx].clone();
                *counts.entry(category).or_insert(0) += 1;
            }
            let total = counts.values().sum::<usize>();
            let sorted_counts = sorted_counts_by_freq_then_key(&counts);
            let keep = self.select_categories(&sorted_counts, &counts, total);

            let kept_count = keep
                .iter()
                .map(|cat| counts.get(cat).copied().unwrap_or(0))
                .sum::<usize>();
            let coverage = if total > 0 {
                kept_count as f64 / total as f64
            } else {
                0.0
            };

            let mut keep_set = HashSet::with_capacity(keep.len());
            for category in &keep {
                keep_set.insert(category.clone());
            }
            self.allowed.insert(idx, keep_set);

            self.summary_rows.push(PreprocessorSummaryRow {
                feature: self.feature_names[idx].clone(),
                original_unique: counts.len(),
                reduced_unique: keep.len() + 1,
                coverage,
            });
        }

        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, rows: &[Vec<String>]) -> Result<Vec<Vec<String>>, WoeError> {
        if !self.fitted {
            return Err(WoeError::NotFitted);
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let ncols = rows[0].len();
        if ncols == 0 || rows.iter().any(|row| row.len() != ncols) {
            return Err(WoeError::InvalidMatrixShape);
        }
        if ncols != self.feature_names.len() {
            return Err(WoeError::FeatureNameCountMismatch);
        }

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let mut out_row = row.clone();
            for &idx in &self.selected_idxs {
                let category = &row[idx];
                let keep = self
                    .allowed
                    .get(&idx)
                    .ok_or(WoeError::NotFitted)?
                    .contains(category);
                if !keep {
                    out_row[idx] = self.other_token.clone();
                }
            }
            out.push(out_row);
        }
        Ok(out)
    }

    pub fn fit_transform(
        &mut self,
        rows: &[Vec<String>],
        feature_names: &[String],
        selected_idxs: &[usize],
    ) -> Result<Vec<Vec<String>>, WoeError> {
        self.fit(rows, feature_names, selected_idxs)?;
        self.transform(rows)
    }

    pub fn summary_rows(&self) -> Result<&[PreprocessorSummaryRow], WoeError> {
        if !self.fitted {
            return Err(WoeError::NotFitted);
        }
        Ok(&self.summary_rows)
    }

    fn select_categories(
        &self,
        sorted_counts: &[(String, usize)],
        counts_map: &HashMap<String, usize>,
        total: usize,
    ) -> Vec<String> {
        let mut keep: Vec<String> = Vec::new();
        let mut cumulative = 0usize;

        for (category, count) in sorted_counts {
            if *count < self.min_count {
                continue;
            }
            keep.push(category.clone());
            cumulative += *count;
            if total > 0 && (cumulative as f64 / total as f64) >= self.top_p {
                break;
            }
        }

        if keep.is_empty() && !sorted_counts.is_empty() {
            keep.push(sorted_counts[0].0.clone());
        }

        if let Some(max_categories) = self.max_categories {
            if keep.len() > max_categories {
                keep.truncate(max_categories);
            }
        }

        if counts_map.contains_key(&self.missing_token) && !keep.contains(&self.missing_token) {
            if let Some(max_categories) = self.max_categories {
                if keep.len() >= max_categories && !keep.is_empty() {
                    let last = keep.len() - 1;
                    keep[last] = self.missing_token.clone();
                } else {
                    keep.push(self.missing_token.clone());
                }
            } else {
                keep.push(self.missing_token.clone());
            }
        }

        stable_unique_strings(&keep)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericBinningMethod {
    Quantile,
    Uniform,
    Kmeans,
    Tree,
}

impl NumericBinningMethod {
    fn from_str(value: &str) -> Result<Self, WoeError> {
        match value {
            "quantile" => Ok(Self::Quantile),
            "uniform" => Ok(Self::Uniform),
            "kmeans" => Ok(Self::Kmeans),
            "tree" => Ok(Self::Tree),
            other => Err(WoeError::InvalidBinningMethod(other.to_string())),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumericBinnerCore {
    n_bins: usize,
    method: NumericBinningMethod,
    missing_token: String,
    fitted: bool,
    feature_names: Vec<String>,
    numeric_idxs: Vec<usize>,
    numeric_edges: HashMap<usize, Vec<f64>>,
    summary_rows: Vec<PreprocessorSummaryRow>,
}

impl NumericBinnerCore {
    pub fn new(n_bins: usize, method: &str, missing_token: String) -> Result<Self, WoeError> {
        if n_bins <= 1 {
            return Err(WoeError::InvalidNBins);
        }

        Ok(Self {
            n_bins,
            method: NumericBinningMethod::from_str(method)?,
            missing_token,
            fitted: false,
            feature_names: Vec::new(),
            numeric_idxs: Vec::new(),
            numeric_edges: HashMap::new(),
            summary_rows: Vec::new(),
        })
    }

    pub fn fit(
        &mut self,
        rows: &[Vec<Option<f64>>],
        feature_names: &[String],
        numeric_idxs: &[usize],
        targets: Option<&[u8]>,
        monotonic_constraints: Option<&[(usize, String)]>,
    ) -> Result<(), WoeError> {
        let ncols = validate_matrix(rows, None)?;
        if feature_names.len() != ncols {
            return Err(WoeError::FeatureNameCountMismatch);
        }

        let numeric_unique = stable_unique_usize(numeric_idxs);
        for &idx in &numeric_unique {
            if idx >= ncols {
                return Err(WoeError::FeatureIndexOutOfRange(idx));
            }
        }

        let needs_target = matches!(self.method, NumericBinningMethod::Tree)
            || monotonic_constraints.is_some_and(|v| !v.is_empty());
        if needs_target && targets.is_none() {
            return Err(WoeError::MissingTargetForBinning);
        }
        if let Some(target_values) = targets {
            validate_binary_targets(target_values)?;
            if target_values.len() != rows.len() {
                return Err(WoeError::LengthMismatch);
            }
        }

        let monotonic_map = parse_monotonic_map(monotonic_constraints, &numeric_unique)?;

        self.feature_names = feature_names.to_vec();
        self.numeric_idxs = numeric_unique;
        self.numeric_edges.clear();
        self.summary_rows.clear();
        self.summary_rows.reserve(self.numeric_idxs.len());

        for &idx in &self.numeric_idxs {
            let mut numeric_values: Vec<f64> = Vec::new();
            let mut numeric_targets: Vec<u8> = Vec::new();
            for (row_idx, row) in rows.iter().enumerate() {
                if let Some(value) = row[idx] {
                    numeric_values.push(value);
                    if let Some(target_values) = targets {
                        numeric_targets.push(target_values[row_idx]);
                    }
                }
            }

            let mut edges = match self.method {
                NumericBinningMethod::Quantile => {
                    compute_quantile_bin_edges(&numeric_values, self.n_bins)
                }
                NumericBinningMethod::Uniform => {
                    compute_uniform_bin_edges(&numeric_values, self.n_bins)
                }
                NumericBinningMethod::Kmeans => {
                    compute_kmeans_bin_edges(&numeric_values, self.n_bins, 100)
                }
                NumericBinningMethod::Tree => {
                    if numeric_targets.len() != numeric_values.len() {
                        return Err(WoeError::LengthMismatch);
                    }
                    compute_tree_bin_edges(&numeric_values, &numeric_targets, self.n_bins)
                }
            };

            if let Some(direction) = monotonic_map.get(&idx) {
                if numeric_targets.len() != numeric_values.len() {
                    return Err(WoeError::LengthMismatch);
                }
                edges =
                    enforce_monotonic_edges(&numeric_values, &numeric_targets, &edges, direction);
            }

            self.numeric_edges.insert(idx, edges.clone());
            let unique_non_missing = count_unique_floats(&numeric_values);
            self.summary_rows.push(PreprocessorSummaryRow {
                feature: self.feature_names[idx].clone(),
                original_unique: unique_non_missing,
                reduced_unique: (edges.len().saturating_sub(1)).max(1),
                coverage: 1.0,
            });
        }

        self.fitted = true;
        Ok(())
    }

    pub fn transform(&self, rows: &[Vec<Option<f64>>]) -> Result<Vec<Vec<String>>, WoeError> {
        if !self.fitted {
            return Err(WoeError::NotFitted);
        }
        if rows.is_empty() {
            return Ok(Vec::new());
        }

        let ncols = rows[0].len();
        if ncols == 0 || rows.iter().any(|row| row.len() != ncols) {
            return Err(WoeError::InvalidMatrixShape);
        }
        if ncols != self.feature_names.len() {
            return Err(WoeError::FeatureNameCountMismatch);
        }

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let mut out_row = vec![self.missing_token.clone(); row.len()];
            for &idx in &self.numeric_idxs {
                let edges = self.numeric_edges.get(&idx).ok_or(WoeError::NotFitted)?;
                out_row[idx] = if let Some(value) = row[idx] {
                    bin_label(value, edges)
                } else {
                    self.missing_token.clone()
                };
            }
            out.push(out_row);
        }
        Ok(out)
    }

    pub fn fit_transform(
        &mut self,
        rows: &[Vec<Option<f64>>],
        feature_names: &[String],
        numeric_idxs: &[usize],
        targets: Option<&[u8]>,
        monotonic_constraints: Option<&[(usize, String)]>,
    ) -> Result<Vec<Vec<String>>, WoeError> {
        self.fit(
            rows,
            feature_names,
            numeric_idxs,
            targets,
            monotonic_constraints,
        )?;
        self.transform(rows)
    }

    pub fn summary_rows(&self) -> Result<&[PreprocessorSummaryRow], WoeError> {
        if !self.fitted {
            return Err(WoeError::NotFitted);
        }
        Ok(&self.summary_rows)
    }
}

#[derive(Debug, Clone)]
pub struct BinaryTabularWoeModel {
    smoothing: f64,
    default_woe: f64,
    base_log_odds: Option<f64>,
    n_samples: usize,
    total_events: usize,
    total_non_events: usize,
    feature_names: Vec<String>,
    feature_index: HashMap<String, usize>,
    feature_mappings: Vec<HashMap<String, f64>>,
    feature_woe_se_mappings: Vec<HashMap<String, f64>>,
    fallback_woe_se: Vec<f64>,
    feature_stats: Vec<Vec<CategoryStats>>,
    feature_iv_stats: Vec<IvFeatureStats>,
}

impl BinaryTabularWoeModel {
    pub fn new(smoothing: f64, default_woe: f64) -> Self {
        Self {
            smoothing,
            default_woe,
            base_log_odds: None,
            n_samples: 0,
            total_events: 0,
            total_non_events: 0,
            feature_names: Vec::new(),
            feature_index: HashMap::new(),
            feature_mappings: Vec::new(),
            feature_woe_se_mappings: Vec::new(),
            fallback_woe_se: Vec::new(),
            feature_stats: Vec::new(),
            feature_iv_stats: Vec::new(),
        }
    }

    pub fn fit_matrix(
        &mut self,
        rows: &[Vec<String>],
        targets: &[u8],
        feature_names: Option<&[String]>,
    ) -> Result<(), WoeError> {
        if self.smoothing <= 0.0 {
            return Err(WoeError::InvalidSmoothing);
        }
        validate_binary_targets(targets)?;
        let ncols = validate_matrix(rows, Some(targets.len()))?;
        self.feature_names = resolve_feature_names(feature_names, ncols)?;

        let n = targets.len() as f64;
        let event_rate = targets.iter().map(|&v| f64::from(v)).sum::<f64>() / n;
        let clipped_event_rate = event_rate.clamp(1e-12, 1.0 - 1e-12);
        self.base_log_odds = Some((clipped_event_rate / (1.0 - clipped_event_rate)).ln());
        self.n_samples = targets.len();
        self.total_events = targets.iter().filter(|&&v| v == 1).count();
        self.total_non_events = self.n_samples.saturating_sub(self.total_events);

        self.feature_index = self
            .feature_names
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), idx))
            .collect::<HashMap<_, _>>();
        self.feature_mappings.clear();
        self.feature_woe_se_mappings.clear();
        self.fallback_woe_se.clear();
        self.feature_stats.clear();
        self.feature_iv_stats.clear();
        self.feature_mappings.reserve(self.feature_names.len());
        self.feature_woe_se_mappings
            .reserve(self.feature_names.len());
        self.fallback_woe_se.reserve(self.feature_names.len());
        self.feature_stats.reserve(self.feature_names.len());
        self.feature_iv_stats.reserve(self.feature_names.len());

        for col_idx in 0..self.feature_names.len() {
            let categories = rows
                .iter()
                .map(|row| row[col_idx].clone())
                .collect::<Vec<String>>();
            let stats = compute_binary_woe(&categories, targets, self.smoothing)?;
            let woe_mapping = stats
                .iter()
                .map(|s| (s.category.clone(), s.woe))
                .collect::<HashMap<_, _>>();
            let woe_se_mapping = stats
                .iter()
                .map(|s| (s.category.clone(), s.woe_se))
                .collect::<HashMap<_, _>>();
            let fallback_se = stats.iter().map(|s| s.woe_se).fold(0.0_f64, f64::max);
            let feature_name = self.feature_names[col_idx].clone();
            let iv_stats = compute_feature_iv_stats(
                &feature_name,
                &stats,
                self.total_events,
                self.total_non_events,
                self.smoothing,
                0.05,
            );

            self.feature_mappings.push(woe_mapping);
            self.feature_woe_se_mappings.push(woe_se_mapping);
            self.fallback_woe_se.push(fallback_se);
            self.feature_stats.push(stats);
            self.feature_iv_stats.push(iv_stats);
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
        if !(0.0..1.0).contains(&alpha) {
            return Err(WoeError::InvalidAlpha);
        }
        let base = self.base_log_odds.ok_or(WoeError::NotFitted)?;
        let z = normal_ppf(1.0 - alpha / 2.0);
        let decisions = self.decision_with_se_matrix(rows)?;
        Ok(decisions
            .into_iter()
            .map(|(score, se)| {
                let center = base + score;
                let lower = sigmoid(center - z * se);
                let upper = sigmoid(center + z * se);
                PredictionCi {
                    prediction: sigmoid(center),
                    lower_ci: lower.min(upper),
                    upper_ci: lower.max(upper),
                }
            })
            .collect())
    }

    pub fn decision_scores_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<f64>, WoeError> {
        Ok(self
            .decision_with_se_matrix(rows)?
            .into_iter()
            .map(|(score, _)| score)
            .collect())
    }

    fn decision_with_se_matrix(&self, rows: &[Vec<String>]) -> Result<Vec<(f64, f64)>, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        validate_matrix(rows, None)?;
        Ok(rows
            .iter()
            .map(|row| {
                let mut score = 0.0_f64;
                let mut se2 = 0.0_f64;
                for (col_idx, value) in row.iter().enumerate() {
                    let woe = self.feature_mappings[col_idx]
                        .get(value)
                        .copied()
                        .unwrap_or(self.default_woe);
                    let woe_se = self.feature_woe_se_mappings[col_idx]
                        .get(value)
                        .copied()
                        .unwrap_or(self.fallback_woe_se[col_idx]);
                    score += woe;
                    se2 += woe_se * woe_se;
                }
                (score, se2.sqrt())
            })
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

    pub fn iv_analysis(&self) -> Result<&[IvFeatureStats], WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        Ok(&self.feature_iv_stats)
    }

    pub fn iv_analysis_feature(&self, feature_name: &str) -> Result<IvFeatureStats, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        let idx = self
            .feature_index
            .get(feature_name)
            .copied()
            .ok_or_else(|| WoeError::UnknownFeature(feature_name.to_string()))?;
        self.feature_iv_stats
            .get(idx)
            .cloned()
            .ok_or_else(|| WoeError::UnknownFeature(feature_name.to_string()))
    }

    pub fn iv_analysis_with_alpha(&self, alpha: f64) -> Result<Vec<IvFeatureStats>, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        if !(0.0..1.0).contains(&alpha) {
            return Err(WoeError::InvalidAlpha);
        }
        Ok(self
            .feature_iv_stats
            .iter()
            .map(|s| with_alpha_iv_stats(s, alpha))
            .collect())
    }

    pub fn iv_analysis_feature_with_alpha(
        &self,
        feature_name: &str,
        alpha: f64,
    ) -> Result<IvFeatureStats, WoeError> {
        if self.base_log_odds.is_none() {
            return Err(WoeError::NotFitted);
        }
        if !(0.0..1.0).contains(&alpha) {
            return Err(WoeError::InvalidAlpha);
        }
        let raw = self.iv_analysis_feature(feature_name)?;
        Ok(with_alpha_iv_stats(&raw, alpha))
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
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        let per_class = self
            .models
            .iter()
            .map(|model| model.predict_ci_matrix(rows, alpha))
            .collect::<Result<Vec<_>, _>>()?;

        if per_class.is_empty() {
            return Ok(Vec::new());
        }
        let n_rows = per_class[0].len();
        let mut out = Vec::with_capacity(n_rows);
        for row_idx in 0..n_rows {
            let mut row = Vec::with_capacity(per_class.len());
            for class_ci in &per_class {
                row.push(class_ci[row_idx]);
            }
            out.push(row);
        }
        Ok(out)
    }

    pub fn predict_ci_class(
        &self,
        rows: &[Vec<String>],
        class_label: &str,
        alpha: f64,
    ) -> Result<Vec<PredictionCi>, WoeError> {
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
        model.predict_ci_matrix(rows, alpha)
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

    pub fn iv_analysis_class(&self, class_label: &str) -> Result<Vec<IvFeatureStats>, WoeError> {
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
        Ok(model.iv_analysis()?.to_vec())
    }

    pub fn iv_analysis_class_feature(
        &self,
        class_label: &str,
        feature_name: &str,
    ) -> Result<IvFeatureStats, WoeError> {
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
        model.iv_analysis_feature(feature_name)
    }

    pub fn iv_analysis_class_with_alpha(
        &self,
        class_label: &str,
        alpha: f64,
    ) -> Result<Vec<IvFeatureStats>, WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        if !(0.0..1.0).contains(&alpha) {
            return Err(WoeError::InvalidAlpha);
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
        model.iv_analysis_with_alpha(alpha)
    }

    pub fn iv_analysis_class_feature_with_alpha(
        &self,
        class_label: &str,
        feature_name: &str,
        alpha: f64,
    ) -> Result<IvFeatureStats, WoeError> {
        if self.class_labels.is_empty() {
            return Err(WoeError::MulticlassNotFitted);
        }
        if !(0.0..1.0).contains(&alpha) {
            return Err(WoeError::InvalidAlpha);
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
        model.iv_analysis_feature_with_alpha(feature_name, alpha)
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

fn validate_matrix<T>(rows: &[Vec<T>], expected_rows: Option<usize>) -> Result<usize, WoeError> {
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

fn stable_unique_usize(values: &[usize]) -> Vec<usize> {
    let mut seen: HashSet<usize> = HashSet::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if seen.insert(*value) {
            out.push(*value);
        }
    }
    out
}

fn stable_unique_strings(values: &[String]) -> Vec<String> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if seen.insert(value.clone()) {
            out.push(value.clone());
        }
    }
    out
}

fn sorted_counts_by_freq_then_key(counts: &HashMap<String, usize>) -> Vec<(String, usize)> {
    let mut out = counts
        .iter()
        .map(|(k, v)| (k.clone(), *v))
        .collect::<Vec<(String, usize)>>();
    out.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

fn parse_monotonic_map(
    monotonic_constraints: Option<&[(usize, String)]>,
    numeric_idxs: &[usize],
) -> Result<HashMap<usize, String>, WoeError> {
    let numeric_set = numeric_idxs.iter().copied().collect::<HashSet<usize>>();
    let mut out = HashMap::new();
    if let Some(constraints) = monotonic_constraints {
        for (idx, direction) in constraints {
            if !numeric_set.contains(idx) {
                return Err(WoeError::FeatureIndexOutOfRange(*idx));
            }
            out.insert(*idx, normalize_monotonic_direction(direction)?);
        }
    }
    Ok(out)
}

fn normalize_monotonic_direction(direction: &str) -> Result<String, WoeError> {
    match direction.trim().to_ascii_lowercase().as_str() {
        "increasing" | "inc" | "ascending" | "up" => Ok("increasing".to_string()),
        "decreasing" | "dec" | "descending" | "down" => Ok("decreasing".to_string()),
        other => Err(WoeError::InvalidMonotonicDirection(other.to_string())),
    }
}

fn count_unique_floats(values: &[f64]) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted.dedup_by(|a, b| *a == *b);
    sorted.len()
}

fn is_close(a: f64, b: f64) -> bool {
    let rel_tol = 1e-9_f64;
    (a - b).abs() <= rel_tol * a.abs().max(b.abs())
}

fn canonicalize_edges(mut edges: Vec<f64>) -> Vec<f64> {
    if edges.is_empty() {
        return vec![0.0, 1.0];
    }
    edges.sort_by(|a, b| a.total_cmp(b));
    edges.dedup_by(|a, b| *a == *b);
    if edges.len() < 2 {
        let only = edges[0];
        return vec![only, only + 1.0];
    }
    let last = edges.len() - 1;
    if is_close(edges[0], edges[last]) {
        edges[last] += 1.0;
    }
    edges
}

fn linear_quantile_sorted(sorted_values: &[f64], q: f64) -> f64 {
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }
    let clamped_q = q.clamp(0.0, 1.0);
    let h = (sorted_values.len() as f64 - 1.0) * clamped_q;
    let lower = h.floor() as usize;
    let upper = h.ceil() as usize;
    if lower == upper {
        return sorted_values[lower];
    }
    let weight = h - lower as f64;
    sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
}

fn compute_quantile_bin_edges(values: &[f64], n_bins: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0, 1.0];
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let vmin = sorted[0];
    let vmax = sorted[sorted.len() - 1];
    if is_close(vmin, vmax) {
        return vec![vmin, vmax + 1.0];
    }

    let mut edges = Vec::with_capacity(n_bins + 1);
    for idx in 0..=n_bins {
        let q = idx as f64 / n_bins as f64;
        edges.push(linear_quantile_sorted(&sorted, q));
    }
    canonicalize_edges(edges)
}

fn compute_uniform_bin_edges(values: &[f64], n_bins: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0, 1.0];
    }

    let vmin = values.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if is_close(vmin, vmax) {
        return vec![vmin, vmax + 1.0];
    }

    let step = (vmax - vmin) / n_bins as f64;
    let mut edges = Vec::with_capacity(n_bins + 1);
    for idx in 0..=n_bins {
        edges.push(vmin + step * idx as f64);
    }
    canonicalize_edges(edges)
}

fn vector_allclose(left: &[f64], right: &[f64], atol: f64) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right.iter())
            .all(|(l, r)| (*l - *r).abs() <= atol)
}

fn compute_kmeans_bin_edges(values: &[f64], n_bins: usize, max_iter: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0, 1.0];
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.total_cmp(b));
    let vmin = sorted_values[0];
    let vmax = sorted_values[sorted_values.len() - 1];
    if is_close(vmin, vmax) {
        return vec![vmin, vmax + 1.0];
    }

    let mut unique_values = sorted_values.clone();
    unique_values.dedup_by(|a, b| *a == *b);
    let mut k = n_bins.min(unique_values.len());
    if k <= 1 {
        return vec![vmin, vmax];
    }

    let mut centers = Vec::with_capacity(k);
    for idx in 0..k {
        let q = idx as f64 / (k - 1) as f64;
        centers.push(linear_quantile_sorted(&sorted_values, q));
    }
    centers.sort_by(|a, b| a.total_cmp(b));
    centers.dedup_by(|a, b| *a == *b);
    if centers.len() < 2 {
        return vec![vmin, vmax];
    }

    k = centers.len();
    for _ in 0..max_iter {
        let mut sums = vec![0.0_f64; k];
        let mut counts = vec![0_usize; k];

        for value in values {
            let mut best_idx = 0_usize;
            let mut best_dist = (value - centers[0]).abs();
            for (center_idx, center_value) in centers.iter().enumerate().skip(1) {
                let dist = (value - center_value).abs();
                if dist < best_dist {
                    best_idx = center_idx;
                    best_dist = dist;
                }
            }
            sums[best_idx] += *value;
            counts[best_idx] += 1;
        }

        let mut new_centers = centers.clone();
        for center_idx in 0..k {
            if counts[center_idx] > 0 {
                new_centers[center_idx] = sums[center_idx] / counts[center_idx] as f64;
            }
        }
        new_centers.sort_by(|a, b| a.total_cmp(b));

        if vector_allclose(&new_centers, &centers, 1e-10) {
            centers = new_centers;
            break;
        }
        centers = new_centers;
    }

    centers.sort_by(|a, b| a.total_cmp(b));
    centers.dedup_by(|a, b| *a == *b);
    if centers.len() < 2 {
        return vec![vmin, vmax];
    }

    let mut edges = Vec::with_capacity(centers.len() + 1);
    edges.push(vmin);
    for idx in 0..(centers.len() - 1) {
        edges.push((centers[idx] + centers[idx + 1]) / 2.0);
    }
    edges.push(vmax);
    canonicalize_edges(edges)
}

fn compute_tree_bin_edges(values: &[f64], targets: &[u8], n_bins: usize) -> Vec<f64> {
    if values.is_empty() {
        return vec![0.0, 1.0];
    }

    let mut pairs = values
        .iter()
        .copied()
        .zip(targets.iter().copied())
        .collect::<Vec<(f64, u8)>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let sorted_values = pairs.iter().map(|(v, _)| *v).collect::<Vec<f64>>();
    let sorted_targets = pairs.iter().map(|(_, t)| *t).collect::<Vec<u8>>();

    let vmin = sorted_values[0];
    let vmax = sorted_values[sorted_values.len() - 1];
    if is_close(vmin, vmax) {
        return vec![vmin, vmax + 1.0];
    }

    let mut prefix_events = Vec::with_capacity(sorted_targets.len() + 1);
    prefix_events.push(0_usize);
    for target in sorted_targets {
        let previous = *prefix_events.last().unwrap_or(&0);
        prefix_events.push(previous + usize::from(target));
    }

    let mut segments = vec![(0_usize, sorted_values.len())];
    let mut thresholds: Vec<f64> = Vec::new();

    for _ in 0..n_bins.saturating_sub(1) {
        let mut best_gain = 0.0_f64;
        let mut best_segment_idx: Option<usize> = None;
        let mut best_split_idx: Option<usize> = None;

        for (segment_idx, (start, end)) in segments.iter().copied().enumerate() {
            let (split_idx, gain) = best_tree_split(&sorted_values, &prefix_events, start, end);
            if let Some(split) = split_idx {
                if gain > best_gain {
                    best_gain = gain;
                    best_segment_idx = Some(segment_idx);
                    best_split_idx = Some(split);
                }
            }
        }

        let Some(segment_idx) = best_segment_idx else {
            break;
        };
        let Some(split_idx) = best_split_idx else {
            break;
        };

        let (start, end) = segments.remove(segment_idx);
        let threshold = (sorted_values[split_idx] + sorted_values[split_idx + 1]) / 2.0;
        thresholds.push(threshold);
        segments.push((start, split_idx + 1));
        segments.push((split_idx + 1, end));
    }

    let mut edges = Vec::with_capacity(thresholds.len() + 2);
    edges.push(vmin);
    edges.extend(thresholds);
    edges.push(vmax);
    canonicalize_edges(edges)
}

fn best_tree_split(
    sorted_values: &[f64],
    prefix_events: &[usize],
    start: usize,
    end: usize,
) -> (Option<usize>, f64) {
    let total_count = end.saturating_sub(start);
    if total_count < 2 {
        return (None, 0.0);
    }

    let total_events = prefix_events[end] - prefix_events[start];
    let parent_impurity = gini_impurity(total_events, total_count);

    let mut best_gain = 0.0_f64;
    let mut best_split_idx: Option<usize> = None;
    for split_idx in start..(end - 1) {
        if is_close(sorted_values[split_idx], sorted_values[split_idx + 1]) {
            continue;
        }

        let left_count = split_idx - start + 1;
        let right_count = total_count - left_count;
        let left_events = prefix_events[split_idx + 1] - prefix_events[start];
        let right_events = total_events - left_events;

        let left_impurity = gini_impurity(left_events, left_count);
        let right_impurity = gini_impurity(right_events, right_count);
        let child_impurity = (left_count as f64 / total_count as f64) * left_impurity
            + (right_count as f64 / total_count as f64) * right_impurity;
        let gain = parent_impurity - child_impurity;
        if gain > best_gain {
            best_gain = gain;
            best_split_idx = Some(split_idx);
        }
    }

    (best_split_idx, best_gain)
}

fn gini_impurity(events: usize, count: usize) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let p = events as f64 / count as f64;
    2.0 * p * (1.0 - p)
}

#[derive(Debug, Clone, Copy)]
struct MonotonicBin {
    lo: f64,
    hi: f64,
    events: f64,
    count: f64,
}

fn merge_empty_monotonic_bins(mut bins: Vec<MonotonicBin>) -> Vec<MonotonicBin> {
    if bins.is_empty() {
        return vec![MonotonicBin {
            lo: 0.0,
            hi: 1.0,
            events: 0.0,
            count: 1.0,
        }];
    }

    let mut idx = 0_usize;
    while idx < bins.len() {
        if bins[idx].count > 0.0 {
            idx += 1;
            continue;
        }

        if bins.len() == 1 {
            bins[idx].count = 1.0;
            break;
        }
        if idx == 0 {
            bins[1].lo = bins[0].lo;
            bins.remove(0);
            continue;
        }

        bins[idx - 1].hi = bins[idx].hi;
        bins.remove(idx);
    }
    bins
}

fn enforce_monotonic_edges(
    values: &[f64],
    targets: &[u8],
    edges: &[f64],
    direction: &str,
) -> Vec<f64> {
    if edges.len() <= 2 || values.is_empty() {
        return edges.to_vec();
    }

    let mut bins = Vec::with_capacity(edges.len().saturating_sub(1));
    for idx in 0..(edges.len() - 1) {
        bins.push(MonotonicBin {
            lo: edges[idx],
            hi: edges[idx + 1],
            events: 0.0,
            count: 0.0,
        });
    }

    for (value, target) in values.iter().zip(targets.iter()) {
        let bin_idx = bin_index(*value, edges);
        bins[bin_idx].events += f64::from(*target);
        bins[bin_idx].count += 1.0;
    }

    bins = merge_empty_monotonic_bins(bins);
    if bins.len() <= 1 {
        return vec![bins[0].lo, bins[0].hi];
    }

    let mut idx = 0_usize;
    while idx < bins.len() - 1 {
        let left_rate = bins[idx].events / bins[idx].count;
        let right_rate = bins[idx + 1].events / bins[idx + 1].count;
        let violation = if direction == "increasing" {
            left_rate > right_rate
        } else {
            left_rate < right_rate
        };
        if !violation {
            idx += 1;
            continue;
        }

        let right = bins.remove(idx + 1);
        bins[idx].hi = right.hi;
        bins[idx].events += right.events;
        bins[idx].count += right.count;
        idx = idx.saturating_sub(1);
    }

    let mut out_edges = Vec::with_capacity(bins.len() + 1);
    out_edges.push(bins[0].lo);
    for bin in bins {
        out_edges.push(bin.hi);
    }
    if out_edges.len() < 2 {
        return vec![out_edges[0], out_edges[0] + 1.0];
    }
    let last = out_edges.len() - 1;
    if is_close(out_edges[0], out_edges[last]) {
        out_edges[last] += 1.0;
    }
    out_edges
}

fn bin_index(value: f64, edges: &[f64]) -> usize {
    if edges.len() < 2 {
        return 0;
    }

    let last_bin = edges.len() - 2;
    if value <= edges[0] {
        return 0;
    }
    if value >= edges[edges.len() - 1] {
        return last_bin;
    }

    let idx = edges.partition_point(|edge| *edge <= value);
    idx.saturating_sub(1).min(last_bin)
}

fn bin_label(value: f64, edges: &[f64]) -> String {
    format!("bin_{}", bin_index(value, edges))
}

fn compute_feature_iv_stats(
    feature: &str,
    stats: &[CategoryStats],
    total_events: usize,
    total_non_events: usize,
    smoothing: f64,
    alpha: f64,
) -> IvFeatureStats {
    let k = stats.len() as f64;
    let n_event = (total_events as f64 + smoothing * k).max(1e-12);
    let n_non_event = (total_non_events as f64 + smoothing * k).max(1e-12);

    let mut iv = 0.0_f64;
    let mut iv_var = 0.0_f64;
    for row in stats {
        let p_event = (row.event_count as f64 + smoothing) / n_event;
        let p_non_event = (row.non_event_count as f64 + smoothing) / n_non_event;
        let woe = row.woe;
        iv += (p_event - p_non_event) * woe;

        // Delta-method approximation of IV contribution variance.
        let d_event = woe + 1.0 - p_non_event / p_event;
        let d_non_event = -woe - p_event / p_non_event + 1.0;
        let var_event = p_event * (1.0 - p_event) / n_event;
        let var_non_event = p_non_event * (1.0 - p_non_event) / n_non_event;
        iv_var += d_event * d_event * var_event + d_non_event * d_non_event * var_non_event;
    }

    let iv_se = iv_var.max(0.0).sqrt();
    let z = normal_ppf(1.0 - alpha / 2.0);
    let iv_ci_lower = (iv - z * iv_se).max(0.0);
    let iv_ci_upper = (iv + z * iv_se).max(0.0);
    let iv_significance = if iv_ci_lower > 0.0 {
        "Significant"
    } else {
        "Not Significant"
    }
    .to_string();

    IvFeatureStats {
        feature: feature.to_string(),
        iv,
        iv_se,
        iv_ci_lower,
        iv_ci_upper,
        iv_significance,
    }
}

fn with_alpha_iv_stats(base: &IvFeatureStats, alpha: f64) -> IvFeatureStats {
    let z = normal_ppf(1.0 - alpha / 2.0);
    let iv_ci_lower = (base.iv - z * base.iv_se).max(0.0);
    let iv_ci_upper = (base.iv + z * base.iv_se).max(0.0);
    let iv_significance = if iv_ci_lower > 0.0 {
        "Significant"
    } else {
        "Not Significant"
    }
    .to_string();

    IvFeatureStats {
        feature: base.feature.clone(),
        iv: base.iv,
        iv_se: base.iv_se,
        iv_ci_lower,
        iv_ci_upper,
        iv_significance,
    }
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

    #[test]
    fn compute_binary_woe_includes_positive_standard_errors() {
        let cats = vec![
            "A".to_string(),
            "A".to_string(),
            "B".to_string(),
            "C".to_string(),
            "C".to_string(),
        ];
        let y = vec![1, 0, 1, 0, 1];
        let stats = compute_binary_woe(&cats, &y, 0.5).expect("woe should compute");
        assert!(stats.iter().all(|r| r.woe_se > 0.0));
    }

    #[test]
    fn binary_iv_analysis_has_consistent_bounds_and_significance() {
        let rows = vec![
            vec!["A".to_string(), "X".to_string()],
            vec!["A".to_string(), "Y".to_string()],
            vec!["B".to_string(), "X".to_string()],
            vec!["C".to_string(), "Z".to_string()],
            vec!["C".to_string(), "Y".to_string()],
        ];
        let y = vec![1, 0, 0, 1, 1];
        let feature_names = vec!["cat".to_string(), "bucket".to_string()];
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
        model
            .fit_matrix(&rows, &y, Some(&feature_names))
            .expect("fit should work");

        let iv_rows = model.iv_analysis().expect("iv analysis should work");
        assert_eq!(iv_rows.len(), feature_names.len());
        assert!(iv_rows.iter().all(|r| r.iv >= 0.0));
        assert!(iv_rows
            .iter()
            .all(|r| r.iv_ci_lower <= r.iv && r.iv <= r.iv_ci_upper));
    }

    #[test]
    fn iv_analysis_alpha_changes_interval_width() {
        let rows = vec![
            vec!["A".to_string()],
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["C".to_string()],
            vec!["C".to_string()],
            vec!["C".to_string()],
        ];
        let y = vec![1, 0, 0, 1, 1, 0];
        let mut model = BinaryTabularWoeModel::new(0.5, 0.0);
        model
            .fit_matrix(&rows, &y, Some(&["cat".to_string()]))
            .expect("fit should work");
        let iv_95 = model
            .iv_analysis_feature_with_alpha("cat", 0.05)
            .expect("iv analysis alpha .05 should work");
        let iv_99 = model
            .iv_analysis_feature_with_alpha("cat", 0.01)
            .expect("iv analysis alpha .01 should work");
        let w95 = iv_95.iv_ci_upper - iv_95.iv_ci_lower;
        let w99 = iv_99.iv_ci_upper - iv_99.iv_ci_lower;
        assert!(w99 > w95);
    }

    #[test]
    fn binary_ci_shrinks_with_more_training_data() {
        let base_rows = vec![
            vec!["A".to_string()],
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["C".to_string()],
        ];
        let base_y = vec![1, 0, 0, 1];
        let predict_rows = vec![vec!["A".to_string()]];

        let mut model_small = BinaryTabularWoeModel::new(0.5, 0.0);
        model_small
            .fit_matrix(&base_rows, &base_y, Some(&["cat".to_string()]))
            .expect("small fit should work");
        let ci_small = model_small
            .predict_ci_matrix(&predict_rows, 0.05)
            .expect("small ci should work")[0];
        let width_small = ci_small.upper_ci - ci_small.lower_ci;

        let mut large_rows = Vec::new();
        let mut large_y = Vec::new();
        for _ in 0..20 {
            large_rows.extend(base_rows.clone());
            large_y.extend(base_y.clone());
        }
        let mut model_large = BinaryTabularWoeModel::new(0.5, 0.0);
        model_large
            .fit_matrix(&large_rows, &large_y, Some(&["cat".to_string()]))
            .expect("large fit should work");
        let ci_large = model_large
            .predict_ci_matrix(&predict_rows, 0.05)
            .expect("large ci should work")[0];
        let width_large = ci_large.upper_ci - ci_large.lower_ci;

        assert!(width_large < width_small);
    }

    #[test]
    fn multiclass_iv_analysis_class_returns_per_feature_rows() {
        let rows = vec![
            vec!["A".to_string(), "X".to_string()],
            vec!["A".to_string(), "Y".to_string()],
            vec!["B".to_string(), "X".to_string()],
            vec!["C".to_string(), "Z".to_string()],
            vec!["C".to_string(), "Y".to_string()],
            vec!["B".to_string(), "Z".to_string()],
        ];
        let labels = vec![
            "c0".to_string(),
            "c1".to_string(),
            "c2".to_string(),
            "c0".to_string(),
            "c1".to_string(),
            "c2".to_string(),
        ];
        let feature_names = vec!["cat".to_string(), "bucket".to_string()];
        let mut model = MulticlassTabularWoeModel::new();
        model
            .fit_matrix(&rows, &labels, Some(&feature_names), 0.5, 0.0)
            .expect("fit should work");

        let rows_c0 = model
            .iv_analysis_class("c0")
            .expect("class iv analysis should work");
        assert_eq!(rows_c0.len(), feature_names.len());
        assert!(rows_c0.iter().all(|r| r.iv >= 0.0));
    }

    #[test]
    fn preprocessor_core_reduces_categories_and_preserves_missing() {
        let rows = vec![
            vec!["A".to_string()],
            vec!["A".to_string()],
            vec!["A".to_string()],
            vec!["B".to_string()],
            vec!["B".to_string()],
            vec!["C".to_string()],
            vec!["__missing__".to_string()],
        ];
        let feature_names = vec!["cat".to_string()];
        let mut pre = PreprocessorCore::new(
            Some(2),
            0.9,
            2,
            "__other__".to_string(),
            "__missing__".to_string(),
        )
        .expect("preprocessor should initialize");

        pre.fit(&rows, &feature_names, &[0])
            .expect("preprocessor fit should work");
        let out = pre.transform(&rows).expect("transform should work");
        let values = out.iter().map(|r| r[0].as_str()).collect::<Vec<_>>();

        assert_eq!(values.iter().filter(|&&v| v == "A").count(), 3);
        assert_eq!(values.iter().filter(|&&v| v == "B").count(), 0);
        assert!(values.contains(&"__missing__"));
        assert!(values.contains(&"__other__"));
    }

    #[test]
    fn preprocessor_core_unknown_maps_to_other() {
        let rows = vec![vec!["A".to_string()], vec!["B".to_string()]];
        let feature_names = vec!["cat".to_string()];
        let mut pre = PreprocessorCore::new(
            None,
            1.0,
            1,
            "__other__".to_string(),
            "__missing__".to_string(),
        )
        .expect("preprocessor should initialize");
        pre.fit(&rows, &feature_names, &[0])
            .expect("fit should work");

        let pred_rows = vec![vec!["A".to_string()], vec!["Z".to_string()]];
        let out = pre.transform(&pred_rows).expect("transform should work");
        assert_eq!(out[0][0], "A");
        assert_eq!(out[1][0], "__other__");
    }

    #[test]
    fn numeric_binner_core_quantile_binning_and_missing_work() {
        let rows = vec![
            vec![Some(1.0)],
            vec![Some(2.0)],
            vec![Some(3.0)],
            vec![Some(4.0)],
            vec![Some(5.0)],
            vec![None],
        ];
        let feature_names = vec!["num".to_string()];

        let mut binner = NumericBinnerCore::new(3, "quantile", "__missing__".to_string())
            .expect("numeric binner should initialize");
        binner
            .fit(&rows, &feature_names, &[0], None, None)
            .expect("numeric fit should work");
        let out = binner
            .transform(&rows)
            .expect("numeric transform should work");

        assert!(out[0][0].starts_with("bin_"));
        assert_eq!(out[5][0], "__missing__");
        let summary = binner.summary_rows().expect("summary should be available");
        assert_eq!(summary.len(), 1);
        assert_eq!(summary[0].feature, "num");
    }

    #[test]
    fn numeric_binner_core_enforces_monotonic_increasing() {
        let rows = vec![
            vec![Some(1.0)],
            vec![Some(2.0)],
            vec![Some(3.0)],
            vec![Some(4.0)],
            vec![Some(5.0)],
            vec![Some(6.0)],
            vec![Some(7.0)],
            vec![Some(8.0)],
        ];
        let targets = vec![0, 0, 1, 1, 0, 0, 1, 1];
        let feature_names = vec!["num".to_string()];
        let monotonic = vec![(0_usize, "increasing".to_string())];

        let mut binner = NumericBinnerCore::new(4, "quantile", "__missing__".to_string())
            .expect("numeric binner should initialize");
        let transformed = binner
            .fit_transform(
                &rows,
                &feature_names,
                &[0],
                Some(&targets),
                Some(&monotonic),
            )
            .expect("fit_transform should work");

        let mut rates: HashMap<usize, (usize, usize)> = HashMap::new();
        for (row, target) in transformed.iter().zip(targets.iter()) {
            let bin_idx = row[0]
                .strip_prefix("bin_")
                .expect("bin prefix should be present")
                .parse::<usize>()
                .expect("bin index should be parseable");
            let entry = rates.entry(bin_idx).or_insert((0, 0));
            entry.0 += usize::from(*target);
            entry.1 += 1;
        }

        let mut sorted_bins = rates.into_iter().collect::<Vec<(usize, (usize, usize))>>();
        sorted_bins.sort_by(|a, b| a.0.cmp(&b.0));
        let event_rates = sorted_bins
            .iter()
            .map(|(_, (events, count))| *events as f64 / *count as f64)
            .collect::<Vec<f64>>();

        assert!(!event_rates.is_empty());
        for idx in 0..event_rates.len().saturating_sub(1) {
            assert!(event_rates[idx] <= event_rates[idx + 1]);
        }
    }
}
