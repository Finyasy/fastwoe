"""High-level Python wrapper with NumPy/pandas-friendly input handling."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Iterable
from numbers import Integral
from typing import Any

from .fastwoe_rs import FastWoe as _RustFastWoe


class FastWoe:
    """Python-friendly wrapper around the Rust-backed FastWoe implementation."""

    def __init__(
        self,
        smoothing: float = 0.5,
        default_woe: float = 0.0,
        warn_on_assumption_risk: bool = True,
        dependence_warning_threshold: float = 0.8,
        sparse_singleton_fraction_threshold: float = 0.1,
        sparse_rare_fraction_threshold: float = 0.2,
        sparse_unique_ratio_threshold: float = 0.5,
        sparse_rare_count: int = 3,
        max_dependency_categories: int = 50,
        min_rows_for_assumption_warnings: int = 50,
    ) -> None:
        self._inner = _RustFastWoe(smoothing=smoothing, default_woe=default_woe)
        if not 0.0 <= dependence_warning_threshold <= 1.0:
            raise ValueError("dependence_warning_threshold must be in [0, 1].")
        if not 0.0 <= sparse_singleton_fraction_threshold <= 1.0:
            raise ValueError("sparse_singleton_fraction_threshold must be in [0, 1].")
        if not 0.0 <= sparse_rare_fraction_threshold <= 1.0:
            raise ValueError("sparse_rare_fraction_threshold must be in [0, 1].")
        if not 0.0 <= sparse_unique_ratio_threshold <= 1.0:
            raise ValueError("sparse_unique_ratio_threshold must be in [0, 1].")
        if sparse_rare_count <= 0:
            raise ValueError("sparse_rare_count must be positive.")
        if max_dependency_categories <= 1:
            raise ValueError("max_dependency_categories must be greater than 1.")
        if min_rows_for_assumption_warnings <= 0:
            raise ValueError("min_rows_for_assumption_warnings must be positive.")

        self._warn_on_assumption_risk = bool(warn_on_assumption_risk)
        self._dependence_warning_threshold = float(dependence_warning_threshold)
        self._sparse_singleton_fraction_threshold = float(sparse_singleton_fraction_threshold)
        self._sparse_rare_fraction_threshold = float(sparse_rare_fraction_threshold)
        self._sparse_unique_ratio_threshold = float(sparse_unique_ratio_threshold)
        self._sparse_rare_count = int(sparse_rare_count)
        self._max_dependency_categories = int(max_dependency_categories)
        self._min_rows_for_assumption_warnings = int(min_rows_for_assumption_warnings)
        self._assumption_diagnostics = _empty_assumption_diagnostics()
        self._assumption_warning_emitted = False

    def fit(self, categories: Any, targets: Any) -> None:
        categories_1d = _to_1d_str(categories)
        self._inner.fit(categories_1d, _to_u8(targets))
        self._record_assumption_diagnostics(
            _to_single_feature_matrix(categories_1d), feature_names=["feature_0"]
        )

    def transform(self, categories: Any, as_frame: bool = False) -> Any:
        values = self._inner.transform(_to_1d_str(categories))
        if as_frame:
            return _to_frame(values, columns=["woe"], index=_extract_index(categories))
        return values

    def fit_transform(self, categories: Any, targets: Any, as_frame: bool = False) -> Any:
        categories_1d = _to_1d_str(categories)
        values = self._inner.fit_transform(categories_1d, _to_u8(targets))
        self._record_assumption_diagnostics(
            _to_single_feature_matrix(categories_1d), feature_names=["feature_0"]
        )
        if as_frame:
            return _to_frame(values, columns=["woe"], index=_extract_index(categories))
        return values

    def predict_proba(self, categories: Any) -> list[float]:
        self._warn_if_assumption_risk("predict_proba")
        return self._inner.predict_proba(_to_1d_str(categories))

    def predict_ci(self, categories: Any, alpha: float = 0.05, as_frame: bool = False) -> Any:
        self._warn_if_assumption_risk("predict_ci")
        values = self._inner.predict_ci(_to_1d_str(categories), alpha)
        if as_frame:
            return _to_ci_frame(values, index=_extract_index(categories))
        return values

    def get_mapping(self) -> list[Any]:
        return self._inner.get_mapping()

    def fit_matrix(self, rows: Any, targets: Any, feature_names: Any = None) -> None:
        matrix = _to_2d_str(rows)
        resolved_feature_names = _to_feature_names(feature_names)
        self._inner.fit_matrix(matrix, _to_u8(targets), resolved_feature_names)
        self._record_assumption_diagnostics(
            matrix, feature_names=_coalesce_feature_names(matrix, resolved_feature_names)
        )

    def transform_matrix(self, rows: Any, as_frame: bool = False) -> Any:
        values = self._inner.transform_matrix(_to_2d_str(rows))
        if as_frame:
            return _to_frame(values, columns=self.get_feature_names(), index=_extract_index(rows))
        return values

    def fit_transform_matrix(
        self, rows: Any, targets: Any, feature_names: Any = None, as_frame: bool = False
    ) -> Any:
        matrix = _to_2d_str(rows)
        resolved_feature_names = _to_feature_names(feature_names)
        values = self._inner.fit_transform_matrix(
            matrix, _to_u8(targets), resolved_feature_names
        )
        self._record_assumption_diagnostics(
            matrix, feature_names=_coalesce_feature_names(matrix, resolved_feature_names)
        )
        if as_frame:
            return _to_frame(values, columns=self.get_feature_names(), index=_extract_index(rows))
        return values

    def predict_proba_matrix(self, rows: Any) -> list[float]:
        self._warn_if_assumption_risk("predict_proba_matrix")
        return self._inner.predict_proba_matrix(_to_2d_str(rows))

    def predict_ci_matrix(self, rows: Any, alpha: float = 0.05, as_frame: bool = False) -> Any:
        self._warn_if_assumption_risk("predict_ci_matrix")
        values = self._inner.predict_ci_matrix(_to_2d_str(rows), alpha)
        if as_frame:
            return _to_ci_frame(values, index=_extract_index(rows))
        return values

    def get_feature_names(self) -> list[str]:
        return self._inner.get_feature_names()

    def get_feature_mapping(self, feature_name: str) -> list[Any]:
        return self._inner.get_feature_mapping(str(feature_name))

    def get_iv_analysis(
        self, feature_name: Any = None, alpha: float = 0.05, as_frame: bool = False
    ) -> Any:
        rows = self._inner.get_iv_analysis(
            None if feature_name is None else str(feature_name),
            alpha,
        )
        if as_frame:
            return _iv_rows_to_frame(rows)
        return rows

    def fit_multiclass(self, categories: Any, class_labels: Any) -> None:
        categories_1d = _to_1d_str(categories)
        self._inner.fit_multiclass(categories_1d, _to_1d_str(class_labels))
        self._record_assumption_diagnostics(
            _to_single_feature_matrix(categories_1d), feature_names=["feature_0"]
        )

    def predict_proba_multiclass(self, categories: Any, as_frame: bool = False) -> Any:
        self._warn_if_assumption_risk("predict_proba_multiclass")
        values = self._inner.predict_proba_multiclass(_to_1d_str(categories))
        if as_frame:
            cols = [f"proba_{c}" for c in self.get_class_labels()]
            return _to_frame(values, columns=cols, index=_extract_index(categories))
        return values

    def predict_ci_multiclass(
        self, categories: Any, alpha: float = 0.05
    ) -> list[list[tuple[float, float, float]]]:
        self._warn_if_assumption_risk("predict_ci_multiclass")
        return self._inner.predict_ci_multiclass(_to_1d_str(categories), alpha)

    def predict_proba_class(self, categories: Any, class_label: Any) -> list[float]:
        self._warn_if_assumption_risk("predict_proba_class")
        return self._inner.predict_proba_class(_to_1d_str(categories), str(class_label))

    def predict_ci_class(
        self, categories: Any, class_label: Any, alpha: float = 0.05
    ) -> list[tuple[float, float, float]]:
        self._warn_if_assumption_risk("predict_ci_class")
        return self._inner.predict_ci_class(_to_1d_str(categories), str(class_label), alpha)

    def get_mapping_multiclass(self, class_label: Any) -> list[Any]:
        return self._inner.get_mapping_multiclass(str(class_label))

    def fit_matrix_multiclass(
        self, rows: Any, class_labels: Any, feature_names: Any = None
    ) -> None:
        matrix = _to_2d_str(rows)
        resolved_feature_names = _to_feature_names(feature_names)
        self._inner.fit_matrix_multiclass(
            matrix, _to_1d_str(class_labels), resolved_feature_names
        )
        self._record_assumption_diagnostics(
            matrix, feature_names=_coalesce_feature_names(matrix, resolved_feature_names)
        )

    def predict_proba_matrix_multiclass(self, rows: Any, as_frame: bool = False) -> Any:
        self._warn_if_assumption_risk("predict_proba_matrix_multiclass")
        values = self._inner.predict_proba_matrix_multiclass(_to_2d_str(rows))
        if as_frame:
            cols = [f"proba_{c}" for c in self.get_class_labels()]
            return _to_frame(values, columns=cols, index=_extract_index(rows))
        return values

    def transform_matrix_multiclass(self, rows: Any, as_frame: bool = False) -> Any:
        values = self._inner.transform_matrix_multiclass(_to_2d_str(rows))
        if as_frame:
            cols = self.get_feature_names_multiclass()
            return _to_frame(values, columns=cols, index=_extract_index(rows))
        return values

    def predict_ci_matrix_multiclass(
        self, rows: Any, alpha: float = 0.05
    ) -> list[list[tuple[float, float, float]]]:
        self._warn_if_assumption_risk("predict_ci_matrix_multiclass")
        return self._inner.predict_ci_matrix_multiclass(_to_2d_str(rows), alpha)

    def predict_proba_matrix_class(self, rows: Any, class_label: Any) -> list[float]:
        self._warn_if_assumption_risk("predict_proba_matrix_class")
        return self._inner.predict_proba_matrix_class(_to_2d_str(rows), str(class_label))

    def predict_ci_matrix_class(
        self, rows: Any, class_label: Any, alpha: float = 0.05
    ) -> list[tuple[float, float, float]]:
        self._warn_if_assumption_risk("predict_ci_matrix_class")
        return self._inner.predict_ci_matrix_class(_to_2d_str(rows), str(class_label), alpha)

    def get_class_labels(self) -> list[str]:
        return self._inner.get_class_labels()

    def get_feature_names_multiclass(self) -> list[str]:
        return self._inner.get_feature_names_multiclass()

    def get_feature_mapping_multiclass(self, class_label: Any, feature_name: Any) -> list[Any]:
        return self._inner.get_feature_mapping_multiclass(str(class_label), str(feature_name))

    def get_iv_analysis_multiclass(
        self,
        class_label: Any,
        feature_name: Any = None,
        alpha: float = 0.05,
        as_frame: bool = False,
    ) -> Any:
        rows = self._inner.get_iv_analysis_multiclass(
            str(class_label),
            None if feature_name is None else str(feature_name),
            alpha,
        )
        if as_frame:
            return _iv_rows_to_frame(rows)
        return rows

    def get_assumption_diagnostics(self) -> dict[str, Any]:
        return _clone_assumption_diagnostics(self._assumption_diagnostics)

    def _record_assumption_diagnostics(
        self, rows: list[list[str]], feature_names: list[str]
    ) -> None:
        self._assumption_diagnostics = _compute_assumption_diagnostics(
            rows,
            feature_names,
            dependence_warning_threshold=self._dependence_warning_threshold,
            sparse_singleton_fraction_threshold=self._sparse_singleton_fraction_threshold,
            sparse_rare_fraction_threshold=self._sparse_rare_fraction_threshold,
            sparse_unique_ratio_threshold=self._sparse_unique_ratio_threshold,
            sparse_rare_count=self._sparse_rare_count,
            max_dependency_categories=self._max_dependency_categories,
        )
        self._assumption_warning_emitted = False

    def _warn_if_assumption_risk(self, api_name: str) -> None:
        if not self._warn_on_assumption_risk or self._assumption_warning_emitted:
            return
        if self._assumption_diagnostics.get("n_rows", 0) < self._min_rows_for_assumption_warnings:
            return
        if not self._assumption_diagnostics.get("at_risk", False):
            return

        dependence = self._assumption_diagnostics["dependence"]
        sparsity = self._assumption_diagnostics["sparsity"]
        messages: list[str] = []
        if dependence["at_risk"]:
            pair = dependence["worst_pair"]
            if pair is None:
                pair_repr = "unknown-pair"
            else:
                pair_repr = f"{pair[0]} vs {pair[1]}"
            messages.append(
                "strong feature dependence"
                f" (max Cramer's V={dependence['max_cramers_v']:.3f} for {pair_repr},"
                f" threshold={dependence['threshold']:.2f})"
            )
        if sparsity["at_risk"]:
            risky_features = sparsity["risky_features"]
            preview = ", ".join(
                f"{row['feature']} (singleton={row['singleton_fraction']:.1%}, "
                f"rare={row['rare_fraction']:.1%}, unique={row['unique_ratio']:.1%})"
                for row in risky_features[:3]
            )
            if preview:
                messages.append(f"ultra-sparse categories detected: {preview}")
            else:
                messages.append("ultra-sparse categories detected")

        warning_message = (
            f"FastWoe assumption-risk warning during {api_name}: "
            + "; ".join(messages)
            + ". Probability/CI inference uses a Naive Bayes-style additive WOE assumption. "
            + "See docs/validation/ASSUMPTIONS_AND_LIMITATIONS.md."
        )
        warnings.warn(warning_message, UserWarning, stacklevel=3)
        self._assumption_warning_emitted = True


def _to_feature_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    names = [str(v) for v in _to_1d_list(value)]
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"feature_names must be unique; duplicate found: {name}")
        seen.add(name)
    return names


def _to_u8(value: Any) -> list[int]:
    out: list[int] = []
    for raw in _to_1d_list(value):
        if isinstance(raw, bool):
            out.append(int(raw))
            continue
        if isinstance(raw, Integral):
            candidate = int(raw)
            if candidate in {0, 1}:
                out.append(candidate)
                continue
            raise ValueError("target must be binary with values in {0, 1}.")
        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped in {"0", "1"}:
                out.append(int(stripped))
                continue
            raise ValueError("target must be binary with values in {0, 1}.")
        if isinstance(raw, float):
            if math.isnan(raw):
                raise ValueError("target must be binary with values in {0, 1}.")
            if raw in {0.0, 1.0}:
                out.append(int(raw))
                continue
            raise ValueError("target must be binary with values in {0, 1}.")
        raise ValueError("target must be binary with values in {0, 1}.")
    return out


def _to_1d_str(value: Any) -> list[str]:
    return [str(v) for v in _to_1d_list(value)]


def _to_1d_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        out = value.tolist()
        if isinstance(out, list):
            if out and isinstance(out[0], list):
                if len(out[0]) == 1:
                    return [row[0] for row in out]
            return out
    if hasattr(value, "to_list"):
        out = value.to_list()
        if isinstance(out, list):
            return out
    if _is_pandas_frame(value):
        rows = value.values.tolist()
        if not rows:
            return []
        if len(rows[0]) != 1:
            raise ValueError("Expected a single-column input for 1D conversion.")
        return [row[0] for row in rows]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return list(value)
    raise TypeError("Unsupported 1D input type.")


def _to_2d_str(value: Any) -> list[list[str]]:
    if _is_pandas_frame(value):
        return [[str(cell) for cell in row] for row in value.astype(str).values.tolist()]

    if hasattr(value, "tolist"):
        out = value.tolist()
        if isinstance(out, list):
            if not out:
                return []
            if isinstance(out[0], list):
                return [[str(cell) for cell in row] for row in out]
            return [[str(cell)] for cell in out]

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        rows = list(value)
        if not rows:
            return []
        if isinstance(rows[0], Iterable) and not isinstance(rows[0], (str, bytes)):
            return [[str(cell) for cell in row] for row in rows]
        return [[str(cell)] for cell in rows]

    raise TypeError("Unsupported matrix input type.")


def _to_single_feature_matrix(values: list[str]) -> list[list[str]]:
    return [[value] for value in values]


def _coalesce_feature_names(rows: list[list[str]], names: list[str] | None) -> list[str]:
    if names is not None:
        return list(names)
    if not rows:
        return []
    return [f"feature_{idx}" for idx in range(len(rows[0]))]


def _empty_assumption_diagnostics() -> dict[str, Any]:
    return {
        "n_rows": 0,
        "n_features": 0,
        "at_risk": False,
        "dependence": {
            "threshold": 0.0,
            "evaluated_pairs": 0,
            "max_cramers_v": 0.0,
            "worst_pair": None,
            "at_risk": False,
        },
        "sparsity": {
            "singleton_fraction_threshold": 0.0,
            "rare_fraction_threshold": 0.0,
            "unique_ratio_threshold": 0.0,
            "rare_count_threshold": 0,
            "risky_features": [],
            "at_risk": False,
        },
    }


def _clone_assumption_diagnostics(value: dict[str, Any]) -> dict[str, Any]:
    dependence = value.get("dependence", {})
    sparsity = value.get("sparsity", {})
    risky_features = [
        {
            "feature": str(item["feature"]),
            "singleton_fraction": float(item["singleton_fraction"]),
            "rare_fraction": float(item["rare_fraction"]),
            "unique_ratio": float(item["unique_ratio"]),
            "unique_count": int(item["unique_count"]),
        }
        for item in sparsity.get("risky_features", [])
    ]
    worst_pair = dependence.get("worst_pair")
    if worst_pair is not None:
        worst_pair = [str(worst_pair[0]), str(worst_pair[1])]

    return {
        "n_rows": int(value.get("n_rows", 0)),
        "n_features": int(value.get("n_features", 0)),
        "at_risk": bool(value.get("at_risk", False)),
        "dependence": {
            "threshold": float(dependence.get("threshold", 0.0)),
            "evaluated_pairs": int(dependence.get("evaluated_pairs", 0)),
            "max_cramers_v": float(dependence.get("max_cramers_v", 0.0)),
            "worst_pair": worst_pair,
            "at_risk": bool(dependence.get("at_risk", False)),
        },
        "sparsity": {
            "singleton_fraction_threshold": float(
                sparsity.get("singleton_fraction_threshold", 0.0)
            ),
            "rare_fraction_threshold": float(sparsity.get("rare_fraction_threshold", 0.0)),
            "unique_ratio_threshold": float(sparsity.get("unique_ratio_threshold", 0.0)),
            "rare_count_threshold": int(sparsity.get("rare_count_threshold", 0)),
            "risky_features": risky_features,
            "at_risk": bool(sparsity.get("at_risk", False)),
        },
    }


def _compute_assumption_diagnostics(
    rows: list[list[str]],
    feature_names: list[str],
    dependence_warning_threshold: float,
    sparse_singleton_fraction_threshold: float,
    sparse_rare_fraction_threshold: float,
    sparse_unique_ratio_threshold: float,
    sparse_rare_count: int,
    max_dependency_categories: int,
) -> dict[str, Any]:
    if not rows:
        return _empty_assumption_diagnostics()

    resolved_names = _coalesce_feature_names(rows, feature_names)
    n_rows = len(rows)
    n_features = len(rows[0])

    dependence = _compute_dependence_diagnostics(
        rows,
        resolved_names,
        threshold=dependence_warning_threshold,
        max_dependency_categories=max_dependency_categories,
    )
    sparsity = _compute_sparsity_diagnostics(
        rows,
        resolved_names,
        singleton_fraction_threshold=sparse_singleton_fraction_threshold,
        rare_fraction_threshold=sparse_rare_fraction_threshold,
        unique_ratio_threshold=sparse_unique_ratio_threshold,
        rare_count_threshold=sparse_rare_count,
    )
    return {
        "n_rows": n_rows,
        "n_features": n_features,
        "at_risk": bool(dependence["at_risk"] or sparsity["at_risk"]),
        "dependence": dependence,
        "sparsity": sparsity,
    }


def _compute_dependence_diagnostics(
    rows: list[list[str]],
    feature_names: list[str],
    threshold: float,
    max_dependency_categories: int,
) -> dict[str, Any]:
    n_features = len(rows[0]) if rows else 0
    if n_features < 2:
        return {
            "threshold": threshold,
            "evaluated_pairs": 0,
            "max_cramers_v": 0.0,
            "worst_pair": None,
            "at_risk": False,
        }

    columns = [[row[idx] for row in rows] for idx in range(n_features)]
    unique_counts = [len(set(column)) for column in columns]

    max_pairs = 24
    evaluated_pairs = 0
    max_v = 0.0
    worst_pair: list[str] | None = None

    for left_idx in range(n_features):
        if unique_counts[left_idx] > max_dependency_categories:
            continue
        for right_idx in range(left_idx + 1, n_features):
            if unique_counts[right_idx] > max_dependency_categories:
                continue
            evaluated_pairs += 1
            v = _cramers_v(columns[left_idx], columns[right_idx])
            if v >= max_v:
                max_v = v
                worst_pair = [feature_names[left_idx], feature_names[right_idx]]
            if evaluated_pairs >= max_pairs:
                break
        if evaluated_pairs >= max_pairs:
            break

    return {
        "threshold": threshold,
        "evaluated_pairs": evaluated_pairs,
        "max_cramers_v": max_v,
        "worst_pair": worst_pair,
        "at_risk": bool(max_v >= threshold and evaluated_pairs > 0),
    }


def _compute_sparsity_diagnostics(
    rows: list[list[str]],
    feature_names: list[str],
    singleton_fraction_threshold: float,
    rare_fraction_threshold: float,
    unique_ratio_threshold: float,
    rare_count_threshold: int,
) -> dict[str, Any]:
    n_rows = len(rows)
    risky_features: list[dict[str, Any]] = []

    for idx, feature_name in enumerate(feature_names):
        counts = Counter(row[idx] for row in rows)
        unique_count = len(counts)
        singleton_rows = sum(count for count in counts.values() if count == 1)
        rare_rows = sum(count for count in counts.values() if count <= rare_count_threshold)
        singleton_fraction = singleton_rows / n_rows
        rare_fraction = rare_rows / n_rows
        unique_ratio = unique_count / n_rows
        if (
            singleton_fraction >= singleton_fraction_threshold
            or rare_fraction >= rare_fraction_threshold
            or unique_ratio >= unique_ratio_threshold
        ):
            risky_features.append(
                {
                    "feature": feature_name,
                    "singleton_fraction": singleton_fraction,
                    "rare_fraction": rare_fraction,
                    "unique_ratio": unique_ratio,
                    "unique_count": unique_count,
                }
            )

    risky_features.sort(
        key=lambda row: (
            row["singleton_fraction"],
            row["rare_fraction"],
            row["unique_ratio"],
        ),
        reverse=True,
    )
    return {
        "singleton_fraction_threshold": singleton_fraction_threshold,
        "rare_fraction_threshold": rare_fraction_threshold,
        "unique_ratio_threshold": unique_ratio_threshold,
        "rare_count_threshold": rare_count_threshold,
        "risky_features": risky_features,
        "at_risk": bool(risky_features),
    }


def _cramers_v(left: list[str], right: list[str]) -> float:
    if len(left) != len(right):
        raise ValueError("left and right vectors must have the same length.")
    if len(left) < 2:
        return 0.0

    left_categories = sorted(set(left))
    right_categories = sorted(set(right))
    n_left = len(left_categories)
    n_right = len(right_categories)
    if n_left < 2 or n_right < 2:
        return 0.0

    left_idx = {value: idx for idx, value in enumerate(left_categories)}
    right_idx = {value: idx for idx, value in enumerate(right_categories)}
    contingency = [[0 for _ in range(n_right)] for _ in range(n_left)]
    for left_value, right_value in zip(left, right):
        contingency[left_idx[left_value]][right_idx[right_value]] += 1

    row_totals = [sum(row) for row in contingency]
    col_totals = [sum(contingency[i][j] for i in range(n_left)) for j in range(n_right)]
    n_obs = float(len(left))
    chi2 = 0.0
    for row_idx, row_total in enumerate(row_totals):
        for col_idx, col_total in enumerate(col_totals):
            expected = row_total * col_total / n_obs
            if expected <= 0.0:
                continue
            observed = contingency[row_idx][col_idx]
            chi2 += ((observed - expected) ** 2) / expected

    phi2 = chi2 / n_obs
    denom = max(1.0, min(float(n_left - 1), float(n_right - 1)))
    return min(1.0, math.sqrt(phi2 / denom))


def _is_pandas_frame(value: Any) -> bool:
    return hasattr(value, "values") and hasattr(value, "columns") and hasattr(value, "dtypes")


def _extract_index(value: Any) -> Any:
    if hasattr(value, "index"):
        return value.index
    return None


def _to_frame(values: Any, columns: list[str], index: Any = None) -> Any:
    pd = _try_import_pandas()
    if pd is None:
        raise RuntimeError("pandas is required for as_frame=True outputs.")
    if values and not isinstance(values[0], list):
        values = [[v] for v in values]
    return pd.DataFrame(values, columns=columns, index=index)


def _to_ci_frame(values: list[tuple[float, float, float]], index: Any = None) -> Any:
    pd = _try_import_pandas()
    if pd is None:
        raise RuntimeError("pandas is required for as_frame=True outputs.")
    return pd.DataFrame(values, columns=["prediction", "lower_ci", "upper_ci"], index=index)


def _try_import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd


def _iv_rows_to_frame(rows: list[Any]) -> Any:
    pd = _try_import_pandas()
    if pd is None:
        raise RuntimeError("pandas is required for as_frame=True outputs.")
    return pd.DataFrame(
        [
            {
                "feature": row.feature,
                "iv": row.iv,
                "iv_se": row.iv_se,
                "iv_ci_lower": row.iv_ci_lower,
                "iv_ci_upper": row.iv_ci_upper,
                "iv_significance": row.iv_significance,
            }
            for row in rows
        ]
    )
