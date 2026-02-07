"""Preprocessing utilities for reducing high-cardinality categorical features."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass
class _InputMeta:
    kind: str
    is_1d: bool
    feature_names: list[str]
    index: Any


class WoePreprocessor:
    """Reduce categorical cardinality before WOE encoding.

    Parameters:
    - max_categories: hard cap on kept categories per feature.
    - top_p: keep categories until cumulative frequency reaches this ratio.
    - min_count: categories below this count are grouped as other.
    - other_token: replacement token for grouped categories.
    - missing_token: canonical token for missing values.
    """

    def __init__(
        self,
        max_categories: int | None = None,
        top_p: float = 1.0,
        min_count: int = 1,
        n_bins: int = 5,
        binning_method: str = "quantile",
        other_token: str = "__other__",
        missing_token: str = "__missing__",
    ) -> None:
        if max_categories is not None and max_categories <= 0:
            raise ValueError("max_categories must be positive when provided.")
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1].")
        if min_count <= 0:
            raise ValueError("min_count must be positive.")
        if n_bins <= 1:
            raise ValueError("n_bins must be greater than 1.")
        if binning_method not in {"quantile", "uniform"}:
            raise ValueError("binning_method must be one of: {'quantile', 'uniform'}.")

        self.max_categories = max_categories
        self.top_p = top_p
        self.min_count = min_count
        self.n_bins = n_bins
        self.binning_method = binning_method
        self.other_token = other_token
        self.missing_token = missing_token

        self._fitted = False
        self._feature_names: list[str] = []
        self._selected_idxs: list[int] = []
        self._numeric_idxs: list[int] = []
        self._numeric_edges: dict[int, list[float]] = {}
        self._allowed: dict[int, set[str]] = {}
        self._fit_counts: dict[int, OrderedDict[str, int]] = {}
        self._summary_rows: list[dict[str, Any]] = []

    def fit(
        self, X: Any, cat_features: Any = None, numerical_features: Any = None
    ) -> WoePreprocessor:
        rows, meta = _coerce_rows(X)
        if not rows:
            raise ValueError("Input must not be empty.")

        self._feature_names = meta.feature_names
        self._numeric_idxs = _resolve_feature_indices(numerical_features, self._feature_names)
        selected_idxs = _resolve_feature_indices(cat_features, self._feature_names)
        if not selected_idxs:
            selected_idxs = [
                i for i in range(len(self._feature_names)) if i not in self._numeric_idxs
            ]
        self._selected_idxs = [i for i in selected_idxs if i not in self._numeric_idxs]

        self._allowed.clear()
        self._numeric_edges.clear()
        self._fit_counts.clear()
        self._summary_rows.clear()

        # Learn numeric bin edges first.
        for idx in self._numeric_idxs:
            numeric_values = [
                float(v)
                for row in rows
                for v in [row[idx]]
                if _try_float(v) is not None
            ]
            edges = _compute_bin_edges(
                numeric_values, n_bins=self.n_bins, method=self.binning_method
            )
            self._numeric_edges[idx] = edges
            unique_non_missing = len(set(numeric_values))
            self._summary_rows.append(
                {
                    "feature": self._feature_names[idx],
                    "original_unique": unique_non_missing,
                    "reduced_unique": max(1, len(edges) - 1),
                    "coverage": 1.0,
                }
            )

        rows_for_cat = self._apply_numeric_binning(rows)

        for idx in self._selected_idxs:
            counts = OrderedDict()
            for row in rows_for_cat:
                category = self._normalize_value(row[idx])
                counts[category] = counts.get(category, 0) + 1

            sorted_counts = OrderedDict(
                sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            )
            total = sum(sorted_counts.values())
            keep = self._select_categories(sorted_counts, total)

            self._fit_counts[idx] = sorted_counts
            self._allowed[idx] = keep
            kept_count = sum(sorted_counts.get(cat, 0) for cat in keep)
            self._summary_rows.append(
                {
                    "feature": self._feature_names[idx],
                    "original_unique": len(sorted_counts),
                    "reduced_unique": len(keep) + 1,  # + __other__
                    "coverage": kept_count / total if total else 0.0,
                }
            )

        self._fitted = True
        return self

    def transform(self, X: Any) -> Any:
        self._require_fitted()
        rows, meta = _coerce_rows(X)
        rows = self._apply_numeric_binning(rows)
        transformed = []

        for row in rows:
            out_row = list(row)
            for idx in self._selected_idxs:
                category = self._normalize_value(row[idx])
                out_row[idx] = category if category in self._allowed[idx] else self.other_token
            transformed.append(out_row)

        return _restore_output(transformed, meta)

    def fit_transform(
        self, X: Any, cat_features: Any = None, numerical_features: Any = None
    ) -> Any:
        return self.fit(
            X, cat_features=cat_features, numerical_features=numerical_features
        ).transform(X)

    def get_reduction_summary(self, as_frame: bool = False) -> Any:
        self._require_fitted()
        rows = list(self._summary_rows)
        if as_frame:
            pd = _try_import_pandas()
            if pd is None:
                raise RuntimeError("pandas is required for as_frame=True.")
            return pd.DataFrame(rows)
        return rows

    def _normalize_value(self, value: Any) -> str:
        if value is None:
            return self.missing_token
        if isinstance(value, float) and math.isnan(value):
            return self.missing_token
        return str(value)

    def _apply_numeric_binning(self, rows: list[list[Any]]) -> list[list[Any]]:
        if not self._numeric_idxs:
            return [list(row) for row in rows]
        out = []
        for row in rows:
            new_row = list(row)
            for idx in self._numeric_idxs:
                value = _try_float(row[idx])
                if value is None:
                    new_row[idx] = self.missing_token
                    continue
                new_row[idx] = _bin_label(value, self._numeric_edges[idx])
            out.append(new_row)
        return out

    def _select_categories(self, counts: OrderedDict[str, int], total: int) -> set[str]:
        keep: list[str] = []
        cumulative = 0
        for category, count in counts.items():
            if count < self.min_count:
                continue
            keep.append(category)
            cumulative += count
            if total and cumulative / total >= self.top_p:
                break

        if not keep and counts:
            keep.append(next(iter(counts)))

        if self.max_categories is not None and len(keep) > self.max_categories:
            keep = keep[: self.max_categories]

        if self.missing_token in counts and self.missing_token not in keep:
            if self.max_categories is not None and len(keep) >= self.max_categories and keep:
                keep[-1] = self.missing_token
            else:
                keep.append(self.missing_token)

        return set(keep)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("WoePreprocessor is not fitted. Call fit() first.")


def _coerce_rows(X: Any) -> tuple[list[list[Any]], _InputMeta]:
    pd = _try_import_pandas()

    if pd is not None and isinstance(X, pd.DataFrame):
        rows = X.values.tolist()
        names = [str(c) for c in X.columns]
        return rows, _InputMeta(
            kind="dataframe", is_1d=False, feature_names=names, index=X.index
        )

    if hasattr(X, "tolist"):
        out = X.tolist()
        if isinstance(out, list):
            if not out:
                return [], _InputMeta(kind="list", is_1d=False, feature_names=[], index=None)
            if isinstance(out[0], list):
                names = [f"feature_{i}" for i in range(len(out[0]))]
                return out, _InputMeta(
                    kind="array", is_1d=False, feature_names=names, index=None
                )
            rows = [[v] for v in out]
            return rows, _InputMeta(
                kind="array", is_1d=True, feature_names=["feature_0"], index=None
            )

    if isinstance(X, Iterable) and not isinstance(X, (str, bytes)):
        rows_raw = list(X)
        if not rows_raw:
            return [], _InputMeta(kind="list", is_1d=False, feature_names=[], index=None)
        if isinstance(rows_raw[0], Iterable) and not isinstance(rows_raw[0], (str, bytes)):
            rows = [list(r) for r in rows_raw]
            names = [f"feature_{i}" for i in range(len(rows[0]))]
            return rows, _InputMeta(kind="list", is_1d=False, feature_names=names, index=None)
        rows = [[v] for v in rows_raw]
        return rows, _InputMeta(kind="list", is_1d=True, feature_names=["feature_0"], index=None)

    raise TypeError("Unsupported input type.")


def _restore_output(rows: list[list[Any]], meta: _InputMeta) -> Any:
    if meta.kind == "dataframe":
        pd = _try_import_pandas()
        if pd is None:
            raise RuntimeError("pandas is required for dataframe output.")
        return pd.DataFrame(rows, columns=meta.feature_names, index=meta.index)

    if meta.kind == "array":
        np = _try_import_numpy()
        if np is None:
            if meta.is_1d:
                return [r[0] for r in rows]
            return rows
        if meta.is_1d:
            return np.array([r[0] for r in rows], dtype=object)
        return np.array(rows, dtype=object)

    if meta.is_1d:
        return [r[0] for r in rows]
    return rows


def _resolve_feature_indices(cat_features: Any, feature_names: list[str]) -> list[int]:
    if cat_features is None:
        return []
    if isinstance(cat_features, (str, int)):
        cat_features = [cat_features]

    resolved: list[int] = []
    for feat in cat_features:
        if isinstance(feat, int):
            if feat < 0 or feat >= len(feature_names):
                raise ValueError(f"Feature index out of range: {feat}")
            resolved.append(feat)
            continue
        name = str(feat)
        if name not in feature_names:
            raise ValueError(f"Unknown feature name: {name}")
        resolved.append(feature_names.index(name))
    # Stable unique order
    return list(dict.fromkeys(resolved))


def _compute_bin_edges(values: list[float], n_bins: int, method: str) -> list[float]:
    if not values:
        # Degenerate case; transform will map everything to missing/other.
        return [0.0, 1.0]
    np = _try_import_numpy()
    if np is None:
        raise RuntimeError("numpy is required for numerical binning.")

    arr = np.array(values, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isclose(vmin, vmax):
        return [vmin, vmax + 1.0]

    if method == "quantile":
        edges = np.quantile(arr, np.linspace(0.0, 1.0, n_bins + 1))
    else:
        edges = np.linspace(vmin, vmax, n_bins + 1)

    unique_edges = sorted(set(float(v) for v in edges))
    if len(unique_edges) < 2:
        unique_edges = [vmin, vmax]
    if math.isclose(unique_edges[0], unique_edges[-1]):
        unique_edges[-1] = unique_edges[-1] + 1.0
    return unique_edges


def _bin_label(value: float, edges: list[float]) -> str:
    # Find highest edge <= value; right-closed final interval.
    bin_idx = 0
    last = len(edges) - 2
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        is_last = i == last
        if (lo <= value < hi) or (is_last and lo <= value <= hi):
            bin_idx = i
            break
    return f"bin_{bin_idx}"


def _try_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _try_import_pandas() -> Any:
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd


def _try_import_numpy() -> Any:
    try:
        import numpy as np
    except ImportError:
        return None
    return np
