"""Preprocessing utilities for reducing high-cardinality categorical features."""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from types import ModuleType
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
    - n_bins: number of bins for numeric features.
    - binning_method: one of {'quantile', 'uniform', 'kmeans', 'tree'}.
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
        if binning_method not in {"quantile", "uniform", "kmeans", "tree"}:
            raise ValueError(
                "binning_method must be one of: {'quantile', 'uniform', 'kmeans', 'tree'}."
            )

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
        self,
        X: Any,
        cat_features: Any = None,
        numerical_features: Any = None,
        target: Any = None,
        monotonic_constraints: Any = None,
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

        monotonic_map = _resolve_monotonic_constraints(
            monotonic_constraints, self._feature_names, self._numeric_idxs
        )
        target_rows = None
        needs_target = (self.binning_method == "tree" and self._numeric_idxs) or bool(monotonic_map)
        if needs_target:
            target_rows = _coerce_binary_target(target, expected_len=len(rows))

        # Learn numeric bin edges first.
        for idx in self._numeric_idxs:
            numeric_values: list[float] = []
            numeric_targets: list[int] = []
            for row_idx, row in enumerate(rows):
                parsed = _try_float(row[idx])
                if parsed is None:
                    continue
                numeric_values.append(parsed)
                if target_rows is not None:
                    numeric_targets.append(target_rows[row_idx])
            edges = _compute_bin_edges(
                numeric_values,
                n_bins=self.n_bins,
                method=self.binning_method,
                targets=numeric_targets if target_rows is not None else None,
            )
            if idx in monotonic_map:
                if target_rows is None:
                    raise RuntimeError(
                        "Internal error: target must be available for monotonic mode."
                    )
                edges = _enforce_monotonic_edges(
                    numeric_values,
                    numeric_targets,
                    edges,
                    direction=monotonic_map[idx],
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
        self,
        X: Any,
        cat_features: Any = None,
        numerical_features: Any = None,
        target: Any = None,
        monotonic_constraints: Any = None,
    ) -> Any:
        return self.fit(
            X,
            cat_features=cat_features,
            numerical_features=numerical_features,
            target=target,
            monotonic_constraints=monotonic_constraints,
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


def _resolve_monotonic_constraints(
    monotonic_constraints: Any,
    feature_names: list[str],
    numeric_idxs: list[int],
) -> dict[int, str]:
    if monotonic_constraints is None:
        return {}
    if not numeric_idxs:
        raise ValueError(
            "monotonic constraints require numerical_features to be selected."
        )

    if isinstance(monotonic_constraints, str):
        direction = _normalize_monotonic_direction(monotonic_constraints)
        return {idx: direction for idx in numeric_idxs}

    if not isinstance(monotonic_constraints, dict):
        raise TypeError(
            "monotonic_constraints must be a direction string or a dict of feature -> direction."
        )

    out: dict[int, str] = {}
    for feature, direction in monotonic_constraints.items():
        resolved = _resolve_feature_indices(feature, feature_names)
        if len(resolved) != 1:
            raise ValueError(f"Invalid monotonic constraint feature selector: {feature!r}")
        idx = resolved[0]
        if idx not in numeric_idxs:
            raise ValueError(
                "monotonic constraints can only be applied to selected numerical features."
            )
        out[idx] = _normalize_monotonic_direction(direction)
    return out


def _normalize_monotonic_direction(direction: Any) -> str:
    value = str(direction).strip().lower()
    aliases = {
        "increasing": "increasing",
        "inc": "increasing",
        "ascending": "increasing",
        "up": "increasing",
        "decreasing": "decreasing",
        "dec": "decreasing",
        "descending": "decreasing",
        "down": "decreasing",
    }
    if value not in aliases:
        raise ValueError(
            "monotonic direction must be one of: "
            "{'increasing', 'decreasing'} (aliases: inc/dec, ascending/descending)."
        )
    return aliases[value]


def _coerce_binary_target(target: Any, expected_len: int) -> list[int]:
    if target is None:
        raise ValueError(
            "target is required when binning_method='tree' and numerical_features are provided."
        )

    if hasattr(target, "tolist"):
        values = target.tolist()
    elif isinstance(target, Iterable) and not isinstance(target, (str, bytes)):
        values = list(target)
    else:
        raise TypeError("target must be an iterable of binary labels.")

    if len(values) != expected_len:
        raise ValueError(f"target length mismatch: expected {expected_len}, got {len(values)}.")

    out: list[int] = []
    for value in values:
        if isinstance(value, bool):
            out.append(int(value))
            continue
        if value in {0, 1}:
            out.append(int(value))
            continue
        if isinstance(value, str) and value.strip() in {"0", "1"}:
            out.append(int(value.strip()))
            continue
        raise ValueError("target must be binary with values in {0, 1}.")
    return out


def _compute_bin_edges(
    values: list[float],
    n_bins: int,
    method: str,
    targets: list[int] | None = None,
) -> list[float]:
    if not values:
        # Degenerate case; transform will map everything to missing/other.
        return [0.0, 1.0]
    if method == "tree":
        if targets is None:
            raise ValueError(
                "target is required when binning_method='tree' and numerical_features are provided."
            )
        if len(targets) != len(values):
            raise ValueError(
                "tree binning requires targets aligned with non-missing numerical rows."
            )
        return _compute_tree_bin_edges(values, targets, n_bins=n_bins)
    if method == "kmeans":
        return _compute_kmeans_bin_edges(values, n_bins=n_bins)

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
    return f"bin_{_bin_index(value, edges)}"


def _bin_index(value: float, edges: list[float]) -> int:
    # Clamp out-of-range values and use binary search for in-range values.
    if len(edges) < 2:
        return 0
    last_bin = len(edges) - 2
    if value <= edges[0]:
        return 0
    if value >= edges[-1]:
        return last_bin
    bin_idx = bisect_right(edges, value) - 1
    return max(0, min(bin_idx, last_bin))


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


def _compute_tree_bin_edges(values: list[float], targets: list[int], n_bins: int) -> list[float]:
    pairs = sorted(zip(values, targets), key=lambda item: item[0])
    sorted_values = [v for v, _ in pairs]
    sorted_targets = [t for _, t in pairs]

    vmin = sorted_values[0]
    vmax = sorted_values[-1]
    if math.isclose(vmin, vmax):
        return [vmin, vmax + 1.0]

    prefix_events = [0]
    for t in sorted_targets:
        prefix_events.append(prefix_events[-1] + int(t))

    segments: list[tuple[int, int]] = [(0, len(sorted_values))]
    thresholds: list[float] = []

    for _ in range(max(0, n_bins - 1)):
        best_gain = 0.0
        best_segment_idx: int | None = None
        best_split_idx: int | None = None

        for seg_idx, (start, end) in enumerate(segments):
            split_idx, gain = _best_tree_split(
                sorted_values, prefix_events, start=start, end=end
            )
            if split_idx is None:
                continue
            if gain > best_gain:
                best_gain = gain
                best_segment_idx = seg_idx
                best_split_idx = split_idx

        if best_segment_idx is None or best_split_idx is None:
            break

        start, end = segments.pop(best_segment_idx)
        split_idx = best_split_idx
        threshold = (sorted_values[split_idx] + sorted_values[split_idx + 1]) / 2.0
        thresholds.append(float(threshold))
        segments.append((start, split_idx + 1))
        segments.append((split_idx + 1, end))

    unique_edges = sorted(set([float(vmin), *thresholds, float(vmax)]))
    if len(unique_edges) < 2:
        unique_edges = [float(vmin), float(vmax)]
    if math.isclose(unique_edges[0], unique_edges[-1]):
        unique_edges[-1] = unique_edges[-1] + 1.0
    return unique_edges


def _compute_kmeans_bin_edges(values: list[float], n_bins: int, max_iter: int = 100) -> list[float]:
    np = _try_import_numpy()
    if np is None:
        raise RuntimeError("numpy is required for kmeans numerical binning.")

    arr = np.array(values, dtype=float)
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if math.isclose(vmin, vmax):
        return [vmin, vmax + 1.0]

    unique_values = np.unique(arr)
    k = int(min(n_bins, unique_values.size))
    if k <= 1:
        return [vmin, vmax]

    centers = np.quantile(arr, np.linspace(0.0, 1.0, k))
    centers = np.array(sorted(set(float(c) for c in centers)), dtype=float)
    if centers.size < 2:
        return [vmin, vmax]

    k = int(centers.size)
    for _ in range(max_iter):
        distances = np.abs(arr[:, None] - centers[None, :])
        labels = np.argmin(distances, axis=1)

        new_centers = centers.copy()
        for idx in range(k):
            mask = labels == idx
            if np.any(mask):
                new_centers[idx] = float(np.mean(arr[mask]))

        new_centers = np.array(sorted(new_centers.tolist()), dtype=float)
        if np.allclose(new_centers, centers, rtol=0.0, atol=1e-10):
            break
        centers = new_centers

    centers = np.array(sorted(set(float(c) for c in centers)), dtype=float)
    if centers.size < 2:
        return [vmin, vmax]

    midpoints = (centers[:-1] + centers[1:]) / 2.0
    edges = [vmin, *[float(v) for v in midpoints], vmax]
    unique_edges = sorted(set(float(v) for v in edges))
    if len(unique_edges) < 2:
        unique_edges = [vmin, vmax]
    if math.isclose(unique_edges[0], unique_edges[-1]):
        unique_edges[-1] = unique_edges[-1] + 1.0
    return unique_edges


def _enforce_monotonic_edges(
    values: list[float],
    targets: list[int],
    edges: list[float],
    *,
    direction: str,
) -> list[float]:
    if len(edges) <= 2 or not values:
        return edges

    bins: list[dict[str, float]] = []
    for i in range(len(edges) - 1):
        bins.append(
            {
                "lo": float(edges[i]),
                "hi": float(edges[i + 1]),
                "events": 0.0,
                "count": 0.0,
            }
        )

    for value, target in zip(values, targets):
        idx = _bin_index(value, edges)
        bins[idx]["events"] += float(target)
        bins[idx]["count"] += 1.0

    bins = _merge_empty_monotonic_bins(bins)
    if len(bins) <= 1:
        return [bins[0]["lo"], bins[0]["hi"]]

    i = 0
    while i < len(bins) - 1:
        left_rate = bins[i]["events"] / bins[i]["count"]
        right_rate = bins[i + 1]["events"] / bins[i + 1]["count"]
        violation = (
            left_rate > right_rate if direction == "increasing" else left_rate < right_rate
        )
        if not violation:
            i += 1
            continue

        bins[i]["hi"] = bins[i + 1]["hi"]
        bins[i]["events"] += bins[i + 1]["events"]
        bins[i]["count"] += bins[i + 1]["count"]
        del bins[i + 1]
        if i > 0:
            i -= 1

    out_edges = [bins[0]["lo"]]
    out_edges.extend(b["hi"] for b in bins)
    unique_edges = [float(v) for v in out_edges]
    if len(unique_edges) < 2:
        unique_edges = [unique_edges[0], unique_edges[0] + 1.0]
    if math.isclose(unique_edges[0], unique_edges[-1]):
        unique_edges[-1] = unique_edges[-1] + 1.0
    return unique_edges


def _merge_empty_monotonic_bins(bins: list[dict[str, float]]) -> list[dict[str, float]]:
    if not bins:
        return [{"lo": 0.0, "hi": 1.0, "events": 0.0, "count": 1.0}]

    i = 0
    while i < len(bins):
        if bins[i]["count"] > 0:
            i += 1
            continue
        if len(bins) == 1:
            bins[i]["count"] = 1.0
            break
        if i == 0:
            bins[1]["lo"] = bins[0]["lo"]
            del bins[0]
            continue
        bins[i - 1]["hi"] = bins[i]["hi"]
        del bins[i]
    return bins


def _best_tree_split(
    sorted_values: list[float],
    prefix_events: list[int],
    *,
    start: int,
    end: int,
) -> tuple[int | None, float]:
    total_count = end - start
    if total_count < 2:
        return None, 0.0

    total_events = prefix_events[end] - prefix_events[start]
    parent_impurity = _gini_impurity(total_events, total_count)

    best_gain = 0.0
    best_split_idx: int | None = None
    for split_idx in range(start, end - 1):
        if math.isclose(sorted_values[split_idx], sorted_values[split_idx + 1]):
            continue

        left_count = split_idx - start + 1
        right_count = total_count - left_count
        left_events = prefix_events[split_idx + 1] - prefix_events[start]
        right_events = total_events - left_events

        left_impurity = _gini_impurity(left_events, left_count)
        right_impurity = _gini_impurity(right_events, right_count)
        child_impurity = (
            (left_count / total_count) * left_impurity
            + (right_count / total_count) * right_impurity
        )
        gain = parent_impurity - child_impurity
        if gain > best_gain:
            best_gain = gain
            best_split_idx = split_idx

    return best_split_idx, best_gain


def _gini_impurity(events: int, count: int) -> float:
    if count <= 0:
        return 0.0
    p = events / count
    return 2.0 * p * (1.0 - p)


@lru_cache(maxsize=1)
def _try_import_pandas() -> ModuleType | None:
    try:
        return import_module("pandas")
    except ImportError:
        return None


@lru_cache(maxsize=1)
def _try_import_numpy() -> ModuleType | None:
    try:
        return import_module("numpy")
    except ImportError:
        return None
