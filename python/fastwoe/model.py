"""High-level Python wrapper with NumPy/pandas-friendly input handling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .fastwoe_rs import FastWoe as _RustFastWoe


class FastWoe:
    """Python-friendly wrapper around the Rust-backed FastWoe implementation."""

    def __init__(self, smoothing: float = 0.5, default_woe: float = 0.0) -> None:
        self._inner = _RustFastWoe(smoothing=smoothing, default_woe=default_woe)

    def fit(self, categories: Any, targets: Any) -> None:
        self._inner.fit(_to_1d_str(categories), _to_u8(targets))

    def transform(self, categories: Any) -> list[float]:
        return self._inner.transform(_to_1d_str(categories))

    def fit_transform(self, categories: Any, targets: Any) -> list[float]:
        return self._inner.fit_transform(_to_1d_str(categories), _to_u8(targets))

    def predict_proba(self, categories: Any) -> list[float]:
        return self._inner.predict_proba(_to_1d_str(categories))

    def predict_ci(self, categories: Any, alpha: float = 0.05) -> list[tuple[float, float, float]]:
        return self._inner.predict_ci(_to_1d_str(categories), alpha)

    def get_mapping(self) -> list[Any]:
        return self._inner.get_mapping()

    def fit_matrix(self, rows: Any, targets: Any, feature_names: Any = None) -> None:
        self._inner.fit_matrix(_to_2d_str(rows), _to_u8(targets), _to_feature_names(feature_names))

    def transform_matrix(self, rows: Any) -> list[list[float]]:
        return self._inner.transform_matrix(_to_2d_str(rows))

    def fit_transform_matrix(
        self, rows: Any, targets: Any, feature_names: Any = None
    ) -> list[list[float]]:
        return self._inner.fit_transform_matrix(
            _to_2d_str(rows), _to_u8(targets), _to_feature_names(feature_names)
        )

    def predict_proba_matrix(self, rows: Any) -> list[float]:
        return self._inner.predict_proba_matrix(_to_2d_str(rows))

    def predict_ci_matrix(self, rows: Any, alpha: float = 0.05) -> list[tuple[float, float, float]]:
        return self._inner.predict_ci_matrix(_to_2d_str(rows), alpha)

    def get_feature_names(self) -> list[str]:
        return self._inner.get_feature_names()

    def get_feature_mapping(self, feature_name: str) -> list[Any]:
        return self._inner.get_feature_mapping(str(feature_name))

    def fit_multiclass(self, categories: Any, class_labels: Any) -> None:
        self._inner.fit_multiclass(_to_1d_str(categories), _to_1d_str(class_labels))

    def predict_proba_multiclass(self, categories: Any) -> list[list[float]]:
        return self._inner.predict_proba_multiclass(_to_1d_str(categories))

    def predict_ci_multiclass(
        self, categories: Any, alpha: float = 0.05
    ) -> list[list[tuple[float, float, float]]]:
        return self._inner.predict_ci_multiclass(_to_1d_str(categories), alpha)

    def predict_proba_class(self, categories: Any, class_label: Any) -> list[float]:
        return self._inner.predict_proba_class(_to_1d_str(categories), str(class_label))

    def predict_ci_class(
        self, categories: Any, class_label: Any, alpha: float = 0.05
    ) -> list[tuple[float, float, float]]:
        return self._inner.predict_ci_class(_to_1d_str(categories), str(class_label), alpha)

    def get_mapping_multiclass(self, class_label: Any) -> list[Any]:
        return self._inner.get_mapping_multiclass(str(class_label))

    def fit_matrix_multiclass(
        self, rows: Any, class_labels: Any, feature_names: Any = None
    ) -> None:
        self._inner.fit_matrix_multiclass(
            _to_2d_str(rows), _to_1d_str(class_labels), _to_feature_names(feature_names)
        )

    def predict_proba_matrix_multiclass(self, rows: Any) -> list[list[float]]:
        return self._inner.predict_proba_matrix_multiclass(_to_2d_str(rows))

    def predict_ci_matrix_multiclass(
        self, rows: Any, alpha: float = 0.05
    ) -> list[list[tuple[float, float, float]]]:
        return self._inner.predict_ci_matrix_multiclass(_to_2d_str(rows), alpha)

    def predict_proba_matrix_class(self, rows: Any, class_label: Any) -> list[float]:
        return self._inner.predict_proba_matrix_class(_to_2d_str(rows), str(class_label))

    def predict_ci_matrix_class(
        self, rows: Any, class_label: Any, alpha: float = 0.05
    ) -> list[tuple[float, float, float]]:
        return self._inner.predict_ci_matrix_class(_to_2d_str(rows), str(class_label), alpha)

    def get_class_labels(self) -> list[str]:
        return self._inner.get_class_labels()

    def get_feature_mapping_multiclass(self, class_label: Any, feature_name: Any) -> list[Any]:
        return self._inner.get_feature_mapping_multiclass(str(class_label), str(feature_name))


def _to_feature_names(value: Any) -> list[str] | None:
    if value is None:
        return None
    return [str(v) for v in _to_1d_list(value)]


def _to_u8(value: Any) -> list[int]:
    return [int(v) for v in _to_1d_list(value)]


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


def _is_pandas_frame(value: Any) -> bool:
    return hasattr(value, "values") and hasattr(value, "columns") and hasattr(value, "dtypes")

