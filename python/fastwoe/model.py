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

    def transform(self, categories: Any, as_frame: bool = False) -> Any:
        values = self._inner.transform(_to_1d_str(categories))
        if as_frame:
            return _to_frame(values, columns=["woe"], index=_extract_index(categories))
        return values

    def fit_transform(self, categories: Any, targets: Any, as_frame: bool = False) -> Any:
        values = self._inner.fit_transform(_to_1d_str(categories), _to_u8(targets))
        if as_frame:
            return _to_frame(values, columns=["woe"], index=_extract_index(categories))
        return values

    def predict_proba(self, categories: Any) -> list[float]:
        return self._inner.predict_proba(_to_1d_str(categories))

    def predict_ci(self, categories: Any, alpha: float = 0.05, as_frame: bool = False) -> Any:
        values = self._inner.predict_ci(_to_1d_str(categories), alpha)
        if as_frame:
            return _to_ci_frame(values, index=_extract_index(categories))
        return values

    def get_mapping(self) -> list[Any]:
        return self._inner.get_mapping()

    def fit_matrix(self, rows: Any, targets: Any, feature_names: Any = None) -> None:
        self._inner.fit_matrix(_to_2d_str(rows), _to_u8(targets), _to_feature_names(feature_names))

    def transform_matrix(self, rows: Any, as_frame: bool = False) -> Any:
        values = self._inner.transform_matrix(_to_2d_str(rows))
        if as_frame:
            return _to_frame(values, columns=self.get_feature_names(), index=_extract_index(rows))
        return values

    def fit_transform_matrix(
        self, rows: Any, targets: Any, feature_names: Any = None, as_frame: bool = False
    ) -> Any:
        values = self._inner.fit_transform_matrix(
            _to_2d_str(rows), _to_u8(targets), _to_feature_names(feature_names)
        )
        if as_frame:
            return _to_frame(values, columns=self.get_feature_names(), index=_extract_index(rows))
        return values

    def predict_proba_matrix(self, rows: Any) -> list[float]:
        return self._inner.predict_proba_matrix(_to_2d_str(rows))

    def predict_ci_matrix(self, rows: Any, alpha: float = 0.05, as_frame: bool = False) -> Any:
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
        self._inner.fit_multiclass(_to_1d_str(categories), _to_1d_str(class_labels))

    def predict_proba_multiclass(self, categories: Any, as_frame: bool = False) -> Any:
        values = self._inner.predict_proba_multiclass(_to_1d_str(categories))
        if as_frame:
            cols = [f"proba_{c}" for c in self.get_class_labels()]
            return _to_frame(values, columns=cols, index=_extract_index(categories))
        return values

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

    def predict_proba_matrix_multiclass(self, rows: Any, as_frame: bool = False) -> Any:
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
        return self._inner.predict_ci_matrix_multiclass(_to_2d_str(rows), alpha)

    def predict_proba_matrix_class(self, rows: Any, class_label: Any) -> list[float]:
        return self._inner.predict_proba_matrix_class(_to_2d_str(rows), str(class_label))

    def predict_ci_matrix_class(
        self, rows: Any, class_label: Any, alpha: float = 0.05
    ) -> list[tuple[float, float, float]]:
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
