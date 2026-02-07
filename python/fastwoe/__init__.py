"""Python package namespace for FastWOE Rust bindings."""

from .fastwoe_rs import compute_binary_woe_py
from .model import FastWoe
from .preprocessor import WoePreprocessor

try:
    from .fastwoe_rs import RustPreprocessor
except Exception:  # pragma: no cover - depends on extension build.
    RustPreprocessor = None

try:
    from .fastwoe_rs import RustNumericBinner
except Exception:  # pragma: no cover - depends on extension build.
    RustNumericBinner = None

__all__ = [
    "FastWoe",
    "WoePreprocessor",
    "compute_binary_woe_py",
    "RustPreprocessor",
    "RustNumericBinner",
]
