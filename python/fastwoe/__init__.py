"""Python package namespace for FastWOE Rust bindings."""

from .fastwoe_rs import compute_binary_woe_py
from .model import FastWoe
from .preprocessor import WoePreprocessor

__all__ = ["FastWoe", "WoePreprocessor", "compute_binary_woe_py"]
