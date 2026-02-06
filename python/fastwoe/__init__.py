"""Python package namespace for FastWOE Rust bindings."""

from .fastwoe_rs import compute_binary_woe_py
from .model import FastWoe

__all__ = ["FastWoe", "compute_binary_woe_py"]
