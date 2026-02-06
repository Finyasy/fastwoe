"""Python package namespace for FastWOE Rust bindings."""

from .fastwoe_rs import FastWoe, compute_binary_woe_py

__all__ = ["FastWoe", "compute_binary_woe_py"]
