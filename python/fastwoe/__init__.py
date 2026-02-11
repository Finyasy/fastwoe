"""Python package namespace for FastWOE Rust bindings."""

from .fastwoe_rs import compute_binary_woe_py
from .model import FastWoe
from .preprocessor import WoePreprocessor


def _patch_row_string_representations() -> None:
    """Apply safe display patches for row-like PyO3 classes.

    This keeps notebook output readable even if a user is running an older
    compiled extension build that predates representation fixes.
    """
    try:
        from .fastwoe_rs import IvRow
    except Exception:  # pragma: no cover - depends on extension build.
        return

    def _iv_row_str(self) -> str:
        return (
            f"feature={self.feature} | iv={self.iv:.6f} | se={self.iv_se:.6f} | "
            f"CI=[{self.iv_ci_lower:.6f}, {self.iv_ci_upper:.6f}] | "
            f"sig={self.iv_significance}"
        )

    IvRow.__str__ = _iv_row_str


_patch_row_string_representations()

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
