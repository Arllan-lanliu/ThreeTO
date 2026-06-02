"""Compatibility shims for third-party Triton kernels used by GRKAN."""

from __future__ import annotations


def patch_triton_language() -> None:
    try:
        import triton
        import triton.language as tl
    except ImportError:
        return

    if hasattr(tl, "fma"):
        return

    @triton.jit
    def _fma(a, b, c):
        return a * b + c

    tl.fma = _fma
