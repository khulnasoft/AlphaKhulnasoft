"""Recombination of smaller matrix-multiplication factorizations."""

from __future__ import annotations

from .compose import Composition, compose_kronecker, compose_strassen, operation_count
from .decomposition import Decomposition, describe_decomposition, verify_decomposition

__all__ = [
    "Composition",
    "compose_kronecker",
    "compose_strassen",
    "operation_count",
    "Decomposition",
    "describe_decomposition",
    "verify_decomposition",
]
