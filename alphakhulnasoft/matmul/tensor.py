"""Immutable data structures for matrix-multiplication tensors and factorizations.

A matrix multiplication tensor describes the bilinear map

    C = A @ B,   A in F^{m x k},  B in F^{k x n},  C in F^{m x n}.

A *factorization* (a bilinear algorithm) is a collection of rank-one terms
that reproduces this bilinear map. Each rank-one term is a triple of factor
matrices ``(u, v, w)`` with shapes ``(m, k)``, ``(k, n)`` and ``(m, n)``
respectively. The reconstructed product is

    C[i][j] = sum_t ( <u_t, A> * <v_t, B> ) * w_t[i][j]

where ``<.>`` is the Frobenius inner product over all matrix entries. The
number of terms is the *rank* of the algorithm.

Exact validation uses integer/rational coefficients (via :class:`fractions.Fraction`);
floating-point fields exist only for performance measurement and are never
treated as exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

FORMAT_VERSION = "1.0"


class Field(str, Enum):
    """Numeric field / dtype a factorization is defined over."""

    INTEGER = "integer"
    RATIONAL = "rational"
    FLOAT32 = "float32"
    FLOAT64 = "float64"

    @property
    def exact(self) -> bool:
        """Whether arithmetic in this field is exact (no rounding)."""
        return self in (Field.INTEGER, Field.RATIONAL)


def _as_fractions(matrix: Sequence[Sequence[Any]]) -> list[list[Fraction]]:
    """Convert a nested sequence of numbers into exact Fractions."""
    return [[Fraction(x) for x in row] for row in matrix]


@dataclass(frozen=True)
class MatmulSpec:
    """Dimensions of a matrix multiplication problem ``C = A @ B``."""

    m: int
    k: int
    n: int

    def __post_init__(self) -> None:
        if self.m <= 0 or self.k <= 0 or self.n <= 0:
            raise ValueError(
                f"All dimensions must be positive, got (m={self.m}, k={self.k}, n={self.n})"
            )

    @property
    def dims(self) -> tuple[int, int, int]:
        return (self.m, self.k, self.n)

    def validate(self) -> None:
        """Raise if the spec is not well-formed (idempotent check)."""
        if self.m <= 0 or self.k <= 0 or self.n <= 0:
            raise ValueError("Matrix dimensions must be positive integers.")


@dataclass(frozen=True)
class RankOneFactor:
    """A single rank-one term ``(u, v, w)`` of a bilinear algorithm.

    ``u`` has shape ``(m, k)`` and is contracted with ``A``.
    ``v`` has shape ``(k, n)`` and is contracted with ``B``.
    ``w`` has shape ``(m, n)`` and scales the resulting scalar product into ``C``.
    """

    u: tuple[tuple[Any, ...], ...]
    v: tuple[tuple[Any, ...], ...]
    w: tuple[tuple[Any, ...], ...]

    def validate(self, spec: MatmulSpec) -> None:
        """Validate shapes against the owning :class:`MatmulSpec`."""
        _require_shape(self.u, (spec.m, spec.k), "u")
        _require_shape(self.v, (spec.k, spec.n), "v")
        _require_shape(self.w, (spec.m, spec.n), "w")


@dataclass(frozen=True)
class Factorization:
    """An immutable bilinear algorithm for a matrix multiplication problem."""

    spec: MatmulSpec
    field: Field
    algorithm_name: str
    factors: tuple[RankOneFactor, ...]
    provenance: dict[str, Any]
    format_version: str = FORMAT_VERSION

    def __post_init__(self) -> None:
        self.spec.validate()
        if not self.factors:
            raise ValueError("A factorization must contain at least one factor.")
        for _idx, factor in enumerate(self.factors):
            factor.validate(self.spec)

    @property
    def rank(self) -> int:
        """Number of rank-one terms (algorithmic rank)."""
        return len(self.factors)

    @property
    def dims(self) -> tuple[int, int, int]:
        return self.spec.dims

    def validate(self) -> None:
        """Validate spec, field, and every factor shape."""
        self.spec.validate()
        if not isinstance(self.field, Field):
            raise ValueError(f"field must be a Field, got {type(self.field)!r}")
        if self.rank < 1:
            raise ValueError("factorization rank must be >= 1")
        for factor in self.factors:
            factor.validate(self.spec)


def _require_shape(matrix: Any, expected: tuple[int, int], name: str) -> None:
    """Validate that ``matrix`` is a 2D nested sequence of shape ``expected``."""
    if not isinstance(matrix, (tuple, list)):
        raise ValueError(f"factor '{name}' must be a 2D sequence, got {type(matrix)!r}")
    if len(matrix) != expected[0]:
        raise ValueError(f"factor '{name}' has {len(matrix)} rows, expected {expected[0]}")
    for row in matrix:
        if not isinstance(row, (tuple, list)):
            raise ValueError(f"factor '{name}' rows must be sequences")
        if len(row) != expected[1]:
            raise ValueError(f"factor '{name}' row has length {len(row)}, expected {expected[1]}")
