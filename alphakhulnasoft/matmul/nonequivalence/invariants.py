"""Cheap invariants used to separate algorithms before any search.

Invariants are preserved under the supported equivalence transformations
(factor permutation, per-factor scaling, and input/output relabelling):
rank, total number of nonzeros, and the multiset of per-factor support sizes.
None of these *proves* equivalence, but a mismatch *proves* nonequivalence.
"""

from __future__ import annotations

from typing import Any

from ..tensor import Factorization


def _nonzeros(matrix: tuple[tuple[Any, ...], ...]) -> int:
    return sum(1 for row in matrix for x in row if x != 0)


def rank(factorization: Factorization) -> int:
    """Algorithmic rank (number of factors)."""
    return factorization.rank


def total_nonzeros(factorization: Factorization) -> int:
    """Total number of nonzero coefficients across all factors."""
    return sum(_nonzeros(f.u) + _nonzeros(f.v) + _nonzeros(f.w) for f in factorization.factors)


def support_signature(factorization: Factorization) -> tuple[tuple[int, int, int], ...]:
    """Sorted multiset of ``(nnz_u, nnz_v, nnz_w)`` per factor.

    Invariant under factor permutation and per-factor scaling (scaling does
    not change which entries are zero). Use as a coarse nonequivalence test.
    """
    sig = [(_nonzeros(f.u), _nonzeros(f.v), _nonzeros(f.w)) for f in factorization.factors]
    return tuple(sorted(sig))


def invariants(factorization: Factorization) -> dict[str, Any]:
    """Collect all invariants for a factorization into one dict."""
    return {
        "dims": factorization.dims,
        "field": factorization.field.value,
        "rank": rank(factorization),
        "total_nonzeros": total_nonzeros(factorization),
        "support_signature": support_signature(factorization),
    }


def separates(a: Factorization, b: Factorization) -> str | None:
    """Return the name of an invariant that proves ``a`` != ``b``, else None."""
    if a.dims != b.dims:
        return "dimensions"
    if a.field != b.field:
        return "field"
    if rank(a) != rank(b):
        return "rank"
    if total_nonzeros(a) != total_nonzeros(b):
        return "total_nonzeros"
    if support_signature(a) != support_signature(b):
        return "support_signature"
    return None
