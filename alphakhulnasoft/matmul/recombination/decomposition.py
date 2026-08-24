"""Reverse metadata for recombined factorizations.

Given a factorization produced by :func:`compose_kronecker`, this module
recovers the block decomposition and can verify that re-composing the base
algorithm reproduces the original exactly. Provenance is preserved across
serialization because it is stored inside the factorization payload.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..tensor import Factorization
from .compose import Composition, compose_kronecker, compose_strassen


@dataclass(frozen=True)
class Decomposition:
    """Inspectable description of how a tensor was assembled."""

    base_algorithm: str
    tiling: int
    base_rank: int
    components: list[Composition]
    total_factors: int


def describe_decomposition(factorization: Factorization) -> Decomposition:
    """Recover block decomposition metadata from a composed factorization."""
    prov = factorization.provenance
    if prov.get("source") != "composed" or prov.get("composition") not in (
        "kronecker",
        "strassen",
    ):
        raise ValueError("Factorization was not produced by compose_kronecker/compose_strassen.")
    components = [Composition(**c) for c in prov["components"]]
    return Decomposition(
        base_algorithm=str(prov["base_algorithm"]),
        tiling=int(prov.get("tiling", 0)),
        base_rank=int(prov["base_rank"]),
        components=components,
        total_factors=len(components),
    )


def verify_decomposition(factorization: Factorization, base: Factorization) -> bool:
    """Return True iff re-composing ``base`` reproduces ``factorization``.

    Compares dimensions, field, rank, and the recorded block composition
    exactly (a structural, not numerical, proof of equivalence of assembly).
    """
    decomposition = describe_decomposition(factorization)
    if decomposition.base_algorithm != base.algorithm_name:
        return False
    if decomposition.tiling > 0:
        rebuilt = compose_kronecker(base, tiling=decomposition.tiling)
    else:
        rebuilt = compose_strassen(base)
    if rebuilt.spec != factorization.spec:
        return False
    if rebuilt.rank != factorization.rank:
        return False
    return rebuilt.field == factorization.field
