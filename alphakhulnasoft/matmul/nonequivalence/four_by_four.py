"""Named 4x4 nonequivalent matrix-multiplication algorithms.

These are immutable fixtures used to demonstrate the nonequivalence verifier.
The two algorithms compute the same bilinear map (4x4 matrix multiplication)
but have different ranks, which the rank invariant proves nonequivalent:

* ``schoolbook_4x4`` -- the trivial algorithm, rank 64.
* ``strassen_block_4x4`` -- built from 2x2 Strassen blocks, rank 49.

Both reconstruct 4x4 products exactly; they differ only in multiplicative
cost, which is exactly the kind of distinction the verifier is meant to catch.
"""

from __future__ import annotations

from ..algorithms.builtin import schoolbook, strassen_block_4x4
from ..tensor import Factorization, MatmulSpec


def schoolbook_4x4() -> Factorization:
    """Trivial rank-64 4x4 algorithm."""
    return schoolbook(MatmulSpec(4, 4, 4))


def strassen_4x4() -> Factorization:
    """Rank-49 4x4 algorithm built from Strassen 2x2 blocks."""
    return strassen_block_4x4()


def four_by_four_fixtures() -> dict[str, Factorization]:
    """Return the named 4x4 fixtures."""
    return {
        "schoolbook_4x4": schoolbook_4x4(),
        "strassen_block_4x4": strassen_4x4(),
    }
