"""Recombination of smaller matrix-multiplication factorizations.

The core operation is a Kronecker-style block composition: a base algorithm
for ``(m, k, n)`` is tiled ``t x t x t`` times to produce an algorithm for
``(t*m, t*k, t*n)`` whose rank is ``t**3 * base.rank``. Each resulting factor
records which base factor and which block ``(p, q, s)`` it came from so the
decomposition is inspectable and reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..tensor import Factorization, MatmulSpec, RankOneFactor


@dataclass(frozen=True)
class Composition:
    """Metadata describing how a factor was assembled from a base algorithm."""

    block_p: int
    block_q: int
    block_s: int
    base_factor_index: int
    base_algorithm: str


def compose_strassen(base: Factorization, algorithm_name: str | None = None) -> Factorization:
    """Recursively build a ``(2s, 2s, 2s)`` Strassen algorithm from a square base.

    The base algorithm (size ``s x s x s``, rank ``r``) is applied inside each
    of the 7 Strassen block products, yielding a rank-``7r`` algorithm for size
    ``2s``. For the rank-7 2x2 Strassen base this produces the classic rank-49
    4x4 Strassen algorithm.
    """
    base.validate()
    if not (base.spec.m == base.spec.k == base.spec.n):
        raise ValueError("compose_strassen requires a square base algorithm.")
    s = base.spec.m
    big = 2 * s
    spec = MatmulSpec(big, big, big)

    # The 7 Strassen block linear forms (identical in shape to a 2x2 algorithm).
    seven = _strassen_block_forms()

    factors: list[RankOneFactor] = []
    composition: list[Composition] = []
    for t, (u_block, v_block, w_block) in enumerate(seven):
        for idx, factor in enumerate(base.factors):
            u = _embed(s, u_block, factor.u, dim_rows=s, dim_cols=s)
            v = _embed(s, v_block, factor.v, dim_rows=s, dim_cols=s)
            w = _embed(s, w_block, factor.w, dim_rows=s, dim_cols=s)
            factors.append(
                RankOneFactor(
                    u=tuple(tuple(row) for row in u),
                    v=tuple(tuple(row) for row in v),
                    w=tuple(tuple(row) for row in w),
                )
            )
            composition.append(Composition(t, t, t, idx, base.algorithm_name))

    provenance = {
        "source": "composed",
        "composition": "strassen",
        "base_algorithm": base.algorithm_name,
        "base_rank": base.rank,
        "components": [c.__dict__ for c in composition],
    }
    return Factorization(
        spec=spec,
        field=base.field,
        algorithm_name=algorithm_name or f"strassen_{base.algorithm_name}",
        factors=tuple(factors),
        provenance=provenance,
    )


def _strassen_block_forms() -> list[tuple[list[list[int]], list[list[int]], list[list[int]]]]:
    """Return the 7 block-level (u, v, w) forms of 2x2 Strassen."""

    def m(a11: int, a12: int, a21: int, a22: int) -> list[list[int]]:
        return [[a11, a12], [a21, a22]]

    u1 = m(1, 0, 0, 1)
    v1 = m(1, 0, 0, 1)
    w1 = m(1, 0, 0, 1)
    u2 = m(0, 0, 1, 1)
    v2 = m(1, 0, 0, 0)
    w2 = m(0, 0, 1, -1)
    u3 = m(1, 0, 0, 0)
    v3 = m(0, 1, 0, -1)
    w3 = m(0, 1, 0, 1)
    u4 = m(0, 0, 0, 1)
    v4 = m(-1, 0, 1, 0)
    w4 = m(1, 0, 1, 0)
    u5 = m(1, 1, 0, 0)
    v5 = m(0, 0, 0, 1)
    w5 = m(-1, 1, 0, 0)
    u6 = m(-1, 0, 1, 0)
    v6 = m(1, 1, 0, 0)
    w6 = m(0, 0, 0, 1)
    u7 = m(0, 1, 0, -1)
    v7 = m(0, 0, 1, 1)
    w7 = m(1, 0, 0, 0)
    return [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7),
    ]


def _embed(
    s: int,
    block_form: list[list[int]],
    inner: tuple[tuple[Any, ...], ...],
    dim_rows: int,
    dim_cols: int,
) -> list[list[int]]:
    """Embed ``inner`` (s x s) into each 2x2 block of a (2s x 2s) matrix.

    ``block_form[rb][cb]`` scales the block placed at output rows
    ``rb*s:(rb+1)s`` and cols ``cb*s:(cb+1)s``.
    """
    big = [[0 for _ in range(2 * s)] for _ in range(2 * s)]
    for rb in range(2):
        for cb in range(2):
            coeff = block_form[rb][cb]
            if coeff == 0:
                continue
            for i in range(s):
                for j in range(s):
                    big[rb * s + i][cb * s + j] = int(coeff) * int(inner[i][j])
    return big


def compose_kronecker(
    base: Factorization,
    tiling: int,
    algorithm_name: str | None = None,
) -> Factorization:
    """Tile ``base`` ``tiling x tiling x tiling`` times into one big algorithm.

    The result computes ``(t*m) x (t*k)`` by ``(t*k) x (t*n)`` matrix
    multiplication using ``t**3 * base.rank`` rank-one terms.
    """
    if tiling <= 0:
        raise ValueError("tiling must be a positive integer.")
    base.validate()
    m, k, n = base.spec.dims
    t = tiling
    big_m, big_k, big_n = t * m, t * k, t * n
    spec = MatmulSpec(big_m, big_k, big_n)

    factors: list[RankOneFactor] = []
    composition: list[Composition] = []
    for p in range(t):
        for q in range(t):
            for s in range(t):
                for idx, factor in enumerate(base.factors):
                    u = _place(base.spec.m, base.spec.k, factor.u, p * m, s * k, big_m, big_k)
                    v = _place(base.spec.k, base.spec.n, factor.v, s * k, q * n, big_k, big_n)
                    w = _place(base.spec.m, base.spec.n, factor.w, p * m, q * n, big_m, big_n)
                    factors.append(
                        RankOneFactor(
                            u=tuple(tuple(row) for row in u),
                            v=tuple(tuple(row) for row in v),
                            w=tuple(tuple(row) for row in w),
                        )
                    )
                    composition.append(Composition(p, q, s, idx, base.algorithm_name))

    provenance = {
        "source": "composed",
        "composition": "kronecker",
        "base_algorithm": base.algorithm_name,
        "tiling": t,
        "base_rank": base.rank,
        "components": [c.__dict__ for c in composition],
    }
    return Factorization(
        spec=spec,
        field=base.field,
        algorithm_name=algorithm_name or f"kronecker_{base.algorithm_name}_t{t}",
        factors=tuple(factors),
        provenance=provenance,
    )


def _place(
    rows: int,
    cols: int,
    block: tuple[tuple[Any, ...], ...],
    row_off: int,
    col_off: int,
    big_rows: int,
    big_cols: int,
) -> list[list[int]]:
    """Embed ``block`` at an offset inside a zero matrix of the big shape."""
    big = [[0 for _ in range(big_cols)] for _ in range(big_rows)]
    for i in range(rows):
        for j in range(cols):
            big[row_off + i][col_off + j] = int(block[i][j])
    return big


def operation_count(factorization: Factorization) -> dict[str, float]:
    """Return multiplicative cost metadata for a factorization."""
    return {
        "rank": factorization.rank,
        "multiplications": factorization.rank,
        "dimensions": factorization.spec.m * factorization.spec.k * factorization.spec.n,
        "schoolbook_rank": factorization.spec.m * factorization.spec.k * factorization.spec.n,
        "speedup_vs_schoolbook": (
            factorization.spec.m * factorization.spec.k * factorization.spec.n
        )
        / factorization.rank,
    }
