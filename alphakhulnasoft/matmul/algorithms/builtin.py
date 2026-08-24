"""Built-in factorizations and algorithmic generators.

These are constructed programmatically so they stay exact and are easy to
audit. The trivial schoolbook factorization always has rank ``m * k * n`` and
serves as the correctness baseline. A Strassen-block generator composes the
rank-7 ``<2,2,2>`` algorithm into larger square multiplications.
"""

from __future__ import annotations

from typing import Any

from ..tensor import Factorization, Field, MatmulSpec, RankOneFactor


def schoolbook(spec: MatmulSpec, field: Field = Field.INTEGER) -> Factorization:
    """Return the naive rank-``m*k*n`` bilinear algorithm for ``spec``."""
    spec.validate()
    factors: list[RankOneFactor] = []
    for i in range(spec.m):
        for r in range(spec.k):
            for j in range(spec.n):
                u = _zero(spec.m, spec.k)
                v = _zero(spec.k, spec.n)
                w = _zero(spec.m, spec.n)
                u[i][r] = 1
                v[r][j] = 1
                w[i][j] = 1
                factors.append(
                    RankOneFactor(
                        u=tuple(tuple(row) for row in u),
                        v=tuple(tuple(row) for row in v),
                        w=tuple(tuple(row) for row in w),
                    )
                )
    return Factorization(
        spec=spec,
        field=field,
        algorithm_name=f"schoolbook_{spec.m}x{spec.k}x{spec.n}",
        factors=tuple(factors),
        provenance={
            "source": "generated",
            "generator": "schoolbook",
            "rank_theoretical": spec.m * spec.k * spec.n,
        },
    )


def strassen_2x2(field: Field = Field.INTEGER) -> Factorization:
    """Return Strassen's rank-7 algorithm for 2x2 matrix multiplication.

    Uses the classic 7-product decomposition over the integers.
    """
    spec = MatmulSpec(2, 2, 2)
    products = _strassen_products()
    factors = [_product_to_factor(*p) for p in products]
    return Factorization(
        spec=spec,
        field=field,
        algorithm_name="strassen_2x2",
        factors=tuple(factors),
        provenance={
            "source": "generated",
            "generator": "strassen",
            "reference": "Strassen 1969, rank 7",
        },
    )


def _strassen_products() -> list[tuple[list[list[int]], list[list[int]], list[list[int]]]]:
    """Return the 7 (u, v, w) rank-one terms of Strassen's algorithm.

    ``u`` (2x2) are linear forms in A, ``v`` (2x2) linear forms in B, and
    ``w`` (2x2) gives each product's exact contribution to C. The
    reconstruction ``C[i][j] = sum_t (u_t.A)(v_t.B) w_t[i][j]`` then yields
    C11 = P1 + P4 - P5 + P7, C12 = P3 + P5, C21 = P2 + P4,
    C22 = P1 - P2 + P3 + P6, which is exactly A @ B.
    """

    def m(a11: int, a12: int, a21: int, a22: int) -> list[list[int]]:
        return [[a11, a12], [a21, a22]]

    # P1 = (A11 + A22)(B11 + B22)
    u1 = m(1, 0, 0, 1)
    v1 = m(1, 0, 0, 1)
    w1 = m(1, 0, 0, 1)  # +C11, +C22

    # P2 = (A21 + A22) B11 ; contributes +C21, -C22
    u2 = m(0, 0, 1, 1)
    v2 = m(1, 0, 0, 0)
    w2 = m(0, 0, 1, -1)

    # P3 = A11 (B12 - B22) ; contributes +C12, +C22
    u3 = m(1, 0, 0, 0)
    v3 = m(0, 1, 0, -1)
    w3 = m(0, 1, 0, 1)

    # P4 = A22 (B21 - B11)
    u4 = m(0, 0, 0, 1)
    v4 = m(-1, 0, 1, 0)
    w4 = m(1, 0, 1, 0)  # +C11, +C21

    # P5 = (A11 + A12) B22
    u5 = m(1, 1, 0, 0)
    v5 = m(0, 0, 0, 1)
    w5 = m(-1, 1, 0, 0)  # -C11, +C12

    # P6 = (A21 - A11)(B11 + B12)
    u6 = m(-1, 0, 1, 0)
    v6 = m(1, 1, 0, 0)
    w6 = m(0, 0, 0, 1)  # +C22

    # P7 = (A12 - A22)(B21 + B22)
    u7 = m(0, 1, 0, -1)
    v7 = m(0, 0, 1, 1)
    w7 = m(1, 0, 0, 0)  # +C11

    return [
        (u1, v1, w1),
        (u2, v2, w2),
        (u3, v3, w3),
        (u4, v4, w4),
        (u5, v5, w5),
        (u6, v6, w6),
        (u7, v7, w7),
    ]


def _product_to_factor(u: list[list[int]], v: list[list[int]], w: list[list[int]]) -> RankOneFactor:
    return RankOneFactor(
        u=tuple(tuple(row) for row in u),
        v=tuple(tuple(row) for row in v),
        w=tuple(tuple(row) for row in w),
    )


def strassen_block_4x4(field: Field = Field.INTEGER) -> Factorization:
    """Return a 4x4 factorization built recursively from 2x2 Strassen (rank 49)."""
    from ..recombination import compose_strassen

    return compose_strassen(strassen_2x2(field), algorithm_name="strassen_block_4x4")


def _zero(rows: int, cols: int) -> list[list[int]]:
    return [[0 for _ in range(cols)] for _ in range(rows)]


def verify_builtin() -> dict[str, bool]:
    """Sanity check that builtin algos reconstruct schoolbook exactly."""
    results: dict[str, bool] = {}
    for spec in (MatmulSpec(2, 2, 2), MatmulSpec(3, 2, 4), MatmulSpec(1, 3, 2)):
        alg = schoolbook(spec)
        a = [[(i * 2 + r) % 5 for r in range(spec.k)] for i in range(spec.m)]
        b = [[(r * 3 + j) % 7 for j in range(spec.n)] for r in range(spec.k)]
        results[f"schoolbook_{spec.dims}"] = _exact_equal(alg, a, b)
    s2 = strassen_2x2()
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    results["strassen_2x2"] = _exact_equal(s2, a, b)
    return results


def _exact_equal(alg: Factorization, a: Any, b: Any) -> bool:
    from ..reference import verify_exact

    return verify_exact(alg, a, b)
