"""Exact reference multiplication and factorization reconstruction.

All exact arithmetic is performed with :class:`fractions.Fraction` so that
basis-case correctness can be proven rather than approximated. Floating-point
fields use native ``float`` arithmetic and are intended for performance
measurement only.
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from .tensor import Factorization, Field, MatmulSpec


def _to_matrix(
    matrix: Sequence[Sequence[Any]], exact: bool
) -> list[list[Fraction]] | list[list[float]]:
    if exact:
        return [[Fraction(x) for x in row] for row in matrix]
    return [[float(x) for x in row] for row in matrix]


def reference_multiply(
    a: Sequence[Sequence[Any]],
    b: Sequence[Sequence[Any]],
    field: Field = Field.INTEGER,
) -> list[list[Any]]:
    """Schoolbook matrix multiplication ``C = A @ B``.

    Returns exact ``Fraction`` values for exact fields and ``float`` values
    otherwise. The caller is responsible for supplying ``A`` and ``B`` with
    compatible dimensions.
    """
    exact = field.exact
    a_m = _to_matrix(a, exact)
    b_m = _to_matrix(b, exact)
    m = len(a_m)
    k = len(b_m)
    n = len(b_m[0])
    c: list[list[Any]] = [[type(a_m[0][0])(0) for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = type(a_m[0][0])(0)
            for r in range(k):
                total += a_m[i][r] * b_m[r][j]
            c[i][j] = total
    return c


def reconstruct(
    factorization: Factorization,
    a: Sequence[Sequence[Any]],
    b: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    """Reconstruct ``C = A @ B`` from a bilinear :class:`Factorization`.

    Uses exact arithmetic for exact fields and ``float`` arithmetic otherwise.
    """
    exact = factorization.field.exact
    spec = factorization.spec
    a_m = _to_matrix(a, exact)
    b_m = _to_matrix(b, exact)
    zero = type(a_m[0][0])(0)
    c: list[list[Any]] = [[zero for _ in range(spec.n)] for _ in range(spec.m)]

    for factor in factorization.factors:
        su = _frobenius(factor.u, a_m, spec.m, spec.k, exact)
        sv = _frobenius(factor.v, b_m, spec.k, spec.n, exact)
        prod = su * sv
        for i in range(spec.m):
            w_row = factor.w[i]
            for j in range(spec.n):
                c[i][j] += prod * _coerce(w_row[j], exact)
    return c


def _frobenius(
    factor_matrix: Sequence[Sequence[Any]],
    input_matrix: list[list[Any]],
    rows: int,
    cols: int,
    exact: bool,
) -> Any:
    total = _coerce(0, exact)
    for i in range(rows):
        f_row = factor_matrix[i]
        x_row = input_matrix[i]
        for r in range(cols):
            total += _coerce(f_row[r], exact) * x_row[r]
    return total


def _coerce(value: Any, exact: bool) -> Any:
    if exact:
        return Fraction(value)
    return float(value)


def verify_exact(
    factorization: Factorization,
    a: Sequence[Sequence[Any]],
    b: Sequence[Sequence[Any]],
) -> bool:
    """Return True iff reconstruction exactly equals the reference product."""
    if not factorization.field.exact:
        raise ValueError("verify_exact requires an exact (integer/rational) field.")
    got = reconstruct(factorization, a, b)
    expected = reference_multiply(a, b, factorization.field)
    for i in range(factorization.spec.m):
        for j in range(factorization.spec.n):
            if Fraction(got[i][j]) != Fraction(expected[i][j]):
                return False
    return True


def random_cpu_inputs(
    spec: MatmulSpec,
    low: int = -3,
    high: int = 3,
    seed: int | None = None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Generate small random integer inputs for randomized correctness tests."""
    import random

    rng = random.Random(seed)
    a = [[rng.randint(low, high) for _ in range(spec.k)] for _ in range(spec.m)]
    b = [[rng.randint(low, high) for _ in range(spec.n)] for _ in range(spec.k)]
    return a, b
