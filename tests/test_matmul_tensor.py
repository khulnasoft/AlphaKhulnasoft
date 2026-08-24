"""Tests for the matrix-multiplication tensor contract and exactness."""

from fractions import Fraction

import pytest

from alphakhulnasoft.matmul.algorithms.builtin import schoolbook, strassen_2x2
from alphakhulnasoft.matmul.reference import (
    random_cpu_inputs,
    reconstruct,
    reference_multiply,
    verify_exact,
)
from alphakhulnasoft.matmul.tensor import (
    Factorization,
    Field,
    MatmulSpec,
    RankOneFactor,
)


def test_invalid_dimensions_rejected():
    with pytest.raises(ValueError):
        MatmulSpec(0, 2, 2)
    with pytest.raises(ValueError):
        MatmulSpec(2, -1, 2)


def test_rank_one_factor_shape_validation():
    spec = MatmulSpec(2, 2, 2)
    with pytest.raises(ValueError):
        RankOneFactor(
            u=((1, 0, 0), (0, 1, 0)),  # wrong cols
            v=((1, 0), (0, 1)),
            w=((1, 0), (0, 1)),
        ).validate(spec)


def test_factorization_requires_factors():
    spec = MatmulSpec(2, 2, 2)
    with pytest.raises(ValueError):
        Factorization(spec, Field.INTEGER, "empty", (), {})


def test_reference_multiply_basis():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    c = reference_multiply(a, b, Field.INTEGER)
    assert c == [[19, 22], [43, 50]]


def test_reconstruct_schoolbook_exact():
    spec = MatmulSpec(3, 2, 4)
    alg = schoolbook(spec)
    a = [[(i * 2 + r) % 5 for r in range(spec.k)] for i in range(spec.m)]
    b = [[(r * 3 + j) % 7 for j in range(spec.n)] for r in range(spec.k)]
    assert verify_exact(alg, a, b)


def test_reconstruct_strassen_exact():
    alg = strassen_2x2()
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    assert verify_exact(alg, a, b)


def test_randomized_cpu_correctness():
    for m, k, n in [(1, 1, 1), (2, 3, 4), (3, 3, 3), (4, 2, 5)]:
        spec = MatmulSpec(m, k, n)
        alg = schoolbook(spec)
        a, b = random_cpu_inputs(spec, seed=42)
        assert verify_exact(alg, a, b)


def test_reconstruct_returns_fractions_for_exact_field():
    alg = schoolbook(MatmulSpec(2, 2, 2))
    c = reconstruct(alg, [[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert all(isinstance(c[i][j], Fraction) for i in range(2) for j in range(2))
