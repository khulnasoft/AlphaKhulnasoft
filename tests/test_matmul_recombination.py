"""Tests for recombination of smaller factorizations."""

import pytest

import alphakhulnasoft.matmul as mm
from alphakhulnasoft.matmul.algorithms.builtin import schoolbook, strassen_2x2
from alphakhulnasoft.matmul.recombination.compose import compose_kronecker, compose_strassen
from alphakhulnasoft.matmul.reference import random_cpu_inputs, verify_exact
from alphakhulnasoft.matmul.tensor import MatmulSpec


def test_compose_kronecker_dimensions_and_rank():
    base = strassen_2x2()
    big = compose_kronecker(base, tiling=2)
    assert big.dims == (4, 4, 4)
    assert big.rank == 8 * base.rank


def test_compose_strassen_rank_49():
    big = compose_strassen(strassen_2x2())
    assert big.dims == (4, 4, 4)
    assert big.rank == 49


def test_composed_kronecker_exact_on_random():
    base = schoolbook(MatmulSpec(2, 2, 2))
    big = compose_kronecker(base, tiling=2)
    a, b = random_cpu_inputs(big.spec, seed=7)
    assert verify_exact(big, a, b)


def test_composed_strassen_exact_on_random():
    big = compose_strassen(strassen_2x2())
    a, b = random_cpu_inputs(big.spec, seed=11)
    assert verify_exact(big, a, b)


def test_compose_strassen_requires_square_base():
    with pytest.raises(ValueError):
        compose_strassen(schoolbook(MatmulSpec(2, 3, 4)))


def test_compose_kronecker_rejects_bad_tiling():
    with pytest.raises(ValueError):
        compose_kronecker(strassen_2x2(), tiling=0)


def test_provenance_survives_serialization():
    big = compose_strassen(strassen_2x2())
    text = mm.formats.to_json(big)
    restored = mm.formats.from_json(text)
    assert restored.provenance["composition"] == "strassen"
    assert restored.provenance["base_algorithm"] == "strassen_2x2"
    assert len(restored.provenance["components"]) == big.rank


def test_verify_decomposition_kronecker():
    base = strassen_2x2()
    big = compose_kronecker(base, tiling=2)
    assert mm.verify_decomposition(big, base)


def test_verify_decomposition_strassen():
    base = strassen_2x2()
    big = compose_strassen(base)
    assert mm.verify_decomposition(big, base)


def test_identity_single_factor_compose():
    # Tiling by 1 is the identity in rank and dims.
    base = schoolbook(MatmulSpec(2, 2, 2))
    big = compose_kronecker(base, tiling=1)
    assert big.dims == base.dims
    assert big.rank == base.rank
