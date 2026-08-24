"""Tests for factorization serialization, loading, and registry."""

import pytest

import alphakhulnasoft.matmul as mm
from alphakhulnasoft.matmul.algorithms import formats, loader, registry
from alphakhulnasoft.matmul.tensor import Factorization, MatmulSpec

DATA = "alphakhulnasoft/matmul/data/schoolbook_2x2.json"


def test_json_round_trip_preserves_values_and_metadata():
    alg = mm.schoolbook(MatmulSpec(2, 3, 4))
    restored = formats.from_dict(formats.to_dict(alg))
    assert restored.rank == alg.rank
    assert restored.algorithm_name == alg.algorithm_name
    assert restored.field == alg.field
    assert restored.dims == alg.dims
    assert restored.provenance == alg.provenance
    assert restored.factors == alg.factors


def test_from_dict_rejects_unknown_format_version():
    payload = formats.to_dict(mm.schoolbook(MatmulSpec(2, 2, 2)))
    payload["format_version"] = "9.9"
    with pytest.raises(ValueError):
        formats.from_dict(payload)


def test_from_dict_rejects_unknown_field():
    payload = formats.to_dict(mm.schoolbook(MatmulSpec(2, 2, 2)))
    payload["field"] = "complex128"
    with pytest.raises(ValueError):
        formats.from_dict(payload)


def test_from_dict_rejects_bad_dimensions():
    payload = formats.to_dict(mm.schoolbook(MatmulSpec(2, 2, 2)))
    payload["dimensions"]["m"] = 0
    with pytest.raises(ValueError):
        formats.from_dict(payload)


def test_from_dict_rejects_empty_factors():
    payload = formats.to_dict(mm.schoolbook(MatmulSpec(2, 2, 2)))
    payload["factors"] = []
    with pytest.raises(ValueError):
        formats.from_dict(payload)


def test_from_dict_rejects_malformed_factor_keys():
    payload = {
        "format_version": "1.0",
        "algorithm": "bad",
        "field": "integer",
        "dimensions": {"m": 2, "k": 2, "n": 2},
        "rank": 1,
        "provenance": {},
        "factors": [{"u": [[1, 0], [0, 1]], "v": [[1, 0], [0, 1]]}],
    }
    with pytest.raises(ValueError):
        formats.from_dict(payload)


def test_loader_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        loader.load_local("does/not/exist.json")


def test_loader_loads_checked_in_fixture():
    alg = loader.load_local(DATA)
    assert alg.rank == 8
    assert alg.dims == (2, 2, 2)
    assert isinstance(alg, Factorization)


def test_registry_names_deterministic():
    names = registry.names()
    assert names == sorted(names)
    assert "schoolbook_2x2x2" in names
    assert "strassen_2x2" in names


def test_registry_ranks_deterministic():
    assert registry.get("schoolbook_2x2x2").rank == 8
    assert registry.get("strassen_2x2").rank == 7
    assert registry.get("strassen_4x4").rank == 49


def test_registry_unknown_raises():
    with pytest.raises(KeyError):
        registry.get("nonexistent_algorithm")


def test_builtin_algorithms_exact():
    assert mm.verify_exact(registry.get("strassen_2x2"), [[1, 2], [3, 4]], [[5, 6], [7, 8]])
    a = [[1, 0, 2], [3, 1, 0]]
    b = [[1, 2], [0, 1], [2, 0]]
    assert mm.verify_exact(
        registry.get("schoolbook_2x3x2")
        if "schoolbook_2x3x2" in registry.names()
        else mm.schoolbook(MatmulSpec(2, 3, 2)),
        a,
        b,
    )
