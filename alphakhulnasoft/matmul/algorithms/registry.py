"""Deterministic discovery and named lookup of built-in algorithms."""

from __future__ import annotations

from ..nonequivalence.four_by_four import schoolbook_4x4, strassen_4x4
from ..tensor import Factorization, MatmulSpec
from .builtin import schoolbook, strassen_2x2, strassen_block_4x4

_REGISTRY: dict[str, Factorization] | None = None


def _build_registry() -> dict[str, Factorization]:
    algos: dict[str, Factorization] = {}
    for m, k, n in ((1, 1, 1), (2, 2, 2), (2, 3, 4), (3, 3, 3)):
        algos[f"schoolbook_{m}x{k}x{n}"] = schoolbook(MatmulSpec(m, k, n))
    algos["strassen_2x2"] = strassen_2x2()
    algos["strassen_block_4x4"] = strassen_block_4x4()
    algos["schoolbook_4x4"] = schoolbook_4x4()
    algos["strassen_4x4"] = strassen_4x4()
    return algos


def registry() -> dict[str, Factorization]:
    """Return the cached, deterministic algorithm registry."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY


def names() -> list[str]:
    """Return registered algorithm names in deterministic (sorted) order."""
    return sorted(registry().keys())


def get(name: str) -> Factorization:
    """Look up a named algorithm, raising ``KeyError`` if absent."""
    try:
        return registry()[name]
    except KeyError:
        raise KeyError(f"Unknown algorithm {name!r}; known: {names()}") from None


def lookup(name: str) -> Factorization:
    """Alias for :func:`get`."""
    return get(name)
