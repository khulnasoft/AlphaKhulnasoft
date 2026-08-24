"""Tensor/matrix-multiplication package for AlphaKhulnasoft.

Provides an exact, immutable representation of bilinear matrix-multiplication
algorithms, serialization, loading, recombination, an equivalence verifier,
and an optional CPU/CUDA benchmark runner. This package is intentionally
independent of API keys, LLM providers, and the proof sandbox.
"""

from __future__ import annotations

from .algorithms import formats as formats
from .algorithms import loader as loader
from .algorithms import registry as registry
from .algorithms.builtin import schoolbook, strassen_2x2, strassen_block_4x4
from .nonequivalence.four_by_four import four_by_four_fixtures, schoolbook_4x4, strassen_4x4
from .nonequivalence.invariants import invariants, separates
from .nonequivalence.verifier import EquivalenceReport, EquivalenceResult, check_equivalence
from .recombination.compose import compose_kronecker, compose_strassen, operation_count
from .recombination.decomposition import Decomposition, describe_decomposition, verify_decomposition
from .reference import reconstruct, reference_multiply, verify_exact
from .tensor import Factorization, Field, MatmulSpec, RankOneFactor

__all__ = [
    "Factorization",
    "Field",
    "MatmulSpec",
    "RankOneFactor",
    "reference_multiply",
    "reconstruct",
    "verify_exact",
    "schoolbook",
    "strassen_2x2",
    "strassen_block_4x4",
    "formats",
    "loader",
    "registry",
    "compose_kronecker",
    "compose_strassen",
    "operation_count",
    "Decomposition",
    "describe_decomposition",
    "verify_decomposition",
    "four_by_four_fixtures",
    "schoolbook_4x4",
    "strassen_4x4",
    "invariants",
    "separates",
    "EquivalenceReport",
    "EquivalenceResult",
    "check_equivalence",
]
