"""Equivalence verifier under a bounded transformation group.

Two factorizations are *equivalent* if one can be obtained from the other by
an allowed transformation: reordering factors and/or rescaling a factor's
``u`` by ``lambda`` while rescaling its ``v`` by ``1/lambda`` (the bilinear
scalar product is invariant). We never treat a failed search as proof of
nonequivalence: if the invariants do not separate the two and the bounded
search does not find a match, the result is INCONCLUSIVE.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from typing import Any

from ..tensor import Factorization
from .invariants import separates


class EquivalenceResult(str, Enum):
    EQUIVALENT = "equivalent"
    NONEQUIVALENT = "nonequivalent"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class EquivalenceReport:
    result: EquivalenceResult
    evidence: str
    invariant: str | None = None
    search_exhausted: bool = False


def _factor_signature(
    factor: Any,
) -> tuple[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    return (factor.u, factor.v, factor.w)


def check_equivalence(
    a: Factorization,
    b: Factorization,
    max_factors_for_search: int = 12,
) -> EquivalenceReport:
    """Decide equivalence under the supported transformation group.

    The decision procedure:
    1. If an invariant separates them -> PROVEN NONEQUIVALENT.
    2. If ranks match and the size is small enough, run a bounded search over
       factor permutations + per-factor scaling. A match -> EQUIVALENT.
    3. Otherwise -> INCONCLUSIVE (we do not overstate certainty).
    """
    if a.spec != b.spec:
        return EquivalenceReport(
            EquivalenceResult.NONEQUIVALENT, "Dimensions differ.", "dimensions"
        )
    if a.field != b.field:
        return EquivalenceReport(EquivalenceResult.NONEQUIVALENT, "Fields differ.", "field")

    invariant = separates(a, b)
    if invariant is not None and invariant not in ("dimensions", "field"):
        return EquivalenceReport(
            EquivalenceResult.NONEQUIVALENT,
            f"Separated by invariant '{invariant}'.",
            invariant,
        )

    if a.rank != b.rank:
        return EquivalenceReport(
            EquivalenceResult.NONEQUIVALENT,
            f"Ranks differ ({a.rank} vs {b.rank}).",
            "rank",
        )

    if a.rank > max_factors_for_search:
        return EquivalenceReport(
            EquivalenceResult.INCONCLUSIVE,
            "Invariants agree; bounded search skipped for large rank. "
            "Nonequivalence is not proven.",
            None,
            search_exhausted=False,
        )

    found, exhausted = _bounded_search(a, b)
    if found:
        return EquivalenceReport(
            EquivalenceResult.EQUIVALENT,
            "Matched under factor permutation + per-factor scaling.",
            None,
        )
    return EquivalenceReport(
        EquivalenceResult.INCONCLUSIVE,
        "Invariants agree but bounded search found no exact match.",
        None,
        search_exhausted=exhausted,
    )


def _bounded_search(a: Factorization, b: Factorization) -> tuple[bool, bool]:
    """Search factor permutations + scalings for an exact match (exact fields)."""
    if not a.field.exact:
        return False, True
    b_factors = list(b.factors)
    for perm in permutations(range(a.rank)):
        if _match_under_scaling([a.factors[i] for i in perm], b_factors):
            return True, True
    return False, True


def _match_under_scaling(ordered_a: list[Any], b_factors: list[Any]) -> bool:
    """Check if a specific ordering of A matches B up to per-factor scaling."""

    return all(_factor_match_scaled(fa, fb) for fa, fb in zip(ordered_a, b_factors, strict=True))


def _factor_match_scaled(fa: Any, fb: Any) -> bool:
    """Two factors match if there exists lambda with u_a*lambda=u_b, v_a/lambda=v_b, w equal."""
    from fractions import Fraction

    ua, va, wa = fa.u, fa.v, fa.w
    ub, vb, wb = fb.u, fb.v, fb.w
    if wa != wb:
        return False
    lam: Fraction | None = None
    for i in range(len(ua)):
        for j in range(len(ua[0])):
            a_val = Fraction(ua[i][j])
            b_val = Fraction(ub[i][j])
            if a_val == 0 and b_val == 0:
                continue
            if a_val == 0 or b_val == 0:
                return False
            cand = b_val / a_val
            if lam is None:
                lam = cand
            elif lam != cand:
                return False
    if lam is None:
        # All u entries zero; check v/scaling consistency instead.
        for i in range(len(va)):
            for j in range(len(va[0])):
                a_val = Fraction(va[i][j])
                b_val = Fraction(vb[i][j])
                if a_val == 0 and b_val == 0:
                    continue
                if a_val == 0 or b_val == 0:
                    return False
                cand = a_val / b_val  # lambda = a_val / b_val since v scaled by 1/lambda
                if lam is None:
                    lam = cand
                elif lam != cand:
                    return False
    # Verify v scaling consistency with found lambda.
    if lam is None:
        return True
    for i in range(len(va)):
        for j in range(len(va[0])):
            a_val = Fraction(va[i][j])
            b_val = Fraction(vb[i][j])
            if a_val == 0 and b_val == 0:
                continue
            if a_val == 0 or b_val == 0:
                return False
            if b_val / a_val != 1 / lam:
                return False
    return True
