"""Planner: sample → filter by visible tests → rank by novelty.

Correctness on visible tests is the hard gate. Among survivors we rank by
novelty so the selected solution is more likely to be *solved* rather than
*copied*. The memorization probe (needs an LLM) is optional and only tightens
the novelty estimate.
"""

from __future__ import annotations

from typing import Any

from .generator import GeneratedCandidate
from .problem import ContestProblem
from .verifier import classify_novelty, novelty_score


def _survivors(pool: list[GeneratedCandidate]) -> list[GeneratedCandidate]:
    return [c for c in pool if c.grade is not None and c.grade.all_passed()]


def rank_candidates(
    pool: list[GeneratedCandidate],
    problem: ContestProblem,
    language: str,
    llm: Any | None = None,
) -> list[GeneratedCandidate]:
    """Return visible-test survivors annotated with novelty, ranked best-first."""
    survivors = _survivors(pool)
    for cand in survivors:
        cand.novelty = novelty_score(problem, cand.code, language, llm=llm)
    survivors.sort(key=lambda c: (c.novelty or 0.0), reverse=True)
    return survivors


def select_top(
    pool: list[GeneratedCandidate],
    problem: ContestProblem,
    language: str,
    k: int = 1,
    llm: Any | None = None,
) -> list[GeneratedCandidate]:
    """Return up to ``k`` ranked, visible-passing candidates."""
    ranked = rank_candidates(pool, problem, language, llm=llm)
    return ranked[:k]


def classify_selection(candidates: list[GeneratedCandidate]) -> list[str]:
    """Novelty labels for a selection (top-1 drives the 'novel' verdict)."""
    return [classify_novelty(c.novelty or 0.0) for c in candidates]
