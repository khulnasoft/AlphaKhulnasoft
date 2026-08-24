"""Novelty & anti-copying verification.

The differentiator for the competitive-programming milestone: every generated
solution is compared against the dataset's reference solutions and (optionally)
probed for memorization. Novelty is a discrete, threshold-based score in [0, 1]
— it never uses floating-point tolerance to declare a solution "new".
"""

from __future__ import annotations

import re
from typing import Any

from .loader import get_references
from .problem import ContestProblem

# Classification cutoffs (documented, discrete).
NOVEL_THRESHOLD = 0.7
RETRIEVED_THRESHOLD = 0.3

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+|[^A-Za-z0-9_\s]")


def tokenize(code: str) -> list[str]:
    """Break source into a stable token stream (identifiers, numbers, punct)."""
    return _TOKEN_RE.findall(code)


def jaccard(a: list[str], b: list[str]) -> float:
    """Token Jaccard similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    set_a, set_b = set(a), set(b)
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def similarity_to_references(problem: ContestProblem, code: str, language: str) -> float:
    """Max token similarity between ``code`` and same-language references."""
    refs = get_references(problem, language)
    if not refs:
        return 0.0
    tokens = tokenize(code)
    return max(jaccard(tokens, tokenize(r.code)) for r in refs)


def memorization_probe(problem: ContestProblem, llm: Any) -> bool:
    """Ask the model to reproduce the canonical solution; true if it complies.

    Returns ``True`` when the model's recalled solution is highly similar to a
    reference (i.e. it is *recalling*, not *solving*). Requires an ``llm`` with a
    ``complete(prompt, system_prompt=...)`` method; returns ``False`` if ``llm``
    is ``None`` (probe skipped).
    """
    if llm is None:
        return False
    prompt = (
        "Reproduce, verbatim, the most well-known reference solution for this "
        f"problem in {problem.language}. Output only code.\n\nPROBLEM:\n{problem.statement}"
    )
    recalled = str(
        llm.complete(prompt, system_prompt="You are a competitive programming historian.")
    )
    recalled = _strip_code_fences(recalled)
    return similarity_to_references(problem, recalled, problem.language) >= NOVEL_THRESHOLD


def novelty_score(
    problem: ContestProblem,
    code: str,
    language: str,
    llm: Any | None = None,
) -> float:
    """Return a novelty score in [0, 1] (1 = unlike all references, not recalled)."""
    sim = similarity_to_references(problem, code, language)
    score = 1.0 - sim
    if llm is not None and memorization_probe(problem, llm):
        # Strong evidence of recall: clamp novelty toward zero.
        score = min(score, RETRIEVED_THRESHOLD)
    return max(0.0, min(1.0, score))


def classify_novelty(score: float) -> str:
    """Map a novelty score to a discrete label."""
    if score >= NOVEL_THRESHOLD:
        return "novel"
    if score <= RETRIEVED_THRESHOLD:
        return "retrieved"
    return "borderline"


def _strip_code_fences(text: str) -> str:
    if "```" in text:
        return text.split("```")[1].split("```")[0].strip()
    return text.strip()
