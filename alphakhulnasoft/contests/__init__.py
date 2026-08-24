"""Competitive-programming problem-solving package for AlphaKhulnasoft.

Targets *novel* solving of unforeseen contest problems (Aizu, AtCoder,
CodeChef, Codeforces, HackerEarth via the Code Contests dataset). Reuses the
Flow Engineering loop and HF publisher but owns its problem model, multi-language
harness, generator, and a novelty/anti-copying verifier. Reference solutions are
quarantined: the generator never sees them; only the verifier may.
"""

from __future__ import annotations

from .benchmark import BenchmarkReport, ProblemResult, run_benchmark
from .generator import ContestAgent, GeneratedCandidate
from .harness import Grade, compare_outputs, grade_solution
from .languages import get_language_spec, known_languages, register_language
from .loader import get_references, load_huggingface, load_local, problem_from_dict
from .planner import classify_selection, rank_candidates, select_top
from .problem import (
    ContestProblem,
    IOFormat,
    ProblemSource,
    ReferenceSolution,
    TestCase,
)
from .prompts import ContestPromptRegistry
from .verifier import (
    NOVEL_THRESHOLD,
    RETRIEVED_THRESHOLD,
    classify_novelty,
    jaccard,
    memorization_probe,
    novelty_score,
    similarity_to_references,
    tokenize,
)

__all__ = [
    "ContestProblem",
    "IOFormat",
    "ProblemSource",
    "ReferenceSolution",
    "TestCase",
    "get_language_spec",
    "known_languages",
    "register_language",
    "grade_solution",
    "Grade",
    "compare_outputs",
    "load_local",
    "load_huggingface",
    "problem_from_dict",
    "get_references",
    "ContestPromptRegistry",
    "ContestAgent",
    "GeneratedCandidate",
    "rank_candidates",
    "select_top",
    "classify_selection",
    "tokenize",
    "jaccard",
    "similarity_to_references",
    "memorization_probe",
    "novelty_score",
    "classify_novelty",
    "NOVEL_THRESHOLD",
    "RETRIEVED_THRESHOLD",
    "run_benchmark",
    "BenchmarkReport",
    "ProblemResult",
]
