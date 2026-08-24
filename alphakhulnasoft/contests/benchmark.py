"""Benchmark harness: pass@k and novel_pass@k on a problem split.

Keeps a stable, reproducible schema and never conflates CPU with GPU/V100 runs
(consistent with the rest of the repo). The headline ``novel_pass@k`` is the
milestone metric: of the problems attempted, the fraction solved by a solution
the verifier judges *novel* (not a near-duplicate of a reference).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .generator import ContestAgent
from .harness import grade_solution
from .loader import get_references
from .planner import classify_selection, select_top
from .problem import ContestProblem
from .verifier import NOVEL_THRESHOLD


@dataclass
class ProblemResult:
    problem_id: str
    source: str
    k: int
    pass_at_k: float
    novel_pass_at_k: float
    top_novelty: float
    top_novelty_label: str
    n_candidates: int
    n_visible_survivors: int
    mean_novelty: float

    def as_dict(self) -> dict:
        return {
            "problem_id": self.problem_id,
            "source": self.source,
            "k": self.k,
            "pass_at_k": self.pass_at_k,
            "novel_pass_at_k": self.novel_pass_at_k,
            "top_novelty": self.top_novelty,
            "top_novelty_label": self.top_novelty_label,
            "n_candidates": self.n_candidates,
            "n_visible_survivors": self.n_visible_survivors,
            "mean_novelty": self.mean_novelty,
        }


@dataclass
class BenchmarkReport:
    language: str
    n_samples: int
    k: int
    problem_results: list[ProblemResult] = field(default_factory=list)
    error: str | None = None
    reference_pass_at_k: float | None = None

    @property
    def n_problems(self) -> int:
        return len(self.problem_results)

    def pass_at_k(self) -> float:
        if not self.problem_results:
            return 0.0
        return sum(r.pass_at_k for r in self.problem_results) / len(self.problem_results)

    def novel_pass_at_k(self) -> float:
        if not self.problem_results:
            return 0.0
        return sum(r.novel_pass_at_k for r in self.problem_results) / len(self.problem_results)

    def mean_novelty(self) -> float:
        if not self.problem_results:
            return 0.0
        return sum(r.mean_novelty for r in self.problem_results) / len(self.problem_results)

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "n_samples": self.n_samples,
            "k": self.k,
            "n_problems": self.n_problems,
            "pass_at_k": self.pass_at_k(),
            "novel_pass_at_k": self.novel_pass_at_k(),
            "reference_pass_at_k": self.reference_pass_at_k,
            "mean_novelty": self.mean_novelty(),
            "error": self.error,
            "problems": [r.as_dict() for r in self.problem_results],
        }

    def to_file(self, path: str) -> None:
        """Write the report as JSON to ``path``."""
        import json

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)

    def publish(self, repo_id: str, token: str | None = None) -> None:
        """Upload the report to a Hugging Face dataset repo (plan §7.3)."""
        import os
        import tempfile

        from alphakhulnasoft.publisher import HFPublisher

        fd, tmp = tempfile.mkstemp(suffix=".json", prefix="contest_bench_")
        os.close(fd)
        try:
            self.to_file(tmp)
            HFPublisher(repo_id, token=token).publish_results(tmp)
        finally:
            os.remove(tmp)


def reference_ceiling(problems: list[ContestProblem], language: str = "py") -> dict[str, Any]:
    """The 'copying' ceiling (plan §7.2): pass@k if we just reused reference solutions.

    For every problem, each same-language reference solution is graded on the full
    (visible + hidden) test set. ``reference_pass_at_k`` is the fraction of problems
    where at least one reference passes. The milestone's ``novel_pass@k`` must beat
    this to demonstrate genuine problem solving rather than memorization.
    Requires no LLM.
    """
    rows: list[dict[str, Any]] = []
    for p in problems:
        refs = get_references(p, language)
        passed = bool(refs) and any(
            grade_solution(p, r.code, language, visible_only=False).all_passed() for r in refs
        )
        rows.append({"problem_id": p.problem_id, "reference_pass": passed})
    n = len(rows)
    ceil = (sum(r["reference_pass"] for r in rows) / n) if n else 0.0
    return {"language": language, "n_problems": n, "reference_pass_at_k": ceil, "problems": rows}


def run_benchmark(
    problems: list[ContestProblem],
    agent: ContestAgent,
    language: str,
    n_samples: int = 10,
    k: int = 1,
    llm: Any | None = None,
    novel_threshold: float = NOVEL_THRESHOLD,
    publish_repo: str | None = None,
    hf_token: str | None = None,
    include_reference_ceiling: bool = False,
) -> BenchmarkReport:
    """Run the sample→filter→rank→grade pipeline over ``problems``."""
    report = BenchmarkReport(language=language, n_samples=n_samples, k=k)
    if include_reference_ceiling:
        report.reference_pass_at_k = reference_ceiling(problems, language)["reference_pass_at_k"]
    for problem in problems:
        try:
            result = _bench_problem(problem, agent, language, n_samples, k, llm, novel_threshold)
        except Exception as exc:  # keep the run going; record the failure
            result = ProblemResult(
                problem_id=problem.problem_id,
                source=problem.source,
                k=k,
                pass_at_k=0.0,
                novel_pass_at_k=0.0,
                top_novelty=0.0,
                top_novelty_label="retrieved",
                n_candidates=n_samples,
                n_visible_survivors=0,
                mean_novelty=0.0,
            )
            report.error = str(exc)
        report.problem_results.append(result)
    if publish_repo:
        report.publish(publish_repo, token=hf_token)
    return report


def _bench_problem(
    problem: ContestProblem,
    agent: ContestAgent,
    language: str,
    n_samples: int,
    k: int,
    llm: Any | None,
    novel_threshold: float,
) -> ProblemResult:
    pool = agent.generate_pool(problem, language, n_samples=n_samples)
    survivors = [c for c in pool if c.grade is not None and c.grade.all_passed()]
    selected = select_top(pool, problem, language, k=k, llm=llm)
    labels = classify_selection(selected)

    # Grade selected candidates on the full (incl. hidden) test set.
    hidden_correct: list[bool] = []
    for cand in selected:
        grade = grade_solution(problem, cand.code, language, visible_only=False)
        hidden_correct.append(grade.all_passed())

    pass_k = 1.0 if any(hidden_correct) else 0.0
    top_novel = selected[0].novelty if selected else 0.0
    novel_pass_k = (
        1.0 if (selected and any(hidden_correct) and (top_novel or 0.0) >= novel_threshold) else 0.0
    )
    mean_nov = (sum(c.novelty or 0.0 for c in selected) / len(selected)) if selected else 0.0

    return ProblemResult(
        problem_id=problem.problem_id,
        source=problem.source,
        k=k,
        pass_at_k=pass_k,
        novel_pass_at_k=novel_pass_k,
        top_novelty=top_novel if top_novel is not None else 0.0,
        top_novelty_label=labels[0] if labels else "retrieved",
        n_candidates=len(pool),
        n_visible_survivors=len(survivors),
        mean_novelty=mean_nov,
    )
