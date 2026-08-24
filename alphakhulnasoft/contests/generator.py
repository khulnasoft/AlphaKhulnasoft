"""Generator: the novel-solving flow loop for contest problems.

Wraps the Analyze → Generate → Verify → Root-Cause → Repair loop. The LLM is
injected (an ``LLMProvider`` or any object with ``complete`` / ``extract_code``)
so tests can use a deterministic fake. Reference solutions are NEVER consulted
here — generation must be from reasoning, not copying.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .harness import grade_solution
from .problem import ContestProblem, TestCase
from .prompts import ContestPromptRegistry


@dataclass
class GeneratedCandidate:
    """One candidate solution produced by the generator."""

    code: str
    analysis: str = ""
    strategy: str = ""
    grade: Any = None  # Grade over visible tests
    status: str = "PENDING"
    novelty: float | None = None


class ContestAgent:
    """Runs the contest flow loop and produces candidate solutions."""

    def __init__(
        self,
        llm: Any,
        prompts: Any = ContestPromptRegistry,
        sandbox_timeout: int = 5,
        max_repair_iters: int = 3,
    ):
        self.llm = llm
        self.prompts = prompts
        self.timeout = sandbox_timeout
        self.max_repair_iters = max_repair_iters

    def analyze(self, problem: ContestProblem) -> str:
        prompt = self.prompts.analyze(problem.statement)
        return str(self.llm.complete(prompt, system_prompt="You are an algorithms analyst."))

    def generate(
        self, problem: ContestProblem, analysis: str, language: str, strategy: str | None = None
    ) -> str:
        prompt = self.prompts.generate(problem.statement, analysis, language, strategy)
        raw = self.llm.complete(
            prompt,
            system_prompt=f"You are an expert {language} competitive programmer. Output only code.",
        )
        return self._clean(raw)

    def repair(self, problem: ContestProblem, code: str, error_log: str, analysis: str) -> str:
        prompt = self.prompts.repair(problem.statement, code, error_log, analysis)
        raw = self.llm.complete(prompt, system_prompt="You are a senior debugging engineer.")
        return self._clean(raw)

    def generate_pool(
        self,
        problem: ContestProblem,
        language: str,
        n_samples: int = 10,
        strategies: list[str] | None = None,
    ) -> list[GeneratedCandidate]:
        """Produce ``n_samples`` diverse candidates, grading each on visible tests."""
        analysis = self.analyze(problem)
        candidates: list[GeneratedCandidate] = []
        for i in range(n_samples):
            strategy = strategies[i % len(strategies)] if strategies else None
            code = self.generate(problem, analysis, language, strategy)
            grade = grade_solution(
                problem, code, language, visible_only=True, limits={"timeout_s": self.timeout}
            )
            status = "SOLVED" if grade.all_passed() else "REPAIRING"
            candidates.append(
                GeneratedCandidate(
                    code=code,
                    analysis=analysis,
                    strategy=strategy or "",
                    grade=grade,
                    status=status,
                )
            )
        return candidates

    def solve(
        self,
        problem: ContestProblem,
        language: str,
        n_samples: int = 5,
        strategies: list[str] | None = None,
    ) -> GeneratedCandidate:
        """Generate a pool and repair visible failures; return the best candidate."""
        pool = self.generate_pool(problem, language, n_samples=n_samples, strategies=strategies)
        best: GeneratedCandidate | None = None
        for cand in pool:
            if cand.grade is not None and cand.grade.all_passed():
                return cand
            best = cand
        # Repair the first failing candidate within the iteration budget.
        if best is not None and best.grade is not None and not best.grade.all_passed():
            error_log = self._error_log(problem, best)
            for _ in range(self.max_repair_iters):
                best.code = self.repair(problem, best.code, error_log, best.analysis)
                best.grade = grade_solution(
                    problem,
                    best.code,
                    language,
                    visible_only=True,
                    limits={"timeout_s": self.timeout},
                )
                if best.grade.all_passed():
                    best.status = "SOLVED"
                    return best
                error_log = self._error_log(problem, best)
            best.status = "FAILED"
        return best or GeneratedCandidate(code="")

    def _error_log(self, problem: ContestProblem, cand: GeneratedCandidate) -> str:
        parts: list[str] = []
        if cand.grade is None or not cand.grade.per_test:
            return "No visible tests passed."
        for i, status in enumerate(cand.grade.per_test):
            if status != "PASS":
                test: TestCase = problem.visible_tests[i]
                parts.append(
                    f"Test {i + 1} {status}\nInput: {test.input}\nExpected: {test.expected_output}"
                )
        return "\n".join(parts[:3]) or "Unknown failure."

    def _clean(self, text: str) -> str:
        if hasattr(self.llm, "extract_code"):
            return str(self.llm.extract_code(text))
        if "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()
