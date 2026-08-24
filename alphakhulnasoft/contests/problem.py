"""Canonical, language-agnostic model of a competitive-programming problem.

A :class:`ContestProblem` captures everything needed to generate and grade a
solution: the statement, extracted I/O format, visible samples, the full set of
test cases (visible + hidden), and the dataset's reference solutions. Reference
solutions are stored but are explicitly *quarantined*: the loader only hands
them to the verifier/plagiarism path, never to the generator, so that "solving"
cannot degenerate into copying.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class ProblemSource:
    """Known problem sources from the Code Contests dataset."""

    AIZU = "aizu"
    ATCODER = "atcoder"
    CODECHEF = "codechef"
    CODEFORCES = "codeforces"
    HACKEREARTH = "hackerearth"

    ALL = (AIZU, ATCODER, CODECHEF, CODEFORCES, HACKEREARTH)


@dataclass(frozen=True)
class IOFormat:
    """Structured summary of a problem's I/O contract."""

    input_format: str = ""
    output_format: str = ""
    constraints: str = ""
    time_limit_s: float | None = None
    memory_limit_mb: int | None = None


@dataclass(frozen=True)
class TestCase:
    """A single input/expected-output pair, optionally hidden from contestants."""

    __test__ = False  # prevent pytest from collecting this dataclass as a test

    input: str
    expected_output: str
    hidden: bool = False


@dataclass(frozen=True)
class ReferenceSolution:
    """A dataset-provided solution, used only for grading / novelty baselines."""

    language: str
    code: str
    status: str = "unknown"  # e.g. "ok", "fail"


@dataclass(frozen=True)
class ContestProblem:
    """Immutable description of one contest problem."""

    problem_id: str
    source: str
    statement: str
    tests: list[TestCase]
    samples: list[tuple[str, str]] = field(default_factory=list)
    io_format: IOFormat = field(default_factory=IOFormat)
    reference_solutions: list[ReferenceSolution] = field(default_factory=list)
    language: str = "py"

    def __post_init__(self) -> None:
        if not self.problem_id:
            raise ValueError("problem_id must be non-empty")
        if self.source not in ProblemSource.ALL:
            raise ValueError(f"Unknown problem source {self.source!r}")
        if not self.statement.strip():
            raise ValueError("statement must be non-empty")
        if not self.tests:
            raise ValueError("a problem must have at least one test case")
        for i, test in enumerate(self.tests):
            if not isinstance(test, TestCase):
                raise ValueError(f"tests[{i}] must be a TestCase")

    @property
    def visible_tests(self) -> list[TestCase]:
        return [t for t in self.tests if not t.hidden]

    @property
    def hidden_tests(self) -> list[TestCase]:
        return [t for t in self.tests if t.hidden]

    def references_for(self, language: str) -> list[ReferenceSolution]:
        """Reference solutions in a specific language (verifier-only use)."""
        return [r for r in self.reference_solutions if r.language == language]

    def validate(self) -> None:
        """Re-validate the problem (raises on malformed data)."""
        self.__post_init__()
