"""Tests for the contest benchmark harness (pass@k / novel_pass@k)."""

from alphakhulnasoft.contests.benchmark import run_benchmark
from alphakhulnasoft.contests.generator import ContestAgent
from alphakhulnasoft.contests.loader import load_local

FIXTURE = "alphakhulnasoft/contests/data/tiny.jsonl"
CORRECT = "import sys\nx,y=map(int,sys.stdin.read().split())\nprint(x+y)"
WRONG = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a-b)"


class _SolvingLLM:
    def complete(self, prompt, system_prompt=None):
        return CORRECT

    def extract_code(self, text):
        return text.strip()


class _FailingLLM:
    def complete(self, prompt, system_prompt=None):
        return WRONG

    def extract_code(self, text):
        return text.strip()


def test_benchmark_stable_schema_and_nonnegative():
    problems = load_local(FIXTURE)
    report = run_benchmark(problems, ContestAgent(llm=_SolvingLLM()), "py", n_samples=6, k=2)
    d = report.as_dict()
    for key in (
        "language",
        "n_samples",
        "k",
        "n_problems",
        "pass_at_k",
        "novel_pass_at_k",
        "mean_novelty",
    ):
        assert key in d
    assert d["n_problems"] == len(problems)
    assert 0.0 <= d["pass_at_k"] <= 1.0
    assert 0.0 <= d["novel_pass_at_k"] <= 1.0
    assert d["error"] is None
    assert len(d["problems"]) == len(problems)


def test_benchmark_pass_at_k_perfect_when_solving():
    problems = load_local(FIXTURE)
    report = run_benchmark(problems, ContestAgent(llm=_SolvingLLM()), "py", n_samples=6, k=1)
    assert report.pass_at_k() == 1.0


def test_benchmark_pass_at_k_zero_when_failing():
    problems = load_local(FIXTURE)
    report = run_benchmark(problems, ContestAgent(llm=_FailingLLM()), "py", n_samples=6, k=1)
    assert report.pass_at_k() == 0.0


def test_benchmark_reports_novelty_labels():
    problems = load_local(FIXTURE)
    report = run_benchmark(problems, ContestAgent(llm=_SolvingLLM()), "py", n_samples=6, k=1)
    for pr in report.problem_results:
        assert pr.top_novelty_label in ("novel", "borderline", "retrieved")
