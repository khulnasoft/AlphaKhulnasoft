"""Tests for the contest benchmark harness (pass@k / novel_pass@k)."""

from unittest import mock

from alphakhulnasoft.contests.benchmark import _bench_problem, run_benchmark
from alphakhulnasoft.contests.generator import ContestAgent, GeneratedCandidate
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


def test_reference_ceiling_is_computed_without_llm():
    from alphakhulnasoft.contests.benchmark import reference_ceiling

    problems = load_local(FIXTURE)
    ceil = reference_ceiling(problems, "py")
    assert ceil["n_problems"] == len(problems)
    assert ceil["reference_pass_at_k"] == 1.0  # fixture references are correct


def test_run_benchmark_can_include_reference_ceiling():
    problems = load_local(FIXTURE)
    report = run_benchmark(
        problems,
        ContestAgent(llm=_SolvingLLM()),
        "py",
        n_samples=6,
        k=1,
        include_reference_ceiling=True,
    )
    assert report.reference_pass_at_k == 1.0
    assert report.as_dict()["reference_pass_at_k"] == 1.0


class _StubAgent:
    def __init__(self, pool):
        self._pool = pool

    def generate_pool(self, problem, language, n_samples=0):
        return self._pool


def _grade_mock(correct_codes):
    def _grade(problem, code, language, visible_only=False, limits=None):
        class _G:
            def all_passed(self):
                return code in correct_codes

        return _G()

    return _grade


def test_novel_pass_requires_same_candidate_to_pass_and_be_novel():
    # Regression for the bug: a novel top candidate that FAILS hidden tests while a
    # sub-threshold candidate PASSES must NOT count as a novel pass.
    p = load_local(FIXTURE)[0]
    novel_fail = GeneratedCandidate(code="NOVEL_FAIL", novelty=0.9)  # novel but wrong
    dull_pass = GeneratedCandidate(code="DULL_PASS", novelty=0.2)  # passes but not novel
    pool = [novel_fail, dull_pass]
    with mock.patch("alphakhulnasoft.contests.benchmark.select_top", return_value=pool), mock.patch(
        "alphakhulnasoft.contests.benchmark.grade_solution", _grade_mock({"DULL_PASS"})
    ):
        res = _bench_problem(p, _StubAgent(pool), "py", n_samples=2, k=2, llm=None, novel_threshold=0.7)
    assert res.pass_at_k == 1.0  # a candidate does pass hidden tests
    assert res.novel_pass_at_k == 0.0  # but the passing one is not novel


def test_novel_pass_when_a_novel_candidate_passes():
    p = load_local(FIXTURE)[0]
    novel_pass = GeneratedCandidate(code="NOVEL_PASS", novelty=0.9)
    dull_fail = GeneratedCandidate(code="DULL_FAIL", novelty=0.2)
    pool = [dull_fail, novel_pass]
    with mock.patch("alphakhulnasoft.contests.benchmark.select_top", return_value=pool), mock.patch(
        "alphakhulnasoft.contests.benchmark.grade_solution", _grade_mock({"NOVEL_PASS"})
    ):
        res = _bench_problem(p, _StubAgent(pool), "py", n_samples=2, k=2, llm=None, novel_threshold=0.7)
    assert res.pass_at_k == 1.0
    assert res.novel_pass_at_k == 1.0  # the passing candidate is also novel
