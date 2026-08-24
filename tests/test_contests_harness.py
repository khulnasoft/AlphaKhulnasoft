"""Tests for the multi-language execution harness and comparator."""

import pytest

from alphakhulnasoft.contests.harness import Grade, compare_outputs, grade_solution
from alphakhulnasoft.contests.problem import ContestProblem, TestCase

CORRECT = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)"
WRONG = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a-b)"


def _problem() -> ContestProblem:
    return ContestProblem(
        "p",
        "codeforces",
        "Add two ints.",
        tests=[TestCase("1 2", "3"), TestCase("5 7", "12")],
    )


def test_python_correct_passes():
    g = grade_solution(_problem(), CORRECT, "py", visible_only=False)
    assert g.all_passed()
    assert g.passed == 2


def test_python_wrong_fails():
    g = grade_solution(_problem(), WRONG, "py")
    assert g.failed == 2
    assert g.passed == 0


def test_visible_only_grades_subset():
    p = ContestProblem(
        "p",
        "codeforces",
        "Add.",
        tests=[TestCase("1 2", "3"), TestCase("5 7", "12", hidden=True)],
    )
    g = grade_solution(p, CORRECT, "py", visible_only=True)
    assert g.total == 1
    assert g.all_passed()


def test_compile_error_classified():
    # Interpreted languages report a syntax error as RE; compiled languages get
    # a dedicated CE from the compile step. We assert it is never a PASS and is
    # counted as an error.
    bad = "def main(:\n  pass"
    g = grade_solution(_problem(), bad, "py")
    assert g.errored == 2
    assert all(status in ("RE", "CE") for status in g.per_test)


def test_timeout_classified():
    # An infinite busy loop must trip the subprocess timeout -> TLE.
    loop = "while True:\n    pass"
    g = grade_solution(_problem(), loop, "py", limits={"timeout_s": 1})
    assert g.timeouts >= 1


def test_compare_outputs_whitespace_tolerant():
    assert compare_outputs("1 2 3", "1 2 3\n")
    assert compare_outputs("1 2 3", "1\n2\n3")
    assert not compare_outputs("1 2 3", "1 2 4")


def test_grade_dataclass_metrics():
    g = Grade(per_test=["PASS", "WA", "RE", "TLE"], total=4)
    assert g.passed == 1 and g.failed == 1 and g.errored == 1 and g.timeouts == 1


@pytest.mark.skipif(__import__("shutil").which("g++") is None, reason="g++ not installed")
def test_cpp_round_trip():
    cpp = "#include <iostream>\nusing namespace std;\nint main(){int a,b;cin>>a>>b;cout<<a+b;return 0;}"
    g = grade_solution(_problem(), cpp, "cpp")
    assert g.all_passed()
