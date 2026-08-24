"""Tests for the contest problem data model and validation."""

import pytest

from alphakhulnasoft.contests.problem import (
    ContestProblem,
    IOFormat,
    ProblemSource,
    ReferenceSolution,
    TestCase,
)


def _problem(**overrides) -> ContestProblem:
    base = {
        "problem_id": "p1",
        "source": "codeforces",
        "statement": "Add two integers.",
        "tests": [TestCase("1 2", "3"), TestCase("5 7", "12", hidden=True)],
    }
    base.update(overrides)
    return ContestProblem(**base)


def test_valid_problem():
    p = _problem()
    assert p.problem_id == "p1"
    assert p.visible_tests == [p.tests[0]]
    assert p.hidden_tests == [p.tests[1]]


def test_missing_problem_id_rejected():
    with pytest.raises(ValueError):
        _problem(problem_id="")


def test_unknown_source_rejected():
    with pytest.raises(ValueError):
        _problem(source="leetcode")


def test_empty_statement_rejected():
    with pytest.raises(ValueError):
        _problem(statement="   ")


def test_no_tests_rejected():
    with pytest.raises(ValueError):
        ContestProblem("p", "atcoder", "stmt", tests=[])


def test_references_for_language():
    p = _problem(
        reference_solutions=[
            ReferenceSolution("py", "print(1)", "ok"),
            ReferenceSolution("cpp", "int main(){}", "ok"),
        ]
    )
    assert len(p.references_for("py")) == 1
    assert len(p.references_for("cpp")) == 1


def test_problemsource_all_contains_five_sources():
    assert set(ProblemSource.ALL) >= {"aizu", "atcoder", "codechef", "codeforces", "hackerearth"}


def test_io_format_optional():
    p = _problem(io_format=IOFormat(time_limit_s=2.0, memory_limit_mb=256))
    assert p.io_format.time_limit_s == 2.0
