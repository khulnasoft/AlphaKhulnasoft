"""Tests for the Code Contests loader and reference quarantine."""

import json

import pytest

import alphakhulnasoft.contests as cc
from alphakhulnasoft.contests.loader import (
    get_references,
    load_local,
    problem_from_dict,
)

FIXTURE = "alphakhulnasoft/contests/data/tiny.jsonl"


def test_fixture_loads():
    problems = load_local(FIXTURE)
    assert len(problems) == 3
    assert {p.problem_id for p in problems} == {
        "demo_aplusb_1",
        "demo_aplusb_2",
        "demo_aplusb_3",
    }


def test_loader_rejects_malformed_line(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"problem_id": "x", "source": "codeforces", "statement": "s", "tests": []}\n')
    with pytest.raises(ValueError):
        load_local(bad)


def test_loader_rejects_invalid_json(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    with pytest.raises(ValueError):
        load_local(bad)


def test_loader_rejects_unknown_source():
    with pytest.raises(ValueError):
        problem_from_dict(
            {
                "problem_id": "x",
                "source": "unknown",
                "statement": "s",
                "tests": [{"input": "1", "output": "1"}],
            }
        )


def test_round_trip_dict():
    p = load_local(FIXTURE)[0]
    again = problem_from_dict(json.loads(json.dumps(_to_dict(p))))
    assert again.problem_id == p.problem_id
    assert again.tests == p.tests


def _to_dict(p) -> dict:
    return {
        "problem_id": p.problem_id,
        "source": p.source,
        "statement": p.statement,
        "tests": [
            {"input": t.input, "output": t.expected_output, "hidden": t.hidden} for t in p.tests
        ],
        "samples": p.samples,
        "reference_solutions": [
            {"language": r.language, "code": r.code, "status": r.status}
            for r in p.reference_solutions
        ],
    }


def test_references_quarantined_from_generator():
    # Static guarantee: the generator module must never consult reference
    # solutions (the "solving vs copying" boundary).
    import pathlib

    src = pathlib.Path(cc.__file__).parent.joinpath("generator.py").read_text()
    assert "reference_solution" not in src
    assert "get_references" not in src


def test_get_references_verifier_only():
    p = load_local(FIXTURE)[0]
    refs = get_references(p, "py")
    assert refs and all(r.language == "py" for r in refs)
