"""Tests for the Hugging Face loader path without touching the network.

We monkeypatch ``datasets.load_dataset`` so the real mapping logic in
``_code_contests_row_to_dict`` is exercised deterministically.
"""

from unittest import mock

import pytest

from alphakhulnasoft.contests.loader import get_references, load_huggingface

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("datasets") is None,
    reason="datasets not installed",
)


def _fake_rows():
    return [
        {
            "name": "cf_123_a",
            "description": "Add two integers a and b.",
            "tests": [
                {"input": "1 2", "output": "3", "hidden": False},
                {"input": "5 7", "output": "12", "hidden": True},
            ],
            "solutions": [
                {
                    "language": "python",
                    "solution": "print(sum(map(int,input().split())))",
                    "status": "OK",
                },
                {"language": "cpp", "solution": "int main(){}", "status": "OK"},
                {"language": "python", "solution": "print(0)", "status": "BAD"},
            ],
        },
        {
            # malformed: no tests -> should be skipped, not raise
            "name": "cf_bad",
            "description": "No tests here.",
            "tests": [],
            "solutions": [],
        },
    ]


def _fake_load_dataset(name, split, streaming):
    assert name == "code_contests"
    return iter(_fake_rows())


def test_load_huggingface_maps_rows_and_keeps_references():
    with mock.patch("datasets.load_dataset", _fake_load_dataset):
        problems = load_huggingface(name="code_contests", split="train", limit=10)
    assert len(problems) == 1  # the malformed row is skipped
    p = problems[0]
    assert p.problem_id == "cf_123_a"
    assert p.source == "codeforces"
    assert len(p.tests) == 2
    assert p.hidden_tests[0].hidden is True
    # References are loaded (and quarantined behind get_references), not dropped.
    refs = get_references(p, "py")
    assert len(refs) == 2  # two python solutions (OK + BAD)
    assert any(r.language == "cpp" for r in get_references(p, "cpp"))


def test_load_huggingface_limit_caps_rows():
    rows = _fake_rows() * 3  # 6 rows, 3 valid
    with mock.patch("datasets.load_dataset", lambda *a, **k: iter(rows)):
        problems = load_huggingface(name="code_contests", split="train", limit=2)
    assert len(problems) <= 2
