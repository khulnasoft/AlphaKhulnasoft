import argparse

import pytest

from alphakhulnasoft.contests.benchmark import reference_ceiling
from alphakhulnasoft.contests.cli import _load_problems
from alphakhulnasoft.contests.problem import ContestProblem, TestCase


def test_reference_ceiling_rejects_negative_max_refs():
    p = ContestProblem("x", "codeforces", "stmt", tests=[TestCase("1", "1")])
    with pytest.raises(ValueError):
        reference_ceiling([p], "py", max_refs=-1)
    # Zero and positive must not raise (preserve existing slicing behavior).
    reference_ceiling([p], "py", max_refs=0)
    reference_ceiling([p], "py", max_refs=3)


def test_load_problems_rejects_negative_limit():
    args = argparse.Namespace(hf_dataset="x", split="valid", dataset=None, limit=-1)
    with pytest.raises(SystemExit):
        _load_problems(args)


def test_load_problems_limit_zero_local_empty():
    args = argparse.Namespace(
        hf_dataset=None,
        split="valid",
        dataset="alphakhulnasoft/contests/data/tiny.jsonl",
        limit=0,
    )
    assert _load_problems(args) == []


def test_make_llm_preserves_explicit_rpm_zero(monkeypatch):
    import alphakhulnasoft.contests.llm_shim as shim

    captured: dict = {}

    class StubProvider:
        def __init__(self, model, max_retries, validate_keys, rate_limit_rpm):
            captured.update(rate_limit_rpm=rate_limit_rpm, validate_keys=validate_keys)

    monkeypatch.setattr("alphakhulnasoft.llm.LLMProvider", StubProvider)

    class StubCfg:
        model_name = "m"
        llm_max_retries = 2
        rate_limit_rpm = 15

    monkeypatch.setattr(
        "alphakhulnasoft.config.AlphaConfig.from_env", staticmethod(lambda: StubCfg())
    )

    shim.make_llm(model=None, rpm=0)
    assert captured["rate_limit_rpm"] == 0 and captured["validate_keys"] is True

    shim.make_llm(model=None, rpm=None)
    assert captured["rate_limit_rpm"] == 15
