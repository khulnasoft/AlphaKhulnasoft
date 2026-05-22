import litellm
import pytest

from alphakhulnasoft.llm import LLMProvider


def test_extract_code_python_block():
    llm = LLMProvider()
    text = "Here's the code:\n```python\ndef solve(n):\n    return n * 2\n```\nEnd"
    assert llm.extract_code(text) == "def solve(n):\n    return n * 2"


def test_extract_code_generic_block():
    llm = LLMProvider()
    text = "```\ndef solve(n):\n    return n * 2\n```"
    assert llm.extract_code(text) == "def solve(n):\n    return n * 2"


def test_extract_code_no_block():
    llm = LLMProvider()
    text = "def solve(n): return n * 2"
    assert llm.extract_code(text) == "def solve(n): return n * 2"


def test_extract_code_empty():
    llm = LLMProvider()
    assert llm.extract_code("") == ""


def test_extract_code_whitespace():
    llm = LLMProvider()
    assert llm.extract_code("   ") == ""


def test_extract_code_prefers_python_over_generic():
    llm = LLMProvider()
    text = "```\ngeneric code\n```\n```python\ndef solve():\n    pass\n```"
    assert llm.extract_code(text) == "def solve():\n    pass"


def test_complete_retries_on_transient_error(monkeypatch):
    llm = LLMProvider()
    call_count = {"value": 0}

    def fake_completion(*args, **kwargs):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise litellm.RateLimitError(
                message="Rate limited",
                llm_provider="openai",
                model="gpt-4-turbo",
            )
        msg = litellm.Message(content="success", role="assistant")
        choice = litellm.Choices(finish_reason="stop", index=0, message=msg)
        resp = litellm.ModelResponse()
        resp.choices = [choice]
        return resp

    monkeypatch.setattr(litellm, "completion", fake_completion)
    result = llm.complete("test prompt")
    assert result == "success"
    assert call_count["value"] == 2


def test_complete_raises_after_max_retries(monkeypatch):
    llm = LLMProvider(max_retries=2)
    call_count = {"value": 0}

    def always_fails(*args, **kwargs):
        call_count["value"] += 1
        raise litellm.RateLimitError(
            message="Still rate limited",
            llm_provider="openai",
            model="gpt-4-turbo",
        )

    monkeypatch.setattr(litellm, "completion", always_fails)

    with pytest.raises(litellm.RateLimitError):
        llm.complete("test prompt")
    assert call_count["value"] == 2


def test_complete_invalid_response_structure_raises_value_error(monkeypatch):
    llm = LLMProvider()

    def invalid_response(*args, **kwargs):
        resp = litellm.ModelResponse()
        resp.choices = []
        return resp

    monkeypatch.setattr(litellm, "completion", invalid_response)

    with pytest.raises(ValueError, match="Invalid LLM response structure"):
        llm.complete("test prompt")
