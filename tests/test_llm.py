import litellm
import pytest

from alphakhulnasoft.llm import LLMProvider


@pytest.fixture
def llm_provider():
    return LLMProvider()


def test_extract_code_python_block(llm_provider):
    text = "Here's the code:\n```python\ndef solve(n):\n    return n * 2\n```\nEnd"
    assert llm_provider.extract_code(text) == "def solve(n):\n    return n * 2"


def test_extract_code_generic_block(llm_provider):
    text = "```\ndef solve(n):\n    return n * 2\n```"
    assert llm_provider.extract_code(text) == "def solve(n):\n    return n * 2"


def test_extract_code_no_block(llm_provider):
    text = "def solve(n): return n * 2"
    assert llm_provider.extract_code(text) == "def solve(n): return n * 2"


def test_extract_code_empty(llm_provider):
    assert llm_provider.extract_code("") == ""


def test_extract_code_whitespace(llm_provider):
    assert llm_provider.extract_code("   ") == ""


def test_extract_code_prefers_python_over_generic(llm_provider):
    text = "```\ngeneric code\n```\n```python\ndef solve():\n    pass\n```"
    assert llm_provider.extract_code(text) == "def solve():\n    pass"


def test_extract_code_multiple_python_blocks(llm_provider):
    text = "```python\ndef first():\n    return 1\n```\n```python\ndef second():\n    return 2\n```"
    assert llm_provider.extract_code(text) == "def first():\n    return 1"


def test_extract_code_unclosed_block(llm_provider):
    text = "```python\ndef solve():\n    return 42"
    assert llm_provider.extract_code(text) == "def solve():\n    return 42"


def test_extract_code_other_language_tags(llm_provider):
    text = "```javascript\nconsole.log('hello');\n```"
    assert llm_provider.extract_code(text) == "javascript\nconsole.log('hello');"


def test_extract_code_mixed_markdown(llm_provider):
    text = "Some text before\n```python\ndef solve():\n    return 42\n```\nSome text after"
    assert llm_provider.extract_code(text) == "def solve():\n    return 42"


def test_complete_retries_on_transient_error(monkeypatch, llm_provider):
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
    result = llm_provider.complete("test prompt")
    assert result == "success"
    assert call_count["value"] == 2


def test_complete_raises_after_max_retries(monkeypatch, llm_provider):
    llm_provider.max_retries = 2
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
        llm_provider.complete("test prompt")
    assert call_count["value"] == 2


def test_complete_invalid_response_structure_raises_value_error(monkeypatch, llm_provider):
    def invalid_response(*args, **kwargs):
        resp = litellm.ModelResponse()
        resp.choices = []
        return resp

    monkeypatch.setattr(litellm, "completion", invalid_response)

    with pytest.raises(ValueError, match="Invalid LLM response structure"):
        llm_provider.complete("test prompt")
