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
