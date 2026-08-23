"""Integration tests for AlphaKhulnasoft with mocked LLM responses."""

import os
from unittest.mock import MagicMock, patch

from alphakhulnasoft.alpha_repair import AlphaRepairAgent
from alphakhulnasoft.config import AlphaConfig


def test_integration_full_flow_basic(monkeypatch):
    """Test the basic flow execution with mocked LLM responses."""
    # Mock LLM responses
    mock_llm = MagicMock()
    call_count = [0]

    def mock_complete(prompt, system_prompt=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return "Algo: Simple arithmetic\nConstraints: Basic\nEdge Cases: None\nComplexity: O(1)"
        elif call_count[0] == 2:
            return "print(42)"
        else:
            return "mock response"

    mock_llm.complete = mock_complete
    mock_llm.extract_code.side_effect = lambda x: x.strip()

    # Mock LLMProvider to return our mock
    with (
        patch("alphakhulnasoft.alpha_repair.LLMProvider", return_value=mock_llm),
        patch("alphakhulnasoft.llm.validate_api_keys"),
    ):  # Skip API key validation
        agent = AlphaRepairAgent()
        agent.llm = mock_llm

        problem = "Simple test problem"
        tests = [{"input": "", "expected": "42"}]

        result = agent.run_flow(problem, tests=tests)

        # Just verify the flow executed and returned a result
        assert result is not None
        assert "status" in result
        assert "metrics" in result


def test_integration_full_flow_failure(monkeypatch):
    """Test the complete flow with mocked LLM responses that never solve the problem."""
    # Mock LLM responses that never produce correct code
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [
        "Algo: Simple arithmetic\nConstraints: Handle negative numbers\nEdge Cases: Zero, negative input\nComplexity: O(1)",
        "def solve():\n    print('wrong')",  # Wrong solution
        "ROOT CAUSE: Incorrect output",
        "def solve():\n    print('still wrong')",  # Still wrong
        "ROOT CAUSE: Logic error persists",
        "def solve():\n    print('wrong again')",  # Still wrong
    ]
    mock_llm.extract_code.side_effect = (
        lambda x: x.split("```")[1].split("```")[0].strip() if "```" in x else x.strip()
    )

    with (
        patch("alphakhulnasoft.alpha_repair.LLMProvider", return_value=mock_llm),
        patch("alphakhulnasoft.llm.validate_api_keys"),
    ):  # Skip API key validation
        config = AlphaConfig(max_retries=3)
        agent = AlphaRepairAgent(config=config)
        agent.llm = mock_llm

        problem = "Write a function to double an integer."
        tests = [{"input": "2", "expected": "4"}]

        result = agent.run_flow(problem, tests=tests)

        assert result["status"] == "FAILED"
        assert result["metrics"]["iterations"] == 3


def test_integration_semantic_analysis_failure(monkeypatch):
    """Test flow when semantic analysis fails."""
    mock_llm = MagicMock()
    mock_llm.complete.side_effect = Exception("API Error")

    with (
        patch("alphakhulnasoft.alpha_repair.LLMProvider", return_value=mock_llm),
        patch("alphakhulnasoft.llm.validate_api_keys"),
    ):
        agent = AlphaRepairAgent()
        agent.llm = mock_llm

        result = agent.run_flow("Test problem", tests=[])

        assert result["status"] == "FAILED"
        assert "Fatal Error" in result["solution"] or result["solution"] == ""


def test_integration_config_from_env(monkeypatch):
    """Test that configuration can be loaded from environment variables."""
    monkeypatch.setenv("ALPHA_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("ALPHA_MAX_RETRIES", "10")
    monkeypatch.setenv("ALPHA_SANDBOX_TIMEOUT", "5")
    monkeypatch.setenv("ALPHA_LLM_RETRIES", "5")
    monkeypatch.setenv("ALPHA_MAX_MEMORY_MB", "1024")

    config = AlphaConfig.from_env()

    assert config.model_name == "gpt-4o-mini"
    assert config.max_retries == 10
    assert config.sandbox_timeout == 5
    assert config.llm_max_retries == 5
    assert config.max_memory_mb == 1024


def test_integration_api_key_validation():
    """Test API key validation function."""
    from alphakhulnasoft.llm import validate_api_keys

    # Save original values
    original_openai = os.getenv("OPENAI_API_KEY")
    original_anthropic = os.getenv("ANTHROPIC_API_KEY")
    original_vertex = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    try:
        # Test no keys
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)

        try:
            validate_api_keys()
            raise AssertionError("Should have raised ValueError")
        except ValueError as e:
            assert "No LLM API keys found" in str(e)

        # Test with OpenAI key
        os.environ["OPENAI_API_KEY"] = "test-key"
        validate_api_keys()  # Should not raise

        # Test with Anthropic key
        os.environ.pop("OPENAI_API_KEY", None)
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        validate_api_keys()  # Should not raise

        # Test with Vertex credentials
        os.environ.pop("ANTHROPIC_API_KEY", None)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/creds.json"
        validate_api_keys()  # Should not raise

    finally:
        # Restore original values
        if original_openai:
            os.environ["OPENAI_API_KEY"] = original_openai
        elif "OPENAI_API_KEY" in os.environ:
            os.environ.pop("OPENAI_API_KEY")

        if original_anthropic:
            os.environ["ANTHROPIC_API_KEY"] = original_anthropic
        elif "ANTHROPIC_API_KEY" in os.environ:
            os.environ.pop("ANTHROPIC_API_KEY")

        if original_vertex:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = original_vertex
        elif "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS")


def test_integration_sandbox_memory_limit():
    """Test that sandbox respects memory limits."""
    from alphakhulnasoft.sandbox import Sandbox

    sb = Sandbox(timeout_seconds=2, max_memory_mb=100)
    assert sb.max_memory_mb == 100
    assert sb.timeout == 2


def test_integration_logging_config():
    """Test logging configuration."""
    from alphakhulnasoft.logging_config import get_logger, setup_logging

    # Test setup with default settings
    setup_logging(level="INFO")
    logger = get_logger("test")
    assert logger is not None
    assert logger.name == "test"

    # Test setup with JSON format
    setup_logging(level="DEBUG", json_format=True)
    logger = get_logger("test2")
    assert logger is not None
