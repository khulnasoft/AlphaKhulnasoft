from alphakhulnasoft.alpha_repair import AlphaRepairAgent, FlowState
from alphakhulnasoft.config import AlphaConfig
from alphakhulnasoft.prompts import PromptRegistry


def test_flow_state_defaults():
    state = FlowState(problem_desc="test")
    assert state.id is not None
    assert state.problem_desc == "test"
    assert state.constraints == ""
    assert state.current_code == ""
    assert state.tests == []
    assert state.execution_logs == []
    assert state.status == "PENDING"
    assert state.iterations == 0
    assert state.confidence_score == 0.0
    assert state.history == []


def test_flow_state_with_tests():
    tests = [{"input": "2", "expected": "4"}]
    state = FlowState(problem_desc="solve", tests=tests)
    assert state.tests == tests


def test_clean_markdown_no_markdown():
    agent = AlphaRepairAgent()
    code = "def solve():\n    return 42"
    assert agent._clean_markdown(code) == code


def test_clean_markdown_python_block():
    agent = AlphaRepairAgent()
    text = "```python\ndef solve():\n    return 42\n```"
    assert agent._clean_markdown(text) == "def solve():\n    return 42"


def test_clean_markdown_generic_block():
    agent = AlphaRepairAgent()
    text = "```\ndef solve():\n    return 42\n```"
    assert agent._clean_markdown(text) == "def solve():\n    return 42"


def test_finalize_result_solved():
    agent = AlphaRepairAgent()
    state = FlowState(problem_desc="test")
    state.status = "SOLVED"
    state.current_code = "print(42)"
    state.iterations = 3
    state.confidence_score = 1.0
    state.history = [{"iter": 1}, {"iter": 2}]

    result = agent._finalize_result(state)
    assert result["solution"] == "print(42)"
    assert result["status"] == "SOLVED"
    assert result["metrics"]["iterations"] == 3
    assert result["metrics"]["confidence"] == 1.0
    assert result["metrics"]["flow_depth"] == 2


def test_finalize_result_failed():
    agent = AlphaRepairAgent()
    state = FlowState(problem_desc="test")
    state.status = "FAILED"
    state.current_code = ""
    state.iterations = 0
    state.confidence_score = 0.0

    result = agent._finalize_result(state)
    assert result["status"] == "FAILED"
    assert result["solution"] == ""


def test_alpha_config_defaults():
    config = AlphaConfig()
    assert config.model_name == "gpt-4o"
    assert config.max_retries == 5
    assert config.sandbox_timeout == 2
    assert config.llm_max_retries == 3


def test_agent_accepts_config():
    config = AlphaConfig(model_name="gpt-4o-mini", max_retries=3, sandbox_timeout=5)
    agent = AlphaRepairAgent(config=config)
    assert agent.model == "gpt-4o-mini"
    assert agent.max_retries == 3
    assert agent.sandbox.timeout == 5


def test_agent_default_uses_prompt_registry():
    agent = AlphaRepairAgent()
    assert agent.prompts == PromptRegistry
