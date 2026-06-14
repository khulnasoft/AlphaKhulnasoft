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


def test_step_execute_tests_logs_warning_on_empty(monkeypatch):
    agent = AlphaRepairAgent()
    state = FlowState(problem_desc="test")

    def fake_run_tests(code, test_cases):
        return 0.0, "No test cases provided"

    monkeypatch.setattr(agent.sandbox, "run_tests", fake_run_tests)
    pass_rate, log = agent.step_execute_tests(state)
    assert pass_rate == 0.0
    assert "No test cases" in log
    assert any("No tests provided" in msg for msg in state.execution_logs)


def test_step_execute_tests_forwards_to_sandbox(monkeypatch):
    agent = AlphaRepairAgent()
    state = FlowState(problem_desc="test", tests=[{"input": "2", "expected": "4"}])

    called = {"value": False}

    def fake_run_tests(code, test_cases):
        called["value"] = True
        assert test_cases == state.tests
        return 1.0, ""

    monkeypatch.setattr(agent.sandbox, "run_tests", fake_run_tests)
    pass_rate, log = agent.step_execute_tests(state)
    assert called["value"]
    assert pass_rate == 1.0


def test_run_flow_handles_step_exception(monkeypatch):
    agent = AlphaRepairAgent()

    def failing_step(state):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "step_generate_solution", failing_step)
    result = agent.run_flow("test problem")
    assert result["status"] == "FAILED"


def test_run_flow_fails_when_max_retries_exhausted(monkeypatch):
    config = AlphaConfig(max_retries=3)
    agent = AlphaRepairAgent(config=config)

    def fake_semantic_analysis(state):
        state.constraints = "dummy constraints"
        return state

    def fake_generate_solution(state):
        state.current_code = "print(1)"
        return state

    def fake_step_execute_tests(state):
        return 0.0, "test failure"

    def fake_step_analyze_failure(state, error_log):
        return "dummy root cause"

    def fake_step_apply_fix(state, root_cause, error_log):
        return state

    monkeypatch.setattr(agent, "step_semantic_analysis", fake_semantic_analysis)
    monkeypatch.setattr(agent, "step_generate_solution", fake_generate_solution)
    monkeypatch.setattr(agent, "step_execute_tests", fake_step_execute_tests)
    monkeypatch.setattr(agent, "step_analyze_failure", fake_step_analyze_failure)
    monkeypatch.setattr(agent, "step_apply_fix", fake_step_apply_fix)

    result = agent.run_flow("test problem")
    assert result["status"] == "FAILED"
    assert result["metrics"]["iterations"] == 3
