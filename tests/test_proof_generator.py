import json
import os

from alphakhulnasoft.config import AlphaConfig
from alphakhulnasoft.proof_evaluator import ProofEvaluator
from alphakhulnasoft.proof_generator import ProofRepairAgent, ProofState
from alphakhulnasoft.proof_prompts import ProofPromptRegistry
from alphakhulnasoft.proof_sandbox import ProofSandbox


def test_proof_state_defaults():
    state = ProofState(theorem_statement="test theorem")
    assert state.id is not None
    assert state.theorem_statement == "test theorem"
    assert state.proof_analysis == ""
    assert state.current_nexus_code == ""
    assert state.current_prose == ""
    assert state.verification_status == "PENDING"
    assert state.status == "PENDING"
    assert state.iterations == 0
    assert state.confidence_score == 0.0
    assert state.history == []


def test_proof_state_with_hints():
    hints = "Use induction."
    state = ProofState(theorem_statement="t", proof_hints=hints)
    assert state.proof_hints == hints
    assert state.expected_technique == ""


def test_proof_sandbox_heuristic_empty():
    sandbox = ProofSandbox()
    is_valid, error = sandbox.verify_nexus_proof("")
    assert is_valid is False
    assert "Empty" in error


def test_proof_sandbox_heuristic_missing_theorem():
    sandbox = ProofSandbox()
    is_valid, error = sandbox.verify_nexus_proof("lemma foo := sorry")
    assert is_valid is False
    assert "theorem" in error.lower()


def test_proof_sandbox_rejects_theorem_word_in_comment():
    sandbox = ProofSandbox()
    is_valid, error = sandbox.verify_nexus_proof("# theorem foo\nlemma bar := by exact 1")
    assert is_valid is False
    assert "theorem" in error.lower()


def test_proof_sandbox_heuristic_ends_with_sorry():
    sandbox = ProofSandbox()
    code = "theorem sum : ∀ n, sum n = n*(n+1)/2 :=\n  sorry"
    is_valid, error = sandbox.verify_nexus_proof(code)
    assert is_valid is False
    assert "incomplete" in error.lower()


def test_proof_sandbox_heuristic_valid():
    sandbox = ProofSandbox()
    code = "theorem sum : ∀ n, sum n = n*(n+1)/2 := by induction n"
    is_valid, error = sandbox.verify_nexus_proof(code)
    assert is_valid is True
    assert error == ""


def test_proof_sandbox_check_tactics():
    sandbox = ProofSandbox()
    code = "theorem t := by induction n\ncases h\nlemma aux := sorry"
    tactics = sandbox.check_proof_tactics(code)
    assert tactics["uses_induction"] is True
    assert tactics["uses_cases"] is True
    assert tactics["has_lemmas"] is True
    assert tactics["uses_sorry"] is True


def test_proof_prompt_registry_methods_return_strings():
    theorem = "For all n, sum n = n*(n+1)/2"
    assert isinstance(ProofPromptRegistry.analyze_proof_requirements(theorem), str)
    assert isinstance(ProofPromptRegistry.generate_nexus_proof(theorem, "analysis"), str)
    assert isinstance(ProofPromptRegistry.analyze_proof_error(theorem, "code", "err"), str)
    assert isinstance(ProofPromptRegistry.repair_nexus_proof(theorem, "code", "cause"), str)
    assert isinstance(ProofPromptRegistry.generate_prose_explanation(theorem, "code"), str)


def test_clean_proof_markdown_nexus_block():
    agent = ProofRepairAgent(config=AlphaConfig(validate_api_keys=False))
    text = "```nexus\ntheorem t := by exact 42\n```"
    assert agent._clean_proof_markdown(text) == "theorem t := by exact 42"


def test_clean_proof_markdown_python_block():
    agent = ProofRepairAgent(config=AlphaConfig(validate_api_keys=False))
    text = "```python\ntheorem t := sorry\n```"
    assert agent._clean_proof_markdown(text) == "theorem t := sorry"


def test_clean_proof_markdown_no_fence():
    agent = ProofRepairAgent(config=AlphaConfig(validate_api_keys=False))
    assert agent._clean_proof_markdown("theorem t := sorry") == "theorem t := sorry"


def test_finalize_proof_result():
    agent = ProofRepairAgent(config=AlphaConfig(validate_api_keys=False))
    state = ProofState(theorem_statement="t")
    state.status = "PROVEN"
    state.verification_status = "VALID"
    state.current_nexus_code = "theorem t := by exact 1"
    state.current_prose = "Proof: ... QED."
    state.iterations = 2
    state.confidence_score = 1.0
    state.history = [{"iter": 1}, {"iter": 2}]

    result = agent._finalize_proof_result(state)
    assert result["status"] == "PROVEN"
    assert result["is_valid"] is True
    assert result["nexus_code"] == "theorem t := by exact 1"
    assert result["prose"] == "Proof: ... QED."
    assert result["metrics"]["iterations"] == 2
    assert result["metrics"]["proof_depth"] == 2
    assert result["error"] is None


def test_agent_accepts_config():
    config = AlphaConfig(model_name="gpt-4o-mini", max_retries=3, validate_api_keys=False)
    agent = ProofRepairAgent(config=config)
    assert agent.model == "gpt-4o-mini"
    assert agent.max_retries == 3
    assert agent.prompts is ProofPromptRegistry


def test_run_proof_flow_handles_step_exception(monkeypatch):
    agent = ProofRepairAgent(config=AlphaConfig(validate_api_keys=False))

    def failing_step(state):
        raise RuntimeError("boom")

    monkeypatch.setattr(agent, "step_analyze_proof_requirements", failing_step)
    result = agent.run_proof_flow("test theorem")
    assert result["status"] == "FAILED"
    assert result["is_valid"] is False


def test_run_proof_flow_fails_when_max_retries_exhausted(monkeypatch):
    config = AlphaConfig(max_retries=2, validate_api_keys=False)
    agent = ProofRepairAgent(config=config)

    def fake_analyze(state):
        state.proof_analysis = "analysis"
        return state

    def fake_generate(state):
        state.current_nexus_code = "theorem t := sorry"
        return state

    def fake_verify(state):
        # Heuristic considers ending in sorry invalid
        return False, "incomplete"

    def fake_analyze_error(state, error_log):
        return "ROOT CAUSE: missing case"

    def fake_repair(state, root_cause, error_log):
        # Repair still leaves it ending in sorry -> still invalid
        return state

    monkeypatch.setattr(agent, "step_analyze_proof_requirements", fake_analyze)
    monkeypatch.setattr(agent, "step_generate_nexus_proof", fake_generate)
    monkeypatch.setattr(agent, "step_verify_proof", fake_verify)
    monkeypatch.setattr(agent, "step_analyze_proof_error", fake_analyze_error)
    monkeypatch.setattr(agent, "step_repair_nexus_proof", fake_repair)

    result = agent.run_proof_flow("test theorem")
    assert result["status"] == "FAILED"
    assert result["metrics"]["iterations"] == 2


def test_run_proof_flow_proves_when_valid(monkeypatch):
    config = AlphaConfig(max_retries=3, validate_api_keys=False)
    agent = ProofRepairAgent(config=config)

    def fake_analyze(state):
        state.proof_analysis = "analysis"
        return state

    def fake_generate(state):
        state.current_nexus_code = "theorem t := by exact 1"
        return state

    def fake_verify(state):
        return True, ""

    def fake_prose(state):
        state.current_prose = "Proof: by exact. QED."
        return state

    monkeypatch.setattr(agent, "step_analyze_proof_requirements", fake_analyze)
    monkeypatch.setattr(agent, "step_generate_nexus_proof", fake_generate)
    monkeypatch.setattr(agent, "step_verify_proof", fake_verify)
    monkeypatch.setattr(agent, "step_generate_prose_explanation", fake_prose)

    result = agent.run_proof_flow("test theorem")
    assert result["status"] == "PROVEN"
    assert result["is_valid"] is True
    assert result["prose"] == "Proof: by exact. QED."


def test_evaluator_score_and_efficiency():
    evaluator = ProofEvaluator()
    score = evaluator.score_proof("theorem t := by\n  exact 1", is_valid=True, iterations=2)
    assert score["is_valid"] is True
    assert score["iterations_to_valid"] == 2
    assert score["conciseness"] == 2
    # depth: one 2-space indented line among 2 lines => max depth 1
    assert score["proof_depth"] == 1
    assert evaluator.calculate_efficiency_score(True, 2) == 0.5
    assert evaluator.calculate_efficiency_score(False, 2) == 0.0


def test_benchmark_dataset_loading(tmp_path):
    dataset = tmp_path / "theorems.jsonl"
    dataset.write_text(
        json.dumps({"id": "x1", "theorem": "t1", "hints": "induction"})
        + "\n"
        + json.dumps({"id": "x2", "theorem": "t2"})
        + "\n"
    )

    # Patch the agent so no LLM calls happen
    import alphakhulnasoft.proof_benchmark as pb

    class FakeAgent:
        model = "fake"

        def run_proof_flow(self, theorem, proof_hints=""):
            return {
                "nexus_code": "theorem t := by exact 1",
                "prose": "Proof. QED.",
                "status": "PROVEN",
                "is_valid": True,
                "theorem": theorem,
                "metrics": {"iterations": 1, "confidence": 1.0, "proof_depth": 0},
                "error": None,
            }

    monkeypatch_agent = FakeAgent()

    out_file = tmp_path / "out.json"
    # Monkeypatch ProofRepairAgent inside the benchmark module
    original = pb.ProofRepairAgent
    pb.ProofRepairAgent = lambda: monkeypatch_agent
    try:
        result = pb.run_proof_benchmark(str(dataset), output_file=str(out_file))
    finally:
        pb.ProofRepairAgent = original

    assert result["summary"]["total"] == 2
    assert result["summary"]["proven"] == 2
    assert os.path.exists(out_file)
    saved = json.loads(out_file.read_text())
    assert saved["summary"]["proof_at_1"] == 1.0
