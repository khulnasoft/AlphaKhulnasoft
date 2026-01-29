import uuid
from dataclasses import dataclass, field
from typing import Literal

from .llm import LLMProvider
from .prompts import PromptRegistry
from .sandbox import Sandbox


# --- 1. The Shared State (The Brain) ---
@dataclass
class FlowState:
    """Tracks the entire lifecycle of a coding problem."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    problem_desc: str = ""
    constraints: str = ""  # Now stores the LLM analysis string
    current_code: str = ""
    tests: list[dict] = field(default_factory=list)  # [{'input': '...', 'expected': '...'}]
    execution_logs: list[str] = field(default_factory=list)
    status: Literal["PENDING", "SOLVED", "FAILED", "REPAIRING"] = "PENDING"
    iterations: int = 0
    confidence_score: float = 0.0
    history: list[dict] = field(default_factory=list)  # Traceability


# --- 2. The Agent Core ---
class AlphaRepairAgent:
    def __init__(self, model_name="gpt-4o", max_retries=5, prompt_registry=PromptRegistry):
        self.model = model_name
        self.max_retries = max_retries
        self.llm = LLMProvider(model=model_name)
        self.prompts = prompt_registry
        self.sandbox = Sandbox(timeout_seconds=2)

    def run_flow(self, problem_description: str, tests: list[dict] | None = None) -> dict:
        """Entry point for the Flow Engineering loop."""
        state = FlowState(problem_desc=problem_description, tests=tests or [])

        print(f"🚀 [AlphaFlow] Starting Logic Flow for Problem ID: {state.id}")

        # Step 1: Semantic Analysis (System 2 Thinking)
        state = self.step_semantic_analysis(state)

        # Step 2: Initial Generation
        state = self.step_generate_solution(state)

        # Step 3: The Repair Loop
        while state.iterations < self.max_retries and state.status != "SOLVED":
            state.iterations += 1
            print(f"🔄 [AlphaFlow] Iteration {state.iterations}/{self.max_retries}")

            # A. Testing
            pass_rate, error_log = self.step_execute_tests(state)

            if pass_rate == 1.0:
                state.status = "SOLVED"
                state.confidence_score = 1.0
                print("✅ [AlphaFlow] Solution Verified!")
                break

            # B. Root Cause Analysis
            root_cause = self.step_analyze_failure(state, error_log)
            print(f"🧐 [Analysis] {root_cause[:100]}...")

            # C. Targeted Repair
            state = self.step_apply_fix(state, root_cause, error_log)

        return self._finalize_result(state)

    # --- 3. Flow Steps (The "Nodes") ---

    def step_semantic_analysis(self, state: FlowState) -> FlowState:
        """Extracts hard constraints and edge cases."""
        print("🧠 [Analysis] Extracting Constraints via Registry...")
        prompt = self.prompts.semantic_analysis(state.problem_desc)
        state.constraints = self.llm.complete(
            prompt, system_prompt="You are an expert algorithm analyst."
        )
        return state

    def step_generate_solution(self, state: FlowState) -> FlowState:
        """Generates code based on constraints."""
        print("✍️ [Generator] Drafting initial solution...")
        prompt = self.prompts.generate_solution(state.problem_desc, state.constraints)
        raw_code = self.llm.complete(
            prompt,
            system_prompt="You are a senior software engineer. Return only code that uses stdin/stdout.",
        )
        state.current_code = self._clean_markdown(raw_code)
        return state

    def step_execute_tests(self, state: FlowState) -> tuple[float, str]:
        """Runs the code in the Sandbox against provided tests."""
        print("⚡ [Runtime] Executing tests in Sandbox...")
        if not state.tests:
            return 0.0, "No tests provided to verify solution."

        pass_rate, error_log = self.sandbox.run_tests(state.current_code, state.tests)
        state.confidence_score = pass_rate

        return pass_rate, error_log

    def step_analyze_failure(self, state: FlowState, error_log: str) -> str:
        """The 'Reasoning' Step."""
        print("🕵️ [Debugger] Analyzing Root Cause...")
        prompt = self.prompts.analyze_failure(state.current_code, error_log, state.problem_desc)
        return str(
            self.llm.complete(prompt, system_prompt="You are a world-class debugging agent.")
        )

    def step_apply_fix(self, state: FlowState, root_cause: str, error_log: str) -> FlowState:
        """Writes the patch based on the analysis."""
        print("🔧 [Repair] Applying fix...")
        prompt = self.prompts.targeted_repair(state.current_code, root_cause)
        raw_code = self.llm.complete(prompt, system_prompt="You are a senior software engineer.")
        state.current_code = self._clean_markdown(raw_code)

        state.history.append({"iter": state.iterations, "cause": root_cause, "error": error_log})
        return state

    def _clean_markdown(self, text: str) -> str:
        """Helper to strip markdown ticks."""
        return str(self.llm.extract_code(text))

    def _finalize_result(self, state: FlowState) -> dict:
        """Formatting for the Leaderboard."""
        return {
            "solution": state.current_code,
            "status": state.status,
            "metrics": {
                "iterations": state.iterations,
                "confidence": state.confidence_score,
                "flow_depth": len(state.history),
            },
        }


# --- 4. Execution Demo ---
if __name__ == "__main__":
    agent = AlphaRepairAgent()
    problem = "Write a function to double an integer, but return 0 for negatives."
    result = agent.run_flow(problem)

    print("\n--- 🏆 Leaderboard Data ---")
    print(result["metrics"])
