"""Proof Generation Agent for formal mathematical proofs.

This module provides the ``ProofRepairAgent``, which specializes in generating
formal Nexus proofs paired with natural language explanations. It reuses the
Flow Engineering loop from ``alphakhulnasoft.alpha_repair.AlphaRepairAgent``
(Analyze -> Generate -> Verify -> Root Cause -> Repair) and adds a final
prose-synthesis step.
"""

import uuid
from dataclasses import dataclass, field
from typing import Literal

from .alpha_repair import AlphaRepairAgent
from .config import AlphaConfig
from .logging_config import get_logger
from .proof_prompts import ProofPromptRegistry
from .proof_sandbox import ProofSandbox

logger = get_logger(__name__)


# --- 1. Proof State Management ---
@dataclass
class ProofState:
    """Tracks the lifecycle of a formal proof generation task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    theorem_statement: str = ""
    proof_hints: str = ""  # Optional hints about proof technique
    expected_technique: str = ""  # Optional expected technique label
    proof_analysis: str = ""  # Structured analysis output
    current_nexus_code: str = ""  # The formal Nexus proof
    current_prose: str = ""  # Natural language explanation
    verification_status: Literal["PENDING", "VALID", "INVALID"] = "PENDING"
    verification_error: str = ""  # Type-checker error message (if any)
    status: Literal["PENDING", "PROVEN", "FAILED", "REPAIRING"] = "PENDING"
    iterations: int = 0
    confidence_score: float = 0.0  # 0.0 to 1.0 (1.0 = valid proof)
    history: list[dict] = field(default_factory=list)  # Traceability of repairs


# --- 2. Proof Repair Agent ---
class ProofRepairAgent(AlphaRepairAgent):
    """Generates formal proofs in Nexus with accompanying natural language explanations.

    Workflow (mirrors ``AlphaRepairAgent`` but is proof-aware):
        1. Analyze Proof Requirements: extract proof technique, base cases, invariants.
        2. Generate Initial Nexus Code: draft formal proof using analysis.
        3. Verify Proof: attempt type-checking via ``ProofSandbox``.
        4. Root Cause Analysis: diagnose any verification errors.
        5. Iterative Repair: fix proof until valid or max retries reached.
        6. Generate Prose Explanation: translate formal proof to natural language.
    """

    def __init__(
        self,
        config: AlphaConfig | None = None,
        prompt_registry=ProofPromptRegistry,
        model_name: str | None = None,
        max_retries: int | None = None,
        llm_max_retries: int | None = None,
        sandbox_timeout: int | None = None,
        nexus_bin: str = "nexus",
        nexus_timeout: int = 30,
    ):
        super().__init__(
            config=config,
            prompt_registry=prompt_registry,
            model_name=model_name,
            max_retries=max_retries,
            llm_max_retries=llm_max_retries,
            sandbox_timeout=sandbox_timeout,
        )
        self.proof_sandbox = ProofSandbox(nexus_bin=nexus_bin, timeout_seconds=nexus_timeout)

    def run_proof_flow(self, theorem_statement: str, proof_hints: str = "") -> dict:
        """Execute the proof generation flow.

        Main entry point for proof generation. Orchestrates the five-step
        repair loop followed by prose explanation generation.

        Args:
            theorem_statement: The mathematical theorem to prove.
            proof_hints: Optional hints about proof structure or technique.

        Returns:
            Dictionary with keys:
                - nexus_code: The formal Nexus proof
                - prose: Natural language explanation
                - status: "PROVEN" or "FAILED"
                - is_valid: bool
                - theorem: the input theorem
                - metrics: dict with iterations, confidence, proof_depth, verification_status
                - error: error string or None
        """
        state = ProofState(theorem_statement=theorem_statement, proof_hints=proof_hints)

        try:
            logger.info(f"Starting Proof Flow for Theorem ID: {state.id}")

            # Step 1: Analyze proof requirements
            state = self.step_analyze_proof_requirements(state)
            if not state.proof_analysis:
                raise RuntimeError("Proof analysis failed: no analysis generated")

            # Step 2: Generate initial Nexus proof
            state = self.step_generate_nexus_proof(state)
            if not state.current_nexus_code:
                raise RuntimeError("Nexus proof generation failed: no code produced")

            # Step 3: The Repair Loop
            while state.iterations < self.max_retries and state.status != "PROVEN":
                state.iterations += 1
                logger.info(f"Verification iteration {state.iterations}/{self.max_retries}")

                is_valid, error_log = self.step_verify_proof(state)

                if is_valid:
                    state.status = "PROVEN"
                    state.verification_status = "VALID"
                    state.confidence_score = 1.0
                    logger.info("Proof Verified!")
                    break

                root_cause = self.step_analyze_proof_error(state, error_log)
                logger.debug(f"Root cause identified: {root_cause[:100]}...")

                state = self.step_repair_nexus_proof(state, root_cause, error_log)
                if state.status == "FAILED":
                    break

            if state.status != "PROVEN":
                state.status = "FAILED"
                state.verification_status = "INVALID"

            # Step 4: Generate prose explanation (even for partial proofs)
            if state.current_nexus_code:
                state = self.step_generate_prose_explanation(state)

        except Exception as e:
            state.status = "FAILED"
            state.verification_error = f"Fatal Error: {str(e)}"
            logger.error(f"Proof flow terminated: {e}", exc_info=True)

        return self._finalize_proof_result(state)

    # --- 3. Flow Steps (The Proof Nodes) ---

    def step_analyze_proof_requirements(self, state: ProofState) -> ProofState:
        """Analyze theorem to extract proof strategy."""
        logger.info("Analyzing proof requirements...")
        try:
            prompt = self.prompts.analyze_proof_requirements(
                state.theorem_statement, state.proof_hints
            )
            state.proof_analysis = self.llm.complete(
                prompt,
                system_prompt="You are an expert mathematician and formal verification specialist.",
            )
            logger.debug("Proof requirements analysis completed successfully")
        except Exception as e:
            logger.error(f"Proof analysis failed: {e}", exc_info=True)
            raise

        return state

    def step_generate_nexus_proof(self, state: ProofState) -> ProofState:
        """Generate initial Nexus proof from analysis."""
        logger.info("Generating Nexus proof...")
        try:
            prompt = self.prompts.generate_nexus_proof(
                state.theorem_statement, state.proof_analysis
            )
            raw_code = self.llm.complete(
                prompt,
                system_prompt="You are a Nexus proof assistant. Write syntactically valid, type-checking formal proofs.",
            )
            state.current_nexus_code = self._clean_proof_markdown(raw_code)
            logger.debug("Nexus proof generated successfully")
        except Exception as e:
            logger.error(f"Nexus proof generation failed: {e}", exc_info=True)
            raise

        return state

    def step_verify_proof(self, state: ProofState) -> tuple[bool, str]:
        """Verify Nexus proof via the ProofSandbox."""
        logger.info("Verifying proof...")
        is_valid, error_log = self.proof_sandbox.verify_nexus_proof(state.current_nexus_code)
        state.verification_status = "VALID" if is_valid else "INVALID"
        if not is_valid:
            state.verification_error = error_log
        logger.info(f"Proof valid: {is_valid}")
        return is_valid, error_log

    def step_analyze_proof_error(self, state: ProofState, error_log: str) -> str:
        """Diagnose proof verification failure."""
        logger.info("Analyzing proof error...")
        try:
            prompt = self.prompts.analyze_proof_error(
                state.theorem_statement,
                state.current_nexus_code,
                error_log,
                state.iterations,
            )
            root_cause = self.llm.complete(
                prompt,
                system_prompt="You are a debugging expert specializing in formal proofs.",
            )
            logger.debug(f"Root cause identified: {root_cause[:100]}...")
            return root_cause
        except Exception as e:
            logger.error(f"Error analysis failed: {e}", exc_info=True)
            raise

    def step_repair_nexus_proof(
        self, state: ProofState, root_cause_analysis: str, error_log: str
    ) -> ProofState:
        """Repair Nexus proof based on root cause diagnosis."""
        logger.info("Repairing Nexus proof...")
        try:
            prompt = self.prompts.repair_nexus_proof(
                state.theorem_statement, state.current_nexus_code, root_cause_analysis
            )
            raw_code = self.llm.complete(
                prompt,
                system_prompt="You are a Nexus proof expert. Generate minimal, targeted repairs.",
            )
            cleaned_code = self._clean_proof_markdown(raw_code)

            if not cleaned_code.strip():
                state.verification_error = (
                    f"Repair iteration {state.iterations} generated empty code"
                )
                state.status = "FAILED"
                logger.warning(f"Repair iteration {state.iterations} generated empty code")
                return state

            state.current_nexus_code = cleaned_code
            state.history.append(
                {
                    "iter": state.iterations,
                    "cause": root_cause_analysis,
                    "error": error_log,
                }
            )
            logger.debug(f"Repair applied at iteration {state.iterations}")
        except Exception as e:
            state.verification_error = f"Repair failed: {str(e)}"
            state.status = "FAILED"
            logger.error(f"Repair failed: {e}", exc_info=True)
            raise

        return state

    def step_generate_prose_explanation(self, state: ProofState) -> ProofState:
        """Generate natural language proof explanation."""
        logger.info("Generating natural language proof explanation...")
        try:
            prompt = self.prompts.generate_prose_explanation(
                state.theorem_statement, state.current_nexus_code, state.proof_analysis
            )
            state.current_prose = self.llm.complete(
                prompt,
                system_prompt="You are a mathematical writer. Produce clear, rigorous, textbook-quality proofs.",
            )
            logger.debug("Prose explanation generated successfully")
        except Exception as e:
            logger.error(f"Prose generation failed: {e}", exc_info=True)
            # Don't fail the entire flow if prose generation fails
            state.current_prose = f"[Prose generation failed: {str(e)}. See Nexus proof above.]"

        return state

    # --- 4. Utilities ---

    def _clean_proof_markdown(self, text: str) -> str:
        """Extract Nexus proof from markdown or raw text.

        Extends the base ``_clean_markdown`` with support for Nexus-fenced
        code blocks (```nexus).
        """
        if "```nexus" in text:
            return text.split("```nexus")[1].split("```")[0].strip()
        return self._clean_markdown(text)

    def _finalize_proof_result(self, state: ProofState) -> dict:
        """Format proof generation result for output."""
        return {
            "proof_id": state.id,
            "nexus_code": state.current_nexus_code,
            "prose": state.current_prose,
            "status": state.status,
            "is_valid": state.verification_status == "VALID",
            "theorem": state.theorem_statement,
            "metrics": {
                "iterations": state.iterations,
                "confidence": state.confidence_score,
                "proof_depth": len(state.history),
                "verification_status": state.verification_status,
            },
            "error": state.verification_error or None,
        }


# --- 5. Demo / Entry Point ---
if __name__ == "__main__":
    agent = ProofRepairAgent()

    theorem = "For all natural numbers n, the sum of integers from 1 to n equals n * (n + 1) / 2"
    hints = (
        "Use mathematical induction. Base case: n = 0. "
        "Inductive step: assume true for n, prove for n+1."
    )

    result = agent.run_proof_flow(theorem, proof_hints=hints)

    print("\n" + "=" * 80)
    print("🏆 PROOF GENERATION RESULT")
    print("=" * 80)
    print(f"\nTheorem: {result['theorem']}")
    print(f"\nStatus: {result['status']}")
    print(f"Valid: {result['is_valid']}")
    print(f"Iterations: {result['metrics']['iterations']}")
    print("\n--- Nexus Proof ---")
    print(result["nexus_code"])
    print("\n--- Natural Language Proof ---")
    print(result["prose"])
    if result["error"]:
        print(f"\nError: {result['error']}")
