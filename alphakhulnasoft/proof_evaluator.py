"""Scoring and metrics for formal proofs.

Mirrors ``alphakhulnasoft.evaluator.Evaluator`` but with proof-specific
metrics (validity, conciseness, repair efficiency, proof depth).
"""


class ProofEvaluator:
    """Calculates metrics for the AI Proof Generator."""

    def __init__(self):
        self.results = []

    def add_result(self, theorem_name: str, result: dict):
        """Stores a result for evaluation."""
        self.results.append(
            {
                "theorem": theorem_name,
                "proven": result.get("status") == "PROVEN",
                "iterations": result["metrics"]["iterations"],
                "depth": result["metrics"]["proof_depth"],
                "is_valid": result.get("is_valid", False),
            }
        )

    def score_proof(self, nexus_code: str, is_valid: bool, iterations: int) -> dict:
        """Score a single proof.

        Metrics:
        - ``is_valid``: boolean (binary weight on whether the proof checks).
        - ``conciseness``: number of lines of Nexus code.
        - ``iterations_to_valid``: repair iterations needed to reach validity.
        - ``proof_depth``: estimated nesting depth of tactics/induction/cases.
        """
        return {
            "is_valid": is_valid,
            "conciseness": len(nexus_code.split("\n")) if nexus_code else 0,
            "iterations_to_valid": iterations,
            "proof_depth": self._estimate_proof_depth(nexus_code),
        }

    def _estimate_proof_depth(self, nexus_code: str) -> int:
        """Count nesting depth of tactics/induction/cases."""
        if not nexus_code:
            return 0

        depth = 0
        max_depth = 0
        indent_unit = 2

        for raw_line in nexus_code.split("\n"):
            line = raw_line.rstrip()
            if not line.strip() or line.strip().startswith("#"):
                continue
            indent = len(raw_line) - len(raw_line.lstrip(" "))
            current_depth = indent // indent_unit if indent_unit else 0
            depth = current_depth
            max_depth = max(max_depth, depth)
        return max_depth

    def calculate_efficiency_score(self, proven: bool, iterations: int) -> float:
        """Calculates efficiency score for a single run."""
        if not proven:
            return 0.0
        return round(1.0 / max(iterations, 1), 3)

    def print_leaderboard(self, results: list[dict]):
        """Prints a leaderboard based on benchmark results."""
        print("\n" + "═" * 60)
        print("🏆 ALPHAKHULNASOFT PROOF LEADERBOARD 🏆")
        print("═" * 60)
        print(f"{'ID':<10} | {'Status':<10} | {'Iters':<6} | {'Depth':<6} | {'Efficiency'}")
        print("─" * 60)

        total_proven = 0
        total_iters = 0

        for i, res in enumerate(results):
            status = "✅ PROVEN" if res.get("proven") else "❌ FAILED"
            print(
                f"{res.get('id', i + 1):<10} | {status:<10} | "
                f"{res['iterations']:<6} | {res.get('depth', 0):<6} | {res.get('cost_score', 0)}"
            )
            if res.get("proven"):
                total_proven += 1
            total_iters += res["iterations"]

        print("═" * 60)
        pass_rate = (total_proven / len(results)) if results else 0
        print(f"OVERALL PROOF@1: {pass_rate:.2%}")
        print(f"AVG ITERATIONS: {total_iters / len(results) if results else 0:.2f}")
        print("═" * 60 + "\n")
