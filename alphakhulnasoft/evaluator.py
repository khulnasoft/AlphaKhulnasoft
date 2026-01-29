class Evaluator:
    """Calculates metrics for the AI Code Fixer."""

    def __init__(self):
        self.results = []

    def add_result(self, problem_name: str, result: dict):
        """Stores a result for evaluation."""
        self.results.append(
            {
                "problem": problem_name,
                "solved": result["status"] == "SOLVED",
                "iterations": result["metrics"]["iterations"],
                "depth": result["metrics"]["flow_depth"],
            }
        )

    def calculate_efficiency_score(self, solved: bool, iterations: int) -> float:
        """Calculates efficiency score for a single run."""
        if not solved:
            return 0.0
        # 1.0 if solved in 1 iteration, decreases as iterations increase
        return round(1.0 / iterations, 3)

    def print_leaderboard(self, results: list[dict]):
        """Prints a professional leaderboard based on benchmark results."""
        print("\n" + "═" * 60)
        print("🏆 ALPHAKHULNASOFT v2 LEADERBOARD 🏆")
        print("═" * 60)
        print(f"{'ID':<10} | {'Status':<10} | {'Iters':<6} | {'Time':<6} | {'Efficiency'}")
        print("─" * 60)

        total_solved = 0
        total_iters = 0
        total_time = 0.0

        for res in results:
            status = "✅ PASS" if res["pass"] else "❌ FAIL"
            print(
                f"{res['id']:<10} | {status:<10} | {res['iterations']:<6} | {res['duration']:<6} | {res['cost_score']}"
            )

            if res["pass"]:
                total_solved += 1
            total_iters += res["iterations"]
            total_time += res["duration"]

        print("═" * 60)
        pass_at_1 = (total_solved / len(results)) if results else 0
        print(f"OVERALL PASS@1: {pass_at_1:.2%}")
        print(f"AVG ITERATIONS: {total_iters / len(results) if results else 0:.2f}")
        print(f"TOTAL DURATION: {total_time:.2f}s")
        print("═" * 60 + "\n")
