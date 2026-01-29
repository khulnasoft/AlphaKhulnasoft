import os
import time

from .alpha_repair import AlphaRepairAgent
from .data_loader import DataLoader
from .evaluator import Evaluator
from .prompts import PromptRegistry


def run_benchmark(dataset_path: str | None = None, limit: int = 5):
    """
    Orchestrates the AlphaKhulnasoft v2 Benchmark.
    """
    # 1. Setup
    loader = DataLoader()
    evaluator = Evaluator()

    # Load real data or use mock if path is None
    problems = loader.load_problems(dataset_path) if dataset_path else [loader.get_mock_problem()]

    # Add another mock problem if it's the default run
    if not dataset_path:
        problems.append(
            {
                "id": "mock-002",
                "title": "Sum of Evens",
                "description": "Write a function `solve(nums)` that returns the sum of all even numbers in a list.",
                "tests": [
                    {"input": "[1, 2, 3, 4]", "expected": "6"},
                    {"input": "[1, 3, 5]", "expected": "0"},
                    {"input": "[]", "expected": "0"},
                ],
            }
        )

    problems = problems[:limit]

    print(f"🔥 Starting AlphaKhulnasoft v2 Benchmark on {len(problems)} problems...")
    print("   Model: GPT-4o (via litellm)")
    print("   Strategy: Flow Engineering v2\n")

    results = []

    # 2. The Contest Loop
    for i, problem in enumerate(problems):
        title = problem.get("title", "Unknown")
        print(f"⚔️  Problem {i + 1}: {title}")

        # Initialize the Agent (injecting the Prompts)
        agent = AlphaRepairAgent(model_name="gpt-4o", prompt_registry=PromptRegistry)

        start_time = time.time()

        # --- RUN THE FLOW ---
        # Pass tests directly from the problem definition
        solution_data = agent.run_flow(problem["description"], tests=problem.get("tests"))
        # --------------------

        duration = time.time() - start_time

        # 3. Evaluation (Adjudication)
        is_solved = solution_data["status"] == "SOLVED"

        metrics = {
            "id": problem.get("id", str(i + 1)),
            "pass": is_solved,
            "iterations": solution_data["metrics"]["iterations"],
            "confidence": solution_data["metrics"]["confidence"],
            "duration": round(duration, 2),
            "cost_score": evaluator.calculate_efficiency_score(
                is_solved, solution_data["metrics"]["iterations"]
            ),
        }

        results.append(metrics)

        # Live Feedback
        icon = "✅" if is_solved else "❌"
        print(
            f"   {icon} Result: {solution_data['status']} | Iters: {metrics['iterations']} | Time: {metrics['duration']}s\n"
        )

    # 4. Final Leaderboard
    evaluator.print_leaderboard(results)

    # 5. Save Results
    import datetime
    import json

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"💾 Benchmark results saved to {filename}")
    return filename


if __name__ == "__main__":
    import sys

    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️ Warning: No API keys found in environment. LLM calls will fail.")

    # Handle dataset path from CLI
    dataset = "data/hard_mode.jsonl" if len(sys.argv) < 2 else sys.argv[1]

    if os.path.exists(dataset):
        print(f"📊 Running benchmark on dataset: {dataset}")
        run_benchmark(dataset_path=dataset, limit=10)
    else:
        print(f"⚠️ Dataset '{dataset}' not found.")
        print("💡 Pro Tip: Run 'python3 -m alphakhulnasoft.dataset_gen' to create one.")
        print("   Falling back to Mock Problems for demonstration...\n")
        run_benchmark(dataset_path=None, limit=2)
