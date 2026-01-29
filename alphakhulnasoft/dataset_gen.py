import json
import os

from .llm import LLMProvider


def generate_hard_problems(count=5, output_file="data/hard_mode.jsonl"):
    """
    Uses the LLM to bootstrap a high-quality, difficult test set.
    """
    llm = LLMProvider(model="gpt-4o")
    problems = []

    print(f"🌋 Generating {count} Hard Algorithmic Problems...")

    os.makedirs("data", exist_ok=True)

    for i in range(count):
        print(f"   Drafting problem {i + 1}/{count}...")
        prompt = """
        Generate a DIFFICULT coding interview problem (Python). 
        The problem must be something that a standard LLM might struggle to solve perfectly on the first try (e.g., complex DP, Graph traversal with edge cases, or advanced Data Structures).

        Constraints:
        1. Problem must be solvable in Python.
        2. Must use stdin/stdout (using `input()` and `print()`).
        3. Must have precisely defined input/output formats.

        OUTPUT JSON ONLY:
        {
            "id": "gen_001",
            "title": "Problem Title",
            "description": "Full problem description including I/O format...",
            "tests": [
                {"input": "test_input_1", "expected": "test_output_1"},
                {"input": "test_input_2", "expected": "test_output_2"}
            ]
        }
        """

        try:
            raw = llm.complete(
                prompt, system_prompt="You are a competitive programming task creator."
            )
            # JSON extraction
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            problem = json.loads(json_str)
            problem["id"] = f"gen_{i + 1:03d}"
            problems.append(problem)
        except Exception as e:
            print(f"   ⚠️ Failed to parse problem {i + 1}: {e}")

    # Save to JSONL
    with open(output_file, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")

    print(f"✅ Saved {len(problems)} problems to {output_file}")


if __name__ == "__main__":
    generate_hard_problems()
