import json
import os


class DataLoader:
    """Loads problems from various coding datasets."""

    def __init__(self, dataset_name: str = "codecontests"):
        self.dataset_name = dataset_name

    def load_problems(self, path: str) -> list[dict]:
        """Loads problems from a local JSONL file."""
        problems = []
        if not path or not os.path.exists(path):
            print(f"Warning: Path {path} does not exist or not provided.")
            return []

        with open(path) as f:
            for line in f:
                problems.append(json.loads(line))
        return problems

    def load_from_hf(self, dataset_name: str, split: str = "test") -> list[dict]:
        """
        Loads a coding dataset from Hugging Face Hub.
        Supports common formats like Humaneval.
        """
        from datasets import load_dataset

        print(f"📥 Fetching dataset '{dataset_name}' [{split}] from Hugging Face...")
        try:
            ds = load_dataset(dataset_name, split=split)
            problems = []
            for item in ds:
                # Standardizing format: Humaneval/MBPP usually has 'prompt' or 'text'
                problems.append(
                    {
                        "id": item.get("task_id") or item.get("id") or str(len(problems)),
                        "title": item.get("entry_point") or "HF Problem",
                        "description": item.get("prompt") or item.get("text") or "",
                        "tests": self._parse_hf_tests(item),
                    }
                )
            return problems
        except Exception as e:
            print(f"Error loading from HF: {e}")
            return []

    def _parse_hf_tests(self, item: dict) -> list[dict]:
        """Heuristic to extract tests from HF dataset items."""
        # This is a basic placeholder; different datasets require different parsing
        if "test" in item:
            # Often datasets have a 'test' string containing assert statements
            return [{"input": "", "expected": item["test"]}]
        return []

    def get_mock_problem(self) -> dict:
        """Returns a dummy problem for testing."""
        return {
            "id": "mock-001",
            "title": "Double and Positives",
            "description": "Write a function `solve(n)` to double an integer, but return 0 for negatives.",
            "tests": [
                {"input": "2", "expected": "4"},
                {"input": "-5", "expected": "0"},
                {"input": "0", "expected": "0"},
            ],
        }
