"""Benchmark runner for formal proof generation.

Mirrors ``alphakhulnasoft.benchmark.run_benchmark`` but drives the
``ProofRepairAgent`` over a dataset of theorems and scores results with
``ProofEvaluator``.
"""

import json
import os
import time

from .proof_evaluator import ProofEvaluator
from .proof_generator import ProofRepairAgent


def _load_theorems(theorem_dataset) -> list[dict]:
    """Normalize a dataset argument into a list of theorem dicts.

    Accepts either a list of dicts or a path to a JSONL file.
    """
    if isinstance(theorem_dataset, str):
        if not os.path.exists(theorem_dataset):
            raise FileNotFoundError(f"Theorem dataset not found: {theorem_dataset}")
        items = []
        with open(theorem_dataset) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if isinstance(theorem_dataset, list):
        return theorem_dataset

    raise TypeError("theorem_dataset must be a list or a path to a .jsonl file")


def run_proof_benchmark(
    theorem_dataset: list[dict] | str,
    output_file: str | None = None,
    limit: int | None = None,
) -> dict:
    """Benchmark proof generation over a dataset of theorems.

    Dataset item schema:
        {
          "theorem": "...",
          "hints": "...",            # optional
          "expected_technique": "..." # optional
        }

    Returns:
        dict with keys ``results`` (list of per-theorem dicts) and
        ``summary`` (aggregate metrics).
    """
    theorems = _load_theorems(theorem_dataset)
    if limit is not None:
        theorems = theorems[:limit]

    agent = ProofRepairAgent()
    evaluator = ProofEvaluator()

    print(f"🔥 Starting AlphaKhulnasoft Proof Benchmark on {len(theorems)} theorems...")
    print(f"   Model: {agent.model} (via litellm)")
    print("   Strategy: Flow Engineering (Proof)\n")

    results = []

    for i, item in enumerate(theorems):
        theorem = item.get("theorem") or item.get("statement") or ""
        hints = item.get("hints", "")
        print(f"⚔️  Theorem {i + 1}: {theorem[:60]}{'...' if len(theorem) > 60 else ''}")

        start_time = time.time()
        proof_data = agent.run_proof_flow(theorem, proof_hints=hints)
        duration = time.time() - start_time

        is_proven = proof_data["status"] == "PROVEN"

        score = evaluator.score_proof(
            proof_data["nexus_code"],
            proof_data["is_valid"],
            proof_data["metrics"]["iterations"],
        )
        evaluator.add_result(theorem[:40], proof_data)

        entry = {
            "id": item.get("id", i + 1),
            "theorem": theorem,
            "nexus_code": proof_data["nexus_code"],
            "prose": proof_data["prose"],
            "status": proof_data["status"],
            "is_valid": proof_data["is_valid"],
            "iterations": proof_data["metrics"]["iterations"],
            "depth": score["proof_depth"],
            "conciseness": score["conciseness"],
            "duration": round(duration, 2),
            "cost_score": evaluator.calculate_efficiency_score(
                is_proven, proof_data["metrics"]["iterations"]
            ),
            "score": score,
        }
        results.append(entry)

        icon = "✅" if is_proven else "❌"
        print(
            f"   {icon} Result: {proof_data['status']} | "
            f"Iters: {entry['iterations']} | Time: {entry['duration']}s\n"
        )

    evaluator.print_leaderboard(results)

    total = len(results)
    proven = sum(1 for r in results if r["is_valid"])
    avg_iters = (sum(r["iterations"] for r in results) / total) if total else 0
    summary = {
        "total": total,
        "proven": proven,
        "proof_at_1": (proven / total) if total else 0,
        "avg_iterations": round(avg_iters, 2),
    }

    if output_file:
        with open(output_file, "w") as f:
            json.dump({"results": results, "summary": summary}, f, indent=2)
        print(f"💾 Proof benchmark results saved to {output_file}")

    return {"results": results, "summary": summary}


if __name__ == "__main__":
    import sys

    path = "data/theorems_easy.jsonl" if len(sys.argv) < 2 else sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "results_proofs_latest.json"

    if os.path.exists(path):
        run_proof_benchmark(path, output_file=out)
    else:
        print(f"⚠️ Dataset '{path}' not found. Falling back to inline sample...\n")
        sample = [
            {
                "theorem": "For all n ≥ 0, the sum of the first n naturals is n(n+1)/2",
                "hints": "Use mathematical induction.",
                "expected_technique": "induction",
            }
        ]
        run_proof_benchmark(sample, output_file=out)
