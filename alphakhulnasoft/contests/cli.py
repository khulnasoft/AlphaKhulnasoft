"""Command-line interface for the contests package.

Subcommands:
  load    -- inspect a local Code Contests JSONL
  solve   -- generate + repair a solution for one problem (needs an API key)
  bench   -- run pass@k / novel_pass@k over a split (needs an API key)

CPU-only by default; compiled-language support is auto-detected from PATH.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m alphakhulnasoft.contests")
    sub = parser.add_subparsers(dest="command", required=True)

    p_load = sub.add_parser("load", help="List problems in a JSONL dataset.")
    p_load.add_argument("--dataset", default=None, help="Local JSONL path.")
    p_load.add_argument(
        "--hf-dataset", default=None, help="Hugging Face dataset id (e.g. code_contests)."
    )
    p_load.add_argument("--split", default="train")
    p_load.add_argument("--limit", type=int, default=None)

    p_ceiling = sub.add_parser(
        "ceiling", help="Compute the 'copying' ceiling from reference solutions (no LLM)."
    )
    p_ceiling.add_argument("--dataset", default=None, help="Local JSONL path.")
    p_ceiling.add_argument("--hf-dataset", default=None, help="Hugging Face dataset id.")
    p_ceiling.add_argument("--split", default="train")
    p_ceiling.add_argument("--language", default="py")
    p_ceiling.add_argument(
        "--max-refs", type=int, default=5, help="Max references graded per problem"
    )
    p_ceiling.add_argument("--limit", type=int, default=None)

    p_solve = sub.add_parser("solve", help="Solve a single problem by id.")
    p_solve.add_argument("--dataset", default=None, help="Local JSONL path.")
    p_solve.add_argument("--hf-dataset", default=None, help="Hugging Face dataset id.")
    p_solve.add_argument("--split", default="train")
    p_solve.add_argument("--limit", type=int, default=None)
    p_solve.add_argument("--problem-id", required=True)
    p_solve.add_argument("--language", default="py")
    p_solve.add_argument(
        "--model", default=None, help="LLM model id (e.g. gemini/gemini-1.5-flash)"
    )
    p_solve.add_argument("--rpm", type=int, default=None, help="Max LLM requests/minute (throttle)")
    p_solve.add_argument("--n-samples", type=int, default=5)
    p_solve.add_argument("--device", default="cpu")

    p_bench = sub.add_parser("bench", help="Benchmark pass@k / novel_pass@k.")
    p_bench.add_argument("--dataset", default=None, help="Local JSONL path.")
    p_bench.add_argument("--hf-dataset", default=None, help="Hugging Face dataset id.")
    p_bench.add_argument("--split", default="train")
    p_bench.add_argument("--language", default="py")
    p_bench.add_argument("--n-samples", type=int, default=10)
    p_bench.add_argument("--k", type=int, default=1)
    p_bench.add_argument(
        "--model", default=None, help="LLM model id (e.g. gemini/gemini-1.5-flash)"
    )
    p_bench.add_argument("--rpm", type=int, default=None, help="Max LLM requests/minute (throttle)")
    p_bench.add_argument("--device", default="cpu")
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--format", choices=["json"], default="json")
    p_bench.add_argument("--output", default=None)
    p_bench.add_argument("--publish", default=None, help="HF dataset repo id to upload the report")
    p_bench.add_argument("--hf-token", default=None, help="HF token (else HF_TOKEN env)")
    p_bench.add_argument(
        "--reference-ceiling",
        action="store_true",
        help="Also compute the 'copying' ceiling from reference solutions",
    )
    return parser


def _load_problems(args: argparse.Namespace):
    from .loader import load_huggingface, load_local

    if args.hf_dataset:
        return load_huggingface(name=args.hf_dataset, split=args.split, limit=args.limit)
    if not args.dataset:
        raise SystemExit("Provide --dataset (local JSONL) or --hf-dataset (Hugging Face).")
    problems = load_local(args.dataset)
    if args.limit:
        problems = problems[: args.limit]
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from .benchmark import run_benchmark
    from .generator import ContestAgent
    from .llm_shim import make_llm

    if args.command == "load":
        problems = _load_problems(args)
        print(json.dumps([p.problem_id for p in problems], indent=2))
        return 0

    if args.command == "ceiling":
        from .benchmark import reference_ceiling

        problems = _load_problems(args)
        print(json.dumps(reference_ceiling(problems, args.language, args.max_refs), indent=2))
        return 0

    if args.command in ("solve", "bench"):
        problems = _load_problems(args)
        if args.command == "solve":
            problem = next((p for p in problems if p.problem_id == args.problem_id), None)
            if problem is None:
                print(f"Problem {args.problem_id!r} not found.", file=sys.stderr)
                return 2
            agent = ContestAgent(llm=make_llm(model=args.model, rpm=args.rpm))
            cand = agent.solve(problem, args.language, n_samples=args.n_samples)
            print(cand.code)
            print(
                f"# status={cand.status} visible_pass={cand.grade.all_passed() if cand.grade else False}"
            )
            return 0

        agent = ContestAgent(llm=make_llm(model=args.model, rpm=args.rpm))
        report = run_benchmark(
            problems,
            agent,
            language=args.language,
            n_samples=args.n_samples,
            k=args.k,
            publish_repo=args.publish,
            hf_token=args.hf_token,
            include_reference_ceiling=args.reference_ceiling,
        )
        payload = report.as_dict()
        text = json.dumps(payload, indent=2)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            print(text)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
