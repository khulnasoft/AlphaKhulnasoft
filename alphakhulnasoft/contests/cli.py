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
    p_load.add_argument("--dataset", required=True)
    p_load.add_argument("--limit", type=int, default=None)

    p_solve = sub.add_parser("solve", help="Solve a single problem by id.")
    p_solve.add_argument("--dataset", required=True)
    p_solve.add_argument("--problem-id", required=True)
    p_solve.add_argument("--language", default="py")
    p_solve.add_argument("--n-samples", type=int, default=5)
    p_solve.add_argument("--device", default="cpu")

    p_bench = sub.add_parser("bench", help="Benchmark pass@k / novel_pass@k.")
    p_bench.add_argument("--dataset", required=True)
    p_bench.add_argument("--language", default="py")
    p_bench.add_argument("--n-samples", type=int, default=10)
    p_bench.add_argument("--k", type=int, default=1)
    p_bench.add_argument("--device", default="cpu")
    p_bench.add_argument("--limit", type=int, default=None)
    p_bench.add_argument("--format", choices=["json", "csv"], default="json")
    p_bench.add_argument("--output", default=None)
    p_bench.add_argument("--publish", default=None, help="HF dataset repo id to upload the report")
    p_bench.add_argument("--hf-token", default=None, help="HF token (else HF_TOKEN env)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    from .benchmark import run_benchmark
    from .generator import ContestAgent
    from .llm_shim import make_llm
    from .loader import load_local

    if args.command == "load":
        problems = load_local(args.dataset)
        if args.limit:
            problems = problems[: args.limit]
        print(json.dumps([p.problem_id for p in problems], indent=2))
        return 0

    if args.command in ("solve", "bench"):
        problems = load_local(args.dataset)
        if args.limit:
            problems = problems[: args.limit]
        if args.command == "solve":
            problem = next((p for p in problems if p.problem_id == args.problem_id), None)
            if problem is None:
                print(f"Problem {args.problem_id!r} not found.", file=sys.stderr)
                return 2
            agent = ContestAgent(llm=make_llm())
            cand = agent.solve(problem, args.language, n_samples=args.n_samples)
            print(cand.code)
            print(
                f"# status={cand.status} visible_pass={cand.grade.all_passed() if cand.grade else False}"
            )
            return 0

        agent = ContestAgent(llm=make_llm())
        report = run_benchmark(
            problems,
            agent,
            language=args.language,
            n_samples=args.n_samples,
            k=args.k,
            publish_repo=args.publish,
            hf_token=args.hf_token,
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
