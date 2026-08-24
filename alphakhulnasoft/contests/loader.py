"""Loading Code Contests problems from local JSONL or Hugging Face.

Reference solutions are loaded into the problem but are only ever accessed
through :func:`get_references`, which the generator must NOT call — keeping the
"solving vs copying" boundary explicit in code, not just in docs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .problem import (
    ContestProblem,
    IOFormat,
    ProblemSource,
    ReferenceSolution,
    TestCase,
)


def _as_str(value: Any, default: str = "") -> str:
    return "" if value is None else str(value)


def problem_from_dict(data: dict[str, Any]) -> ContestProblem:
    """Build a validated :class:`ContestProblem` from a raw mapping."""
    if not isinstance(data, dict):
        raise ValueError("Problem payload must be a JSON object.")

    pid = _as_str(data.get("problem_id"))
    if not pid:
        raise ValueError("Missing 'problem_id'.")
    source = _as_str(data.get("source")).lower()
    if source not in ProblemSource.ALL:
        raise ValueError(f"Unknown/empty source for problem {pid!r}: {source!r}")
    statement = _as_str(data.get("statement"))
    if not statement.strip():
        raise ValueError(f"Empty statement for problem {pid!r}.")

    tests = _parse_tests(data.get("tests", []), pid)
    if not tests:
        raise ValueError(f"Problem {pid!r} has no test cases.")

    samples = _parse_samples(data.get("samples", []))
    io_format = _parse_io_format(data.get("io_format", {}))

    refs: list[ReferenceSolution] = []
    for raw in data.get("reference_solutions", []) or []:
        if not isinstance(raw, dict):
            continue
        lang = _as_str(raw.get("language"))
        code = _as_str(raw.get("code"))
        if lang and code:
            refs.append(
                ReferenceSolution(
                    language=lang, code=code, status=_as_str(raw.get("status"), "unknown")
                )
            )

    language = _as_str(data.get("language"), "py") or "py"
    return ContestProblem(
        problem_id=pid,
        source=source,
        statement=statement,
        tests=tests,
        samples=samples,
        io_format=io_format,
        reference_solutions=refs,
        language=language,
    )


def _parse_tests(raw: Any, pid: str) -> list[TestCase]:
    if not isinstance(raw, list):
        raise ValueError(f"'tests' for {pid!r} must be a list.")
    out: list[TestCase] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"tests[{i}] for {pid!r} must be an object.")
        inp = _as_str(item.get("input"))
        exp = _as_str(item.get("output") if "output" in item else item.get("expected"))
        if "output" not in item and "expected" not in item:
            raise ValueError(f"tests[{i}] for {pid!r} missing 'output'/'expected'.")
        hidden = bool(item.get("hidden", False))
        out.append(TestCase(input=inp, expected_output=exp, hidden=hidden))
    return out


def _parse_samples(raw: Any) -> list[tuple[str, str]]:
    if not isinstance(raw, list):
        return []
    out: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append((_as_str(item.get("input")), _as_str(item.get("output"))))
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            out.append((_as_str(item[0]), _as_str(item[1])))
    return out


def _parse_io_format(raw: Any) -> IOFormat:
    if not isinstance(raw, dict):
        return IOFormat()
    return IOFormat(
        input_format=_as_str(raw.get("input_format")),
        output_format=_as_str(raw.get("output_format")),
        constraints=_as_str(raw.get("constraints")),
        time_limit_s=raw.get("time_limit_s"),
        memory_limit_mb=raw.get("memory_limit_mb"),
    )


def load_local(path: str | Path) -> list[ContestProblem]:
    """Load problems from a JSONL file (one problem per line)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    problems: list[ContestProblem] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                problems.append(problem_from_dict(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    if not problems:
        raise ValueError(f"No valid problems found in {path}")
    return problems


def load_huggingface(
    name: str = "code_contests",
    split: str = "train",
    streaming: bool = True,
    limit: int | None = None,
) -> list[ContestProblem]:
    """Load problems from a Hugging Face dataset (optional dependency).

    Requires the ``datasets`` package. Streaming avoids downloading the full
    dataset; ``limit`` caps the number of problems for dev runs.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Loading from Hugging Face requires the 'datasets' package.") from exc

    ds = load_dataset(name, split=split, streaming=streaming)
    problems: list[ContestProblem] = []
    count = 0
    for row in ds:
        mapped = _code_contests_row_to_dict(row)
        try:
            problems.append(problem_from_dict(mapped))
        except ValueError:
            # Skip malformed rows rather than failing the whole split.
            continue
        count += 1
        if limit is not None and count >= limit:
            break
    return problems


def _code_contests_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    """Map a raw Code Contests row to our loader schema."""
    tests: list[dict] = []
    for t in row.get("tests", []) or []:
        tests.append(
            {
                "input": t.get("input", ""),
                "output": t.get("output", ""),
                "hidden": bool(t.get("hidden", False)),
            }
        )
    refs: list[dict] = []
    for s in row.get("solutions", []) or []:
        refs.append(
            {
                "language": "cpp" if s.get("language") == "cpp" else "py",
                "code": s.get("solution", ""),
                "status": "ok" if s.get("status") == "OK" else "fail",
            }
        )
    return {
        "problem_id": str(row.get("name", row.get("id", "unknown"))),
        "source": "codeforces",
        "statement": row.get("description", ""),
        "tests": tests,
        "reference_solutions": refs,
        "language": "py",
    }


def get_references(problem: ContestProblem, language: str | None = None) -> list[ReferenceSolution]:
    """Verifier/plagiarism-only accessor for a problem's reference solutions."""
    if language is None:
        return list(problem.reference_solutions)
    return problem.references_for(language)
