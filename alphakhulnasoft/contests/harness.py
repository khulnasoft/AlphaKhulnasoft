"""Multi-language execution harness and contest-style output comparison.

The harness compiles (if needed) and runs a candidate solution against every
test case, classifying each run as PASS / WA / RE / CE / TLE / MLE. It reuses
the same isolation idea as ``alphakhulnasoft.sandbox`` (subprocess + resource
limits + timeout) but is generalized to compiled languages.
"""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

from .languages import get_language_spec
from .problem import ContestProblem, TestCase

# Signals a child is likely to die with when our sandbox caps are exceeded. Under
# RLIMIT_AS a failed allocation is usually dereferenced (SIGSEGV/SIGBUS) or aborts
# (SIGABRT/SIGKILL); under RLIMIT_CPU the kernel sends SIGXCPU. These are kept
# separate from ordinary runtime errors (which also surface as signals).
_MEMORY_LIMIT_SIGNALS = tuple(
    s
    for s in (
        getattr(signal, "SIGSEGV", None),
        getattr(signal, "SIGBUS", None),
        getattr(signal, "SIGABRT", None),
        getattr(signal, "SIGKILL", None),
    )
    if s is not None
)


def compare_outputs(expected: str, actual: str, ignore_whitespace: bool = True) -> bool:
    """Contest-style comparison: exact on tokens, tolerant of whitespace layout.

    By default the two strings are compared after splitting on whitespace so
    trailing newlines / spaces never cause a mismatch. Set ``ignore_whitespace``
    to ``False`` for byte-exact comparison.
    """
    if ignore_whitespace:
        return expected.split() == actual.split()
    return expected == actual


@dataclass
class Grade:
    """Result of grading a solution against a set of test cases."""

    per_test: list[str] = field(default_factory=list)  # "PASS"/"WA"/"RE"/"CE"/"TLE"/"MLE"
    total: int = 0

    @property
    def passed(self) -> int:
        return self.per_test.count("PASS")

    @property
    def failed(self) -> int:
        return self.per_test.count("WA")

    @property
    def errored(self) -> int:
        return self.per_test.count("RE") + self.per_test.count("CE")

    @property
    def timeouts(self) -> int:
        return self.per_test.count("TLE")

    @property
    def resource(self) -> int:
        return self.per_test.count("MLE")

    def all_passed(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "errored": self.errored,
            "timeouts": self.timeouts,
            "resource": self.resource,
            "total": self.total,
            "per_test": list(self.per_test),
        }


def grade_solution(
    problem: ContestProblem,
    code: str,
    language: str,
    limits: dict | None = None,
    visible_only: bool = False,
    workdir: str | None = None,
) -> Grade:
    """Grade ``code`` in ``language`` against the problem's tests.

    If ``visible_only`` is True only public samples are used (the filter stage);
    otherwise all tests (including hidden) are used for final grading.
    """
    spec = get_language_spec(language)
    tests: list[TestCase] = problem.visible_tests if visible_only else problem.tests
    if not tests:
        return Grade(total=0)

    timeout = (
        int(limits.get("timeout_s", spec.default_timeout_s)) if limits else spec.default_timeout_s
    )
    memory_mb = (
        int(limits.get("memory_mb", spec.default_memory_mb)) if limits else spec.default_memory_mb
    )

    tmp_dir = workdir or tempfile.mkdtemp(prefix="contest_")
    filename = spec.source_filename or f"solution{spec.ext}"
    src_path = os.path.join(tmp_dir, filename)

    try:
        with open(src_path, "w", encoding="utf-8") as fh:
            fh.write(code)

        compile_error = _compile_if_needed(spec, src_path, tmp_dir)
        if compile_error is not None:
            return Grade(per_test=["CE"] * len(tests), total=len(tests))

        per_test: list[str] = []
        for test in tests:
            status, _out = _run_once(spec, src_path, tmp_dir, test.input, timeout, memory_mb)
            if status in ("RE", "TLE", "MLE"):
                per_test.append(status)
            elif status == "OK":
                actual = _out.strip() if _out is not None else ""
                per_test.append("PASS" if compare_outputs(test.expected_output, actual) else "WA")
            else:
                per_test.append("RE")
        return Grade(per_test=per_test, total=len(tests))
    finally:
        if workdir is None:
            _cleanup_dir(tmp_dir)


def _compile_if_needed(spec: object, src_path: str, tmp_dir: str) -> str | None:
    """Compile if the language needs it. Return an error string, or None on success."""
    compile_cmd = getattr(spec, "compile_cmd", None)
    if not compile_cmd:
        return None
    exe_path = os.path.join(tmp_dir, "solution_bin")
    cmd = [c.replace("{source}", src_path).replace("{output}", exe_path) for c in compile_cmd]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "Compile timed out"
    except Exception as exc:  # pragma: no cover - defensive
        return f"Compile failed: {exc}"
    if proc.returncode != 0:
        return proc.stderr.strip() or "Compile failed"
    # Stash the executable path on the spec-like object for the run step.
    spec.__dict__["_exe"] = exe_path  # type: ignore[attr-defined]
    return None


def _run_once(
    spec: object, src_path: str, tmp_dir: str, input_str: str, timeout: int, memory_mb: int
) -> tuple[str, str | None]:
    """Run the (compiled or interpreted) solution once. Returns (status, output)."""
    exe = spec.__dict__.get("_exe") if getattr(spec, "compile_cmd", None) else src_path
    cmd = [c.replace("{executable}", exe) for c in spec.run_cmd]  # type: ignore[attr-defined]

    def limit_resources() -> None:
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024)
            )
            resource.setrlimit(resource.RLIMIT_CPU, (timeout + 1, timeout + 1))
        except (OSError, ValueError):
            pass

    preexec = limit_resources if sys.platform != "win32" else None
    try:
        proc = subprocess.run(
            cmd,
            input=input_str,
            text=True,
            capture_output=True,
            timeout=timeout,
            preexec_fn=preexec,
        )
    except subprocess.TimeoutExpired:
        return "TLE", None
    except MemoryError:
        return "MLE", None
    except Exception:
        return "RE", None

    if proc.returncode != 0:
        # A negative return code means the child was terminated by a signal.
        # Under our sandbox the memory cap (RLIMIT_AS) manifests as SIGSEGV/SIGBUS
        # (a failed allocation dereferenced) or SIGABRT/SIGKILL, and the CPU cap
        # (RLIMIT_CPU) as SIGXCPU. Classify these as MLE / TLE rather than RE.
        if proc.returncode < 0:
            sig = -proc.returncode
            sigxcpu = getattr(signal, "SIGXCPU", None)
            if sigxcpu is not None and sig == sigxcpu:
                return "TLE", None
            if sig in _MEMORY_LIMIT_SIGNALS:
                return "MLE", None
            return "RE", None
        if "MemoryError" in (proc.stderr or ""):
            return "MLE", None
        return "RE", None
    return "OK", proc.stdout


def _cleanup_dir(path: str) -> None:
    import shutil

    shutil.rmtree(path, ignore_errors=True)
