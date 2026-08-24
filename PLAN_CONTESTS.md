# Competitive Programming Milestone — Implementation Plan

## 0. Goal & Thesis

Extend AlphaKhulnasoft from *code repair* into *novel competitive-programming
problem solving*: from a previously unseen contest problem statement
(natural language + sample I/O), produce a correct program in a target language
**without retrieving or copying an existing solution**.

The ML community can generate/understand text, but contest-level problem solving
remains limited to (a) simple drills or (b) recalling training-seen solutions. The
milestone targets **generalization to unforeseen problems**, validated on the
[Code Contests](https://github.com/google-research/google-research/tree/master/code_contests)
dataset (problems from Aizu, AtCoder, CodeChef, Codeforces, HackerEarth, with
paired input/output tests and correct/incorrect human solutions in many languages).

Success = `pass@k` on held-out problems **plus** a **novelty rate**: of problems
solved, the fraction whose solution is not a near-duplicate of any reference. This
directly attacks the "copying" failure mode.

---

## 1. Project Analysis

### Current state
- Flow engine in `alpha_repair.py` (Analyze → Generate → Verify → Root Cause → Repair).
- `sandbox.py` runs Python in a `subprocess` with time/memory limits; `proof_sandbox.py`
  validates proofs; both language-specific.
- `evaluator.py` scores runs; `visualizer.py` plots; `publisher.py` pushes to HF Hub.
- `data_loader.py` already ingests local files + Hugging Face (`datasets`).
- `llm.py` wraps OpenAI/Anthropic/litellm; `prompts.py` holds personas we can extend.
- Isolated `matmul/` and `proof_*` packages must stay decoupled.

### Gaps
- No **multi-language** execution (C++, Java, Python).
- No **contest problem model** (statement parsing, I/O extraction, sample/hidden tests).
- No **sampler + filter** loop (generate many → grade on visible tests → rank/select)
  with an added *novelty* filter.
- No **plagiarism / memorization** guard (the milestone differentiator).
- No `pass@k` + novelty benchmark on a held-out split.

### Decision
New self-contained `alphakhulnasoft/contests/` reusing the **flow loop** and **HF
publisher** but owning its problem model, harness, generator, verifier, benchmark.
LLM providers, proof sandbox, and matmul stay untouched. Default to **Python for CI**;
C++/Java compilation is opt-in/gated so CPU-only CI stays green.

---

## 2. Problem Contract

`ContestProblem` (language-agnostic):
- `problem_id`, `source` (aizu/atcoder/codechef/codeforces/hackerearth).
- `statement`; `io_format` (structured input/output format, constraints, limits).
- `samples`: `[(input, output)]` from visible tests.
- `tests`: `TestCase(input, expected_output, hidden: bool)`.
- `reference_solutions`: `ReferenceSolution(language, code, status)` — used ONLY by the
  verifier/plagiarism path, **never** as few-shot exemplars for generation.
- `language`: target generation language.

Acceptance: malformed/empty tests or missing statement → actionable error; visible
samples parse into pairs; reference solutions quarantined by the loader API.

---

## 3. Execution Harness (multi-language)

1. `languages.py`: registry `language → {ext, compile_cmd, run_cmd, default_limits}`.
2. `harness.py`: `grade_solution(problem, code, language)` writes source, compiles (if
   needed, hard timeout), runs **every** test, captures stdout/stderr/exit/wall-time/peak
   memory, and compares with a contest comparator (trailing-whitespace tolerant, exact
   otherwise). Returns `Grade(per_test_status, passed, failed, errored, timeouts, resource)`.
3. Isolation reuses `sandbox.py` primitives (subprocess, `ulimit`/`resource`, tmp dir,
   timeout); never run untrusted output outside the sandbox.

Tests: Python hello passes / wrong fails; C++ compile error → `errored` not `failed`;
TLE/MLE distinguished from WA/RE; comparator tolerant to whitespace, rejects wrong numbers.

---

## 4. Generator & Flow Loop (novel solving)

1. `prompts.py`: add CP personas (Analyst, Designer, Implementer, Tester) on `PromptRegistry`.
2. `generator.py`: `ContestAgent` reuses the flow loop:
   - **Analyze**: extract `io_format`, constraints, sub-problems (System-2 decomposition —
     the anti-retrieval step: reason about the problem, don't regurgitate).
   - **Generate**: `N` diverse candidates via temp/top-p + varied strategy prompts.
   - **Verify**: grade each on **visible** samples.
   - **Repair**: feed failing input + diff back; bounded iterations.
   - **Filter**: keep only candidates passing all visible samples.
3. `planner.py`: sample pool → filter by visible pass → rank by (verifier score +
   self-consistency + novelty) → select top-`k` for hidden grading.

Tests (mocked LLM): agent solves a tiny seeded problem; repair loop reduces failures
within cap; planner returns ≤ `k` candidates, only visible-test survivors.

---

## 5. Novelty & Anti-Copying Guard (differentiator)

1. `verifier.py`: `novelty_score(problem, code, language)`:
   - **AST/token similarity** vs every same-language reference (tree-sitter if available,
     else token/Jaccard fallback).
   - **Memorization probe**: separate LLM call asks the model to reproduce the canonical
     solution verbatim; compliance + match → down-weight.
   - Returns `novelty ∈ [0,1]` (1 = unlike all refs and not recalled).
2. Planner **prefers novel survivors**: correctness on visible tests is the gate; novelty is
   the tie-breaker/multiplier. A solved problem with `novelty < threshold` counts as
   *solved* but is reported separately as *retrieved*, not *novel*.
3. `benchmark` emits both `pass@k` and `novel_pass@k` (pass on the subset where the chosen
   solution is novel). The "new milestone" is supported only by `novel_pass@k`.

Tests: identical-to-reference → `novelty ≈ 0` + flagged retrieved; whitespace/renamed
duplicate still caught by AST/token (not exact match); novelty uses discrete thresholds,
never float tolerance.

---

## 6. Dataset Integration & Loader

1. `loader.py`: load Code Contests from local JSONL or HF (`datasets`, streaming for big
   splits). Exposes `reference_solutions` only via a separate accessor for verifier/plagiarism.
2. Splits: `train/val` for dev/tuning (no leakage into the metric); a **`novel` split** held
   out from likely pretraining (recent Codeforces rounds / curated unseen) — headline
   `novel_pass@k` computed only here.
3. `cli.py`: `python -m alphakhulnasoft.contests` with `load`/`solve`/`bench`
   (`--problem-id`, `--language`, `--n-samples`, `--split`, `--device cpu`).

Tests: rejects malformed lines; tiny checked-in fixture (2–3 problems, Py+C++ refs)
round-trips; references not exposed to the generator path (assert).

---

## 7. Benchmark & Milestone Definition

1. `benchmark.py`: `run_benchmark(split, language, n_samples, k)` → per problem: generate →
   filter → rank → grade top-`k` on hidden tests → record `pass@k`, `novel_pass@k`, mean
   novelty, mean repair iters, cost. Emits JSON + feeds `visualizer`.
2. **Milestone bar (honest, reproducible)**: internal target e.g. `novel_pass@5 ≥ X%` on the
   `novel` split (lang `py`), calibrated vs a retrieval/majority baseline from reference
   solutions (the "copying" ceiling). "New" only if `novel_pass@k` beats that ceiling beyond
   its variance. State explicitly that surpassing SoTA (e.g. AlphaCode) at scale needs large
   sample counts + a capable model; the repo delivers the *framework* and a *reproducible
   local milestone*, not a claim of beating published results without that compute.
3. Reuse `publisher.py` to push the report to HF Hub.

Tests: `pass@k` math unit-tested on a synthetic correctness matrix; fixture benchmark runs
end-to-end with mocked LLM, asserts stable JSON schema + non-negative metrics.

---

## 8. Recommended Implementation Order

1. Problem contract: `problem.py`, `languages.py`.
2. Harness `harness.py` + Python/C++/Java, unit tests on tiny problems.
3. Loader `loader.py` + fixture + splits.
4. Verifier: comparator + `novelty_score` (AST/token + memorization probe).
5. Generator `generator.py` + prompts, mocked-LLM tests.
6. Planner `planner.py` (sample → filter → rank by correctness+novelty).
7. CLI `cli.py` + `benchmark.py`; wire `visualizer` + `publisher`.
8. Novelty/memorization integration tests; small end-to-end fixture run.
9. README + notebook section; dependency groups (`tree-sitter` optional); CI markers.

---

## 9. Validation Gates

```bash
uv run ruff check alphakhulnasoft/contests tests/test_contests_*.py
uv run ruff format --check .
uv run mypy alphakhulnasoft/contests
uv run pytest tests/test_contests_*.py
uv run pytest tests/
```

GPU/large-model runs are opt-in; CI stays CPU-only with mocked LLMs. Compiled-language
tests are marked and skipped when no compiler is present.

---

## 10. Risks & Open Decisions

- Compilers (g++, javac) may be absent in CI → gate those tests; default Python for CI.
- Dataset is large; stream from HF and keep only a tiny fixture in-repo.
- LLM cost/recall: the memorization probe adds calls; make it toggleable.
- tree-sitter adds a dependency; provide a token-Jaccard fallback when absent.
- "Novel" threshold needs tuning on `val`; document cutoffs and keep them in config.
- Surpassing published SoTA requires compute beyond this repo; scope the milestone claim
  to the reproducible local framework + `novel_pass@k` vs the in-dataset copying ceiling.
