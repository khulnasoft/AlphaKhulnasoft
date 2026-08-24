"""Tests for the contest generator and planner (flow loop) with a fake LLM."""

from alphakhulnasoft.contests.generator import ContestAgent, GeneratedCandidate
from alphakhulnasoft.contests.loader import load_local
from alphakhulnasoft.contests.planner import rank_candidates, select_top

FIXTURE = "alphakhulnasoft/contests/data/tiny.jsonl"
CORRECT = "import sys\nx,y=map(int,sys.stdin.read().split())\nprint(x+y)"


class _SolvingLLM:
    """Always returns a correct a+b solution."""

    def complete(self, prompt, system_prompt=None):
        return CORRECT

    def extract_code(self, text):
        return text.strip()


class _RepairingLLM:
    """Returns a wrong solution on generate, correct on the repair prompt."""

    def complete(self, prompt, system_prompt=None):
        if system_prompt and "debugging" in system_prompt:
            return CORRECT  # repair -> correct
        return "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a-b)"  # wrong

    def extract_code(self, text):
        return text.strip()


def _agent(llm):
    return ContestAgent(llm=llm, sandbox_timeout=5, max_repair_iters=3)


def test_generate_pool_produces_graded_candidates():
    p = load_local(FIXTURE)[0]
    pool = _agent(_SolvingLLM()).generate_pool(p, "py", n_samples=4)
    assert len(pool) == 4
    assert all(c.grade is not None for c in pool)
    assert all(c.grade.all_passed() for c in pool)


def test_solve_returns_solved_candidate():
    p = load_local(FIXTURE)[0]
    cand = _agent(_SolvingLLM()).solve(p, "py", n_samples=4)
    assert isinstance(cand, GeneratedCandidate)
    assert cand.status == "SOLVED"
    assert cand.grade.all_passed()


def test_repair_loop_fixes_visible_failures():
    p = load_local(FIXTURE)[0]
    cand = _agent(_RepairingLLM()).solve(p, "py", n_samples=3)
    # First generations fail; the repair step must produce a passing candidate.
    assert cand.status == "SOLVED"
    assert cand.grade.all_passed()


def test_planner_filters_and_ranks_by_novelty():
    p = load_local(FIXTURE)[0]
    pool = _agent(_SolvingLLM()).generate_pool(p, "py", n_samples=5)
    ranked = rank_candidates(pool, p, "py")
    assert ranked  # all survived visible tests
    assert all(c.novelty is not None for c in ranked)
    top = select_top(pool, p, "py", k=2)
    assert len(top) <= 2
    assert top[0].novelty >= top[-1].novelty
