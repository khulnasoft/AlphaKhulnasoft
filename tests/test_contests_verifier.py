"""Tests for the novelty / anti-copying verifier."""

from alphakhulnasoft.contests.loader import load_local
from alphakhulnasoft.contests.verifier import (
    NOVEL_THRESHOLD,
    RETRIEVED_THRESHOLD,
    classify_novelty,
    jaccard,
    memorization_probe,
    novelty_score,
    similarity_to_references,
    tokenize,
)

FIXTURE = "alphakhulnasoft/contests/data/tiny.jsonl"
REF_PY = "import sys\na,b=map(int,sys.stdin.read().split())\nprint(a+b)"
RENAMED = "import sys\nx,y=map(int,sys.stdin.read().split())\nprint(x+y)"


class _HistorianLLM:
    """Pretends to recall the canonical reference verbatim."""

    def complete(self, prompt, system_prompt=None):
        if system_prompt and "historian" in system_prompt:
            return REF_PY
        return "import sys\nx,y=map(int,sys.stdin.read().split())\nprint(x+y)"

    def extract_code(self, text):
        return text.strip()


class _QuietLLM:
    """Returns a structurally unrelated program (low token similarity)."""

    def complete(self, prompt, system_prompt=None):
        return "def fib(n):\n  a,b=0,1\n  for _ in range(n):\n    a,b=b,a+b\n  return a"

    def extract_code(self, text):
        return text.strip()


def _p():
    return load_local(FIXTURE)[0]


def test_identical_to_reference_is_retrieved():
    nov = novelty_score(_p(), REF_PY, "py")
    assert nov == 0.0
    assert classify_novelty(nov) == "retrieved"


def test_renamed_duplicate_is_less_novel_than_identical():
    nov_ident = novelty_score(_p(), REF_PY, "py")
    nov_renamed = novelty_score(_p(), RENAMED, "py")
    assert nov_renamed > nov_ident
    assert nov_renamed <= NOVEL_THRESHOLD


def test_novelty_not_exact_match_rename_caught():
    # A superficial rename changes identifier tokens; similarity must drop but
    # remain high (it is still essentially the same program).
    sim = similarity_to_references(_p(), RENAMED, "py")
    assert 0.0 < sim < 1.0


def test_memorization_probe_detects_recall():
    assert memorization_probe(_p(), _HistorianLLM()) is True
    assert memorization_probe(_p(), _QuietLLM()) is False


def test_memorization_probe_clamps_novelty():
    nov = novelty_score(_p(), REF_PY, "py", llm=_HistorianLLM())
    assert nov <= RETRIEVED_THRESHOLD


def test_classify_thresholds():
    assert classify_novelty(0.9) == "novel"
    assert classify_novelty(0.1) == "retrieved"
    assert classify_novelty(0.5) == "borderline"


def test_jaccard_and_tokenize():
    assert jaccard(tokenize("a b c"), tokenize("a b c")) == 1.0
    assert jaccard(tokenize("a b"), tokenize("c d")) == 0.0
    assert 0.0 < jaccard(tokenize("a b c"), tokenize("a b d")) < 1.0
