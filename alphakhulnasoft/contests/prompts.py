"""Prompt personas for competitive-programming solving.

Reuses the Flow Engineering style (logic in Python, reasoning in English) and
adds the analyst/designer/implementer/tester + memorization-probe personas. The
*analysis* step is the explicit anti-retrieval move: the model must reason about
the problem rather than recall a stored solution.
"""

from __future__ import annotations


class ContestPromptRegistry:
    """Prompt templates for the contest flow loop."""

    @staticmethod
    def analyze(problem_statement: str) -> str:
        return f"""
        ACT AS: A competitive programming analyst (System 2 thinking).

        PROBLEM:
        {problem_statement}

        TASK (do NOT write code yet):
        1. Identify the algorithmic category (DP / Graph / Greedy / Math / Data structures).
        2. Extract HARD constraints: variable bounds, time/memory limits, I/O shape.
        3. List edge cases (empty input, duplicates, large N, overflow, modulo).
        4. Choose a target complexity and a concrete strategy.
        Return a strict bulleted analysis.
        """

    @staticmethod
    def generate(
        problem_statement: str, analysis: str, language: str, strategy: str | None = None
    ) -> str:
        strat = f"\nSTRATEGY HINT (vary it across attempts): {strategy}\n" if strategy else ""
        return f"""
        ACT AS: A {language} competitive programmer.

        PROBLEM:
        {problem_statement}

        ANALYSIS:
        {analysis}
        {strat}
        TASK: Write a complete, self-contained {language} program reading from stdin and
        writing to stdout. Handle the listed edge cases. Output ONLY code, no explanation,
        no markdown fences.
        """

    @staticmethod
    def repair(problem_statement: str, code: str, error_log: str, analysis: str) -> str:
        return f"""
        ACT AS: A debugging agent.

        PROBLEM:
        {problem_statement}

        ANALYSIS:
        {analysis}

        CODE:
        {code}

        FAILURE (failing input / diff / error):
        {error_log}

        TASK: Apply the minimum patch to fix the failure. Return the FULL corrected code only.
        """

    @staticmethod
    def memorization(problem_statement: str, language: str) -> str:
        return f"""
        ACT AS: A competitive programming historian.
        Reproduce, verbatim, the most well-known reference solution for this problem in {language}.
        Output only code.
        PROBLEM:
        {problem_statement}
        """
