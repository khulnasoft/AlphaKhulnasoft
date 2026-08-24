"""Specialized prompt registry for mathematical proof generation.

This module adapts the Flow Engineering prompt strategy (see
``alphakhulnasoft.prompts.PromptRegistry``) to the domain of formal proofs.
Each method returns a fully-specified prompt string; the reasoning persona
lives in English here, while the proof logic lives in the LLM call sites.
"""


class ProofPromptRegistry:
    """Central repository for mathematical proof generation prompts."""

    @staticmethod
    def analyze_proof_requirements(theorem_statement: str, proof_hints: str = "") -> str:
        """Analyze a theorem to extract proof requirements and strategy."""
        hints_context = f"\n\nHINTS:\n{proof_hints}" if proof_hints else ""

        return f"""
        ACT AS: A Senior Formal Mathematics Expert and Proof Strategist.

        GOAL: Analyze the following theorem WITHOUT writing a formal proof yet.

        THEOREM:
        {theorem_statement}{hints_context}

        TASK:
        1. Identify the primary proof technique (induction, contradiction, direct, by cases, constructive, contrapositive, etc.).
        2. Extract core proof constraints:
           - Base cases (if induction)
           - Inductive hypothesis (if induction)
           - Key invariants or assumptions
           - Quantified variables and their domains
        3. Identify lemmas or auxiliary theorems that might be needed.
        4. List potential edge cases or special cases to handle.
        5. Suggest a proof strategy skeleton (e.g., "Prove base case, then inductive step").
        6. Estimate proof depth (nested inductions, case splits, etc.).

        OUTPUT FORMAT (strict):
        - Technique: [name of technique]
        - Base Cases: [list or "N/A"]
        - Inductive Hypothesis: [statement or "N/A"]
        - Key Invariants: [list]
        - Needed Lemmas: [list or "none"]
        - Edge Cases: [list]
        - Strategy Skeleton: [brief outline]
        - Proof Depth: [low/medium/high]
        - Complexity Notes: [any special considerations]
        """

    @staticmethod
    def generate_nexus_proof(
        theorem_statement: str, analysis: str, proof_technique: str = ""
    ) -> str:
        """Generate a formal Nexus proof from a structured theorem analysis."""
        technique_hint = (
            f"\nFOCUS: Use {proof_technique} proof technique." if proof_technique else ""
        )

        return f"""
        ACT AS: A Nexus Formal Proof Assistant.

        GOAL: Write a complete, type-checking Nexus proof for the following theorem.

        THEOREM:
        {theorem_statement}

        PROOF ANALYSIS (from domain expert):
        {analysis}{technique_hint}

        NEXUS PROOF GENERATION TASK:
        1. Write syntactically valid Nexus proof code.
        2. Use Nexus keywords: theorem, proof, by, cases, induction, simp, intro, exact, sorry.
        3. Implement the proof strategy from the analysis above.
        4. Ensure all hypotheses and cases are explicitly addressed (do NOT use sorry for the main goal).
        5. For complex sub-goals, use sorry ONLY for intermediate lemmas if necessary.
        6. Use proper indentation and Nexus syntax (colons, tactics indentation).
        7. Include type annotations where required.
        8. Make proof steps explicit and verifiable.

        NEXUS SYNTAX REMINDERS:
        - theorem name : statement := proof
        - proof by (tactic1; tactic2; ...)
        - induction n with base => step =>
        - cases h where
        - exact term
        - simp [lemma1, lemma2]
        - intro x h => ...
        - sorry  (only for intermediate lemmas, NOT main goal)

        OUTPUT: Return ONLY the Nexus proof code. No markdown, explanations, or preamble.
        Start with "theorem" and end with proof completion or "sorry" for lemmas only.
        """

    @staticmethod
    def analyze_proof_error(
        theorem_statement: str, nexus_code: str, error_log: str, attempt_number: int = 1
    ) -> str:
        """Diagnose why a Nexus proof failed type-checking or verification."""
        return f"""
        ACT AS: A Proof Assistant Debugger and Nexus Expert.

        STATUS: Proof verification FAILED (attempt {attempt_number}).

        THEOREM:
        {theorem_statement}

        NEXUS CODE (FAILED):
        {nexus_code}

        TYPE-CHECK / VERIFICATION ERROR:
        {error_log}

        ROOT CAUSE ANALYSIS TASK:
        1. Identify which proof step failed (missing case, wrong tactic, type mismatch, incomplete proof).
        2. Explain why the Nexus code does not type-check or verify:
           - Is a case missing?
           - Is the inductive hypothesis incorrectly applied?
           - Is there a type annotation error?
           - Are proof obligations not fully discharged?
           - Is a lemma incorrectly invoked?
        3. Suggest a targeted fix (minimum viable change).
        4. Be specific: quote the problematic line and explain the issue.

        OUTPUT FORMAT:
        Start with "ROOT CAUSE:" on its own line.
        Then provide:
        - Problem: [1-sentence technical diagnosis]
        - Location: [which line/tactic in Nexus code]
        - Fix Strategy: [specific steps to resolve]
        """

    @staticmethod
    def repair_nexus_proof(
        theorem_statement: str, nexus_code: str, root_cause_analysis: str
    ) -> str:
        """Generate a repaired Nexus proof based on a root cause analysis."""
        return f"""
        ACT AS: A Nexus Proof Maintenance Engineer.

        TASK: Repair the Nexus proof based on the diagnosed root cause.

        THEOREM:
        {theorem_statement}

        ORIGINAL CODE (BROKEN):
        {nexus_code}

        ROOT CAUSE DIAGNOSIS:
        {root_cause_analysis}

        REPAIR TASK:
        1. Apply the minimum viable fix to resolve the root cause.
        2. Do NOT rewrite the entire proof unnecessarily.
        3. Maintain all correct proof structure from the original.
        4. Ensure the repaired code is syntactically valid Nexus.
        5. All proof obligations must be discharged (no "sorry" for main goal).
        6. Test your logic: does each step follow from the premises?

        OUTPUT: Return ONLY the complete, repaired Nexus proof code.
        No markdown, no explanations, no preamble. Start with "theorem".
        """

    @staticmethod
    def generate_prose_explanation(
        theorem_statement: str, nexus_code: str, analysis: str = ""
    ) -> str:
        """Generate a natural language proof explanation from a formal Nexus proof."""
        analysis_context = (
            f"\n\nPROOF STRATEGY ANALYSIS (for reference):\n{analysis}" if analysis else ""
        )

        return f"""
        ACT AS: A Mathematical Writer and Formal Logic Translator.

        GOAL: Translate a formal Nexus proof into clear, rigorous natural language.

        THEOREM (to prove):
        {theorem_statement}

        FORMAL NEXUS PROOF:
        {nexus_code}{analysis_context}

        PROSE GENERATION TASK:
        1. Write a complete natural language proof suitable for a university mathematics textbook.
        2. Start with "Proof:" and explicitly state the proof technique used.
        3. For each major step in the Nexus proof:
           - Explain the intuition (why this step is valid)
           - Translate tactic to plain English
           - Show the logical connection to previous steps
        4. Highlight key lemmas and their essential roles.
        5. Use standard mathematical notation and language (assume reader knows proof techniques).
        6. Keep explanations rigorous but accessible.
        7. End with "QED." or "∎".

        STRUCTURE:
        - Introduction: State what technique is used
        - Main Body: Systematic proof of each case/step
        - Lemmas (if any): Brief separate proofs of auxiliary lemmas
        - Conclusion: "QED."

        OUTPUT: Return the complete natural language proof.
        Use markdown formatting: **bold** for key terms, `code` for formal symbols.
        Make it readable as a standalone mathematical exposition.
        """

    @staticmethod
    def verify_proof_completeness(theorem_statement: str, nexus_code: str) -> str:
        """Check whether a Nexus proof fully discharges all proof obligations."""
        return f"""
        ACT AS: A Formal Verification Expert.

        TASK: Assess whether the Nexus proof is complete and rigorous.

        THEOREM:
        {theorem_statement}

        NEXUS PROOF:
        {nexus_code}

        COMPLETENESS CHECK:
        1. Is the main theorem fully proven (no "sorry" at top level)?
        2. Are all cases or inductive branches covered?
        3. Is the inductive hypothesis correctly invoked?
        4. Are base cases explicitly handled?
        5. Do all tactics logically follow from premises?
        6. Are there any gaps or unjustified steps?

        OUTPUT FORMAT:
        - Complete: [yes/no]
        - Missing Elements: [list or "none"]
        - Gaps: [any logical gaps or unjustified steps]
        - Recommendations: [how to strengthen if incomplete]
        """

    @staticmethod
    def compare_proofs(theorem_statement: str, proof1: str, proof2: str) -> str:
        """Compare two proof approaches for equivalence and efficiency."""
        return f"""
        ACT AS: A Proof Theorist and Complexity Analyst.

        THEOREM:
        {theorem_statement}

        PROOF APPROACH 1:
        {proof1}

        PROOF APPROACH 2:
        {proof2}

        COMPARISON TASK:
        1. Do both proofs prove the same theorem? (logical equivalence)
        2. Which proof is more concise?
        3. Which proof has clearer structure?
        4. Which proof depth is lower (fewer nested tactics)?
        5. Are there any tactics in one that could improve the other?
        6. Which is preferable for a textbook exposition?

        OUTPUT:
        - Logically Equivalent: [yes/no]
        - Conciseness Winner: [proof 1/2 or tied]
        - Clarity Winner: [proof 1/2 or tied]
        - Recommended Version: [1 or 2 with brief justification]
        """
