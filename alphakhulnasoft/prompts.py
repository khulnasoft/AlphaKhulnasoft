class PromptRegistry:
    """
    Central repository for Flow Engineering prompts.
    Separates logic (Python) from reasoning (English).
    """

    @staticmethod
    def semantic_analysis(problem_description: str) -> str:
        return f"""
        ACT AS: A Senior Systems Architect and Algorithms Expert.
        
        GOAL: Analyze the following competitive programming problem. Do NOT write code yet.
        
        PROBLEM:
        {problem_description}
        
        TASK:
        1. Identify the core algorithmic category (e.g., DP, Graph, Greedy).
        2. Extract HARD CONSTRAINTS (Time limit, Input size N, Memory).
        3. Identify EDGE CASES (Empty input, Max/Min values, Negative numbers).
        4. Suggest a Time Complexity (e.g., O(N log N)) required to pass.
        
        OUTPUT FORMAT:
        Return a strict bulleted list.
        - Algo: ...
        - Constraints: ...
        - Edge Cases: ...
        - Complexity: ...
        """

    @staticmethod
    def generate_solution(problem_desc: str, analysis: str) -> str:
        return f"""
        ACT AS: A 10x Python Developer.
        
        CONTEXT:
        We are solving this problem:
        {problem_desc}
        
        Analysis constraints provided by Architect:
        {analysis}
        
        TASK:
        Write a complete, self-contained Python solution.
        1. Import all necessary libraries.
        2. Handle standard input (stdin) properly.
        3. Address the Edge Cases identified in the analysis.
        4. Do NOT output markdown ticks or explanations, just the code.
        """

    @staticmethod
    def analyze_failure(code: str, error_log: str, problem_desc: str) -> str:
        return f"""
        ACT AS: A Lead Debugging Agent.
        
        STATUS: The code failed the tests.
        
        PROBLEM:
        {problem_desc}
        
        CODE:
        {code}
        
        ERROR LOG / TEST FAILURE:
        {error_log}
        
        TASK:
        1. Analyze the stack trace or output mismatch.
        2. Compare the Code logic against the Problem requirements.
        3. Formulate a HYPOTHESIS for the root cause (e.g., "Off-by-one error in loop", "Integer overflow", "Missed edge case X").
        
        OUTPUT:
        Start your response with "ROOT CAUSE:" followed by a 1-sentence technical diagnosis.
        Then explain briefly.
        """

    @staticmethod
    def targeted_repair(code: str, root_cause: str) -> str:
        return f"""
        ACT AS: A Maintenance Engineer.
        
        TASK: Fix the code based EXACTLY on the diagnosed root cause.
        
        ROOT CAUSE:
        {root_cause}
        
        ORIGINAL CODE:
        {code}
        
        INSTRUCTION:
        1. Apply the minimum viable patch to fix the root cause.
        2. Ensure you do not break existing logic.
        3. Return the full corrected Python code.
        """
