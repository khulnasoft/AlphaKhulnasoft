import os
import subprocess
import sys
import tempfile


class Sandbox:
    """
    Safely executes generated Python code against test cases.
    Handles timeouts, captures stdout/stderr, and isolates the process.
    """

    def __init__(self, timeout_seconds: int = 2):
        self.timeout = timeout_seconds

    def run_tests(self, code: str, test_cases: list[dict]) -> tuple[float, str]:
        """
        Runs the code against all provided test cases.
        Returns: (pass_rate [0.0-1.0], error_log [str])
        """
        if not code.strip():
            return 0.0, "❌ Error: Empty code generated."

        passes = 0
        logs = []

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            for i, test in enumerate(test_cases):
                input_data = str(test.get("input", ""))
                expected = str(test.get("expected", "")).strip()

                result = self._execute_single_run(tmp_path, input_data)

                if result["error"]:
                    logs.append(f"Test {i + 1} ❌: Runtime Error\n{result['error']}")
                    continue

                actual = result["output"].strip()

                if actual == expected:
                    passes += 1
                else:
                    logs.append(
                        f"Test {i + 1} ❌: Failed.\n   Input: {input_data}\n   Expected: '{expected}'\n   Got: '{actual}'"
                    )

        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        pass_rate = passes / len(test_cases) if test_cases else 0.0
        final_log = "\n".join(logs[:3])  # Only return top 3 errors to save tokens

        return pass_rate, final_log

    def _execute_single_run(self, file_path: str, input_str: str) -> dict:
        """
        Low-level execution with timeout and pipe management.
        """
        try:
            # Run the python script as a subprocess
            process = subprocess.run(
                [sys.executable, file_path],
                input=input_str,
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )

            # Check for non-zero exit codes (Runtime Errors)
            if process.returncode != 0:
                return {"output": "", "error": process.stderr}

            return {"output": process.stdout, "error": None}

        except subprocess.TimeoutExpired:
            return {"output": "", "error": f"⏱️ Time Limit Exceeded ({self.timeout}s)"}
        except Exception as e:
            return {"output": "", "error": f"System Error: {str(e)}"}
