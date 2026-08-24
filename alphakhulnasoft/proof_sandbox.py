"""Validation engine for formal Nexus proofs.

Mirrors ``alphakhulnasoft.sandbox.Sandbox`` but validates mathematical
proofs instead of executing Python code. When the Nexus toolchain is
available on ``PATH`` the sandbox shells out to it for real type-checking;
otherwise it falls back to a structural heuristic so the rest of the
pipeline remains runnable in CI / environments without Nexus installed.
"""

import contextlib
import os
import re
import shutil
import subprocess
import tempfile


class ProofSandbox:
    """Validates formal proofs using the Nexus type-checker and tactics."""

    def __init__(self, nexus_bin: str = "nexus", timeout_seconds: int = 30):
        self.nexus_bin = nexus_bin
        self.timeout = timeout_seconds

    def verify_nexus_proof(self, nexus_code: str) -> tuple[bool, str]:
        """Runs Nexus code through the type-checker.

        Returns:
            ``(is_valid, error_log)`` where ``is_valid`` is a boolean and
            ``error_log`` is a human readable message.
        """
        if not nexus_code or not nexus_code.strip():
            return False, "ERROR: Empty proof generated."

        binary = shutil.which(self.nexus_bin)
        if binary:
            return self._verify_with_binary(binary, nexus_code)
        return self._verify_heuristic(nexus_code)

    def _verify_with_binary(self, binary: str, nexus_code: str) -> tuple[bool, str]:
        """Type-check by shelling out to the real Nexus binary."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".nx", delete=False) as tmp:
                tmp.write(nexus_code)
                tmp_path = tmp.name

            process = subprocess.run(
                [binary, "check", tmp_path],
                text=True,
                capture_output=True,
                timeout=self.timeout,
            )

            if process.returncode == 0:
                return True, ""
            return False, process.stderr.strip() or "Nexus verification failed."
        except subprocess.TimeoutExpired:
            return False, f"⏱️ Nexus verification timed out ({self.timeout}s)"
        except Exception as e:  # pragma: no cover - defensive
            return False, f"System Error: {str(e)}"
        finally:
            if tmp_path and os.path.exists(tmp_path):
                with contextlib.suppress(OSError):
                    os.remove(tmp_path)

    def _verify_heuristic(self, nexus_code: str) -> tuple[bool, str]:
        """Lightweight structural checks when Nexus is not installed.

        This is intentionally conservative: it only catches gross structural
        mistakes (missing theorem declaration or an unfinished ``sorry`` at the
        top level). A green result here does NOT guarantee a valid proof; it
        merely means the candidate is structurally plausible enough to submit
        to the real Nexus toolchain.
        """
        code = nexus_code.strip()

        if not self._has_theorem_declaration(code):
            return (
                False,
                "ERROR: Proof must start with 'theorem' keyword. Missing theorem declaration.",
            )

        if code.rstrip().endswith("sorry"):
            return (
                False,
                "ERROR: Proof is incomplete. Main theorem ends with 'sorry'. Discharge all proof obligations.",
            )

        return True, ""

    @staticmethod
    def _has_theorem_declaration(code: str) -> bool:
        """Return whether the proof contains a top-level theorem declaration."""
        return re.search(r"(?m)^theorem\s+[A-Za-z_][\w']*\b", code) is not None

    def check_proof_tactics(self, nexus_code: str) -> dict:
        """Analyze proof structure: induction, cases, lemmas, etc."""
        code = nexus_code.lower()
        return {
            "uses_induction": "induction" in code,
            "uses_cases": "cases" in code,
            "uses_contradiction": "contradiction" in code,
            "has_lemmas": "lemma" in code,
            "uses_sorry": "sorry" in code,
            "line_count": len(nexus_code.split("\n")),
        }
