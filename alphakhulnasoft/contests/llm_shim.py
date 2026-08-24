"""Thin factory for the LLM used by the contests CLI.

Kept separate so the rest of the package can be tested with a fake LLM that
exposes the same ``complete`` / ``extract_code`` surface, while the CLI uses the
real multi-provider :class:`LLMProvider`.
"""

from __future__ import annotations

from typing import Any


def make_llm(model: str | None = None) -> Any:
    """Return the production LLM provider (validates API keys)."""
    from ..config import AlphaConfig
    from ..llm import LLMProvider

    config = AlphaConfig()
    return LLMProvider(model=model or config.model_name, validate_keys=True)
