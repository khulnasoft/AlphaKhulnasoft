import os
import time

from dotenv import load_dotenv

load_dotenv()


def validate_api_keys() -> None:
    """Validate that at least one LLM API key is configured."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_vertex = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    has_gemini = bool(os.getenv("GOOGLE_API_KEY"))

    if not any([has_openai, has_anthropic, has_vertex, has_gemini]):
        raise ValueError(
            "No LLM API keys found. Please set at least one of: "
            "OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_APPLICATION_CREDENTIALS, or GOOGLE_API_KEY"
        )


class LLMProvider:
    """
    Wrapper for LLM calls using litellm for multi-provider support.
    Supports OpenAI, Anthropic, Hugging Face, and Vertex AI.

    Vertex AI Example: model="vertex_ai/gemini-1.5-pro"
    """

    def __init__(
        self, model: str = "gpt-4-turbo", max_retries: int = 3, validate_keys: bool = True
    ):
        if max_retries <= 0:
            raise ValueError(f"max_retries must be positive, got {max_retries}")
        if validate_keys:
            validate_api_keys()
        self.model = model
        self.max_retries = max_retries

    def complete(self, prompt: str, system_prompt: str | None = None) -> str:
        """Sends a completion request to the LLM with retry logic."""
        import litellm

        # Disable telemetry and version checks to prevent hangs
        litellm.telemetry = False
        litellm.version_check = False
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = litellm.completion(model=self.model, messages=messages)
                if not response.choices or not response.choices[0].message:
                    raise ValueError("Invalid LLM response structure")
                return str(response.choices[0].message.content)
            except (litellm.RateLimitError, litellm.APIConnectionError):
                if attempt < self.max_retries - 1:
                    wait_time = (2**attempt) + 1
                    print(
                        f"⚠️ Transient error (attempt {attempt + 1}/{self.max_retries}), retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    continue
                raise
            except litellm.APIError as e:
                print(f"❌ LLM API Error: {e}")
                raise
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                raise

        raise RuntimeError("Max retries exceeded")

    def extract_code(self, text: str) -> str:
        """Heuristic to extract code from markdown backticks."""
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()
        return text.strip()
