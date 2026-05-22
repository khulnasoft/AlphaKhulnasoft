import time

from dotenv import load_dotenv

load_dotenv()


class LLMProvider:
    """
    Wrapper for LLM calls using litellm for multi-provider support.
    Supports OpenAI, Anthropic, Hugging Face, and Vertex AI.

    Vertex AI Example: model="vertex_ai/gemini-1.5-pro"
    """

    def __init__(self, model: str = "gpt-4-turbo"):
        self.model = model

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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = litellm.completion(model=self.model, messages=messages)
                if not response.choices or not response.choices[0].message:
                    raise ValueError("Invalid LLM response structure")
                return str(response.choices[0].message.content)
            except (litellm.RateLimitError, litellm.APIConnectionError):
                if attempt < max_retries - 1:
                    wait_time = (2**attempt) + 1
                    print(
                        f"⚠️ Transient error (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s..."
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
