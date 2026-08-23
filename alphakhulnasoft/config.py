from dataclasses import dataclass


@dataclass
class AlphaConfig:
    model_name: str = "gpt-4o"
    max_retries: int = 5
    sandbox_timeout: int = 2
    llm_max_retries: int = 3
    max_memory_mb: int = 512
    validate_api_keys: bool = True

    @classmethod
    def from_env(cls):
        import os

        return cls(
            model_name=os.getenv("ALPHA_MODEL", "gpt-4o"),
            max_retries=int(os.getenv("ALPHA_MAX_RETRIES", 5)),
            sandbox_timeout=int(os.getenv("ALPHA_SANDBOX_TIMEOUT", 2)),
            llm_max_retries=int(os.getenv("ALPHA_LLM_RETRIES", 3)),
            max_memory_mb=int(os.getenv("ALPHA_MAX_MEMORY_MB", 512)),
            validate_api_keys=os.getenv("ALPHA_VALIDATE_API_KEYS", "true").lower() == "true",
        )
