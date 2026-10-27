from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True, slots=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    state_backend: str = os.getenv("STATE_BACKEND", "memory").strip().lower()
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    queue_maxsize: int = int(os.getenv("QUEUE_MAXSIZE", "4"))
    agent_timeout_seconds: float = float(os.getenv("AGENT_TIMEOUT_SECONDS", "40"))
    workflow_timeout_seconds: float = float(os.getenv("WORKFLOW_TIMEOUT_SECONDS", "90"))
    max_agent_retries: int = int(os.getenv("MAX_AGENT_RETRIES", "1"))
    structured_output_retries: int = int(os.getenv("STRUCTURED_OUTPUT_RETRIES", "2"))
    retry_base_delay_seconds: float = float(os.getenv("RETRY_BASE_DELAY_SECONDS", "0.25"))

    def validate(self) -> None:
        if self.state_backend not in {"memory", "redis"}:
            raise ValueError("STATE_BACKEND must be 'memory' or 'redis'.")
        if self.queue_maxsize < 1:
            raise ValueError("QUEUE_MAXSIZE must be at least 1.")
        if self.agent_timeout_seconds <= 0 or self.workflow_timeout_seconds <= 0:
            raise ValueError("Timeout settings must be positive.")
        if self.max_agent_retries < 0 or self.structured_output_retries < 0:
            raise ValueError("Retry settings cannot be negative.")
