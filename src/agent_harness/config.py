import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = "http://127.0.0.1:8080/v1"
    llm_timeout: float = 120.0
    model: str | None = None
    temperature: float = 0.0
    tool_protocol: str = "native"
    max_iterations: int = 5
    max_tokens: int = 1024
    tool_timeout: float = 10.0
    execution_timeout: float = 300.0
    max_tool_calls: int = 10

    def __post_init__(self) -> None:
        if not 1 <= self.max_iterations <= 5:
            raise ValueError("Agent iterations must be between 1 and 5")
        if not 1 <= self.max_tokens <= 4096:
            raise ValueError("Maximum tokens must be between 1 and 4096")
        if not 0 < self.llm_timeout <= 600 or not 0 < self.tool_timeout <= 60:
            raise ValueError("Timeouts must be positive and bounded")
        if not 0 < self.execution_timeout <= 3600:
            raise ValueError("Execution timeout must be positive and at most 3600 seconds")
        if not 1 <= self.max_tool_calls <= 50:
            raise ValueError("Tool calls must be between 1 and 50")
        if self.tool_protocol not in {"native", "json"}:
            raise ValueError("Tool protocol must be 'native' or 'json'")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_base_url=os.getenv("LLM_BASE_URL", cls.llm_base_url),
            llm_timeout=float(os.getenv("LLM_TIMEOUT", str(cls.llm_timeout))),
            model=os.getenv("LLM_MODEL") or None,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            tool_protocol=os.getenv("TOOL_PROTOCOL", "native"),
            max_iterations=min(5, max(1, int(os.getenv("AGENT_MAX_ITERATIONS", "5")))),
            max_tokens=max(1, int(os.getenv("LLM_MAX_TOKENS", "1024"))),
            tool_timeout=max(0.1, float(os.getenv("TOOL_TIMEOUT", "10"))),
            execution_timeout=float(os.getenv("AGENT_EXECUTION_TIMEOUT", "300")),
            max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "10")),
        )
