from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class FGAConfig:
    """Runtime limits and model settings for FGA v0."""

    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"

    temperature: float = 0.2
    max_tokens: int = 4096

    max_talk_depth: int = 8
    max_messages_per_task: int = 60
    max_agent_steps: int = 24
    tool_result_chars: int = 8000
    shell_timeout_seconds: int = 120

    # Context window management for each actor's persistent conversation.
    # When the estimated token count exceeds max_context_tokens, older turns are
    # summarized and replaced, keeping system + the most recent turns intact.
    max_context_tokens: int = 24000
    keep_recent_messages: int = 12
    chars_per_token: int = 4

    # The shell is intentionally not a code-reading/editing tool.
    forbidden_shell_commands: tuple[str, ...] = (
        "cat",
        "sed",
        "grep",
        "rg",
        "awk",
        "less",
        "more",
        "head",
        "tail",
        "nl",
        "vi",
        "vim",
        "nano",
        "emacs",
        "tee",
    )

    @classmethod
    def from_env(cls) -> "FGAConfig":
        return cls(
            model=os.getenv("FGA_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
            base_url=os.getenv("FGA_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
            api_key_env=os.getenv("FGA_API_KEY_ENV", "DEEPSEEK_API_KEY"),
            temperature=float(os.getenv("FGA_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("FGA_MAX_TOKENS", "4096")),
            max_talk_depth=int(os.getenv("FGA_MAX_TALK_DEPTH", "8")),
            max_messages_per_task=int(os.getenv("FGA_MAX_MESSAGES", "60")),
            max_agent_steps=int(os.getenv("FGA_MAX_AGENT_STEPS", "24")),
            shell_timeout_seconds=int(os.getenv("FGA_SHELL_TIMEOUT", "120")),
            max_context_tokens=int(os.getenv("FGA_MAX_CONTEXT_TOKENS", "24000")),
            keep_recent_messages=int(os.getenv("FGA_KEEP_RECENT_MESSAGES", "12")),
        )
