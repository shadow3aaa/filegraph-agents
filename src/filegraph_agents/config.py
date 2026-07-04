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

    tool_result_chars: int = 8000
    shell_timeout_seconds: int = 120

    # Max LLM requests in flight at once across all actors. Actors still run on
    # their own threads, but this caps simultaneous API calls so a wide talk
    # fan-out does not overwhelm rate-limited providers. Set <= 0 to disable.
    max_parallel_talks: int = 2

    # Context window management for each actor's persistent conversation.
    #
    # The compaction watermark is derived from the model's real context window
    # (looked up from litellm's catalog) times context_use_ratio. When an actor's
    # estimated tokens exceed that, older turns are summarized, keeping system +
    # the most recent turns whose tokens fit keep_recent_ratio of the window.
    #
    # context_window_override forces a specific window size (in tokens) instead of
    # catalog lookup; context_window_fallback is used only when the model is not
    # found in the catalog.
    context_use_ratio: float = 0.75
    keep_recent_ratio: float = 0.4
    context_window_override: int = 0
    context_window_fallback: int = 32000
    keep_recent_messages_floor: int = 4
    chars_per_token: int = 4

    # The shell is intentionally not a code-reading/editing tool.
    log_steps: bool = True

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

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return val.lower() not in ("0", "false", "no", "off", "")

    @classmethod
    def from_env(cls) -> "FGAConfig":
        return cls(
            model=os.getenv("FGA_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")),
            base_url=os.getenv("FGA_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
            api_key_env=os.getenv("FGA_API_KEY_ENV", "DEEPSEEK_API_KEY"),
            temperature=float(os.getenv("FGA_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("FGA_MAX_TOKENS", "4096")),
            shell_timeout_seconds=int(os.getenv("FGA_SHELL_TIMEOUT", "120")),
            context_use_ratio=float(os.getenv("FGA_CONTEXT_USE_RATIO", "0.75")),
            keep_recent_ratio=float(os.getenv("FGA_KEEP_RECENT_RATIO", "0.4")),
            context_window_override=int(os.getenv("FGA_CONTEXT_WINDOW", "0")),
            context_window_fallback=int(os.getenv("FGA_CONTEXT_WINDOW_FALLBACK", "32000")),
            max_parallel_talks=int(os.getenv("FGA_MAX_PARALLEL_TALKS", "2")),
            log_steps=cls._env_bool("FGA_LOG_STEPS", default=True),
        )
