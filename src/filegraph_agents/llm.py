from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from typing import Any, Callable, Protocol

from .config import FGAConfig
from .errors import FGAError


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelResponse:
    """Structured response from the model.

    One of `content` or `tool_calls` will be set (or both).
    `reasoning_content` is the model's chain-of-thought (DeepSeek o1/R1 etc.)
    and must be preserved across turns so the model can continue its reasoning.
    """
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    reasoning_content: str | None = None


class ChatModel(Protocol):
    """Minimal chat model interface used by the actor runtime."""

    def complete(
        self,
        actor_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse: ...


@dataclass(slots=True)
class LiteLLMModel:
    """Lightweight model client backed by litellm with tool-calling support.

    Configure with env vars, for example:
      DEEPSEEK_API_KEY=...
      FGA_BASE_URL=https://api.deepseek.com
      FGA_MODEL=deepseek-v4-flash
    """

    config: FGAConfig

    def complete(
        self,
        actor_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        import litellm

        api_key = os.getenv("FGA_API_KEY") or os.getenv(self.config.api_key_env)
        if not api_key:
            raise FGAError(
                f"Missing API key. Set FGA_API_KEY or {self.config.api_key_env}."
            )

        model = self.config.model
        if "/" not in model:
            # A custom api_base implies an OpenAI-compatible endpoint. litellm
            # needs an explicit provider prefix to route the request there.
            model = f"openai/{model}"

        kwargs: dict[str, Any] = {
            "model": model,
            "api_base": self.config.base_url.rstrip("/"),
            "api_key": api_key,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            response = litellm.completion(**kwargs)
        except Exception as e:
            raise FGAError(f"Model request failed: {e}") from e

        choice = response.choices[0]
        msg = choice.message

        # DeepSeek / o-series models return chain-of-thought in a separate field.
        # Preserve it so subsequent turns can continue the same reasoning line.
        reasoning = getattr(msg, "reasoning_content", None) or None

        if msg.tool_calls:
            tcs: list[ToolCall] = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))
            return ModelResponse(tool_calls=tcs, reasoning_content=reasoning)

        return ModelResponse(content=msg.content or "", reasoning_content=reasoning)


@dataclass(slots=True)
class ScriptedModel:
    """Deterministic model for tests.

    handler receives (actor_id, messages, tools) and returns a ModelResponse.
    """

    handler: Callable[[str, list[dict[str, Any]], list[dict[str, Any]] | None], ModelResponse]

    def complete(
        self,
        actor_id: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        return self.handler(actor_id, messages, tools)
