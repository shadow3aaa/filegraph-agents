from __future__ import annotations

from typing import Protocol


class FGAObserver(Protocol):
    """Sink for runtime events, used to drive a UI. All methods are optional
    no-ops by default via NullObserver."""

    def on_start(self, *, model: str, repo: str, instruction: str) -> None: ...
    def on_log_path(self, path: str) -> None: ...
    def on_talk(self, *, caller: str, target: str, depth: int) -> None: ...
    def on_step(self, *, actor_id: str, step: int, max_steps: int) -> None: ...
    def on_reply(self, *, actor_id: str, depth: int) -> None: ...
    def on_compact(self, *, actor_id: str, dropped: int) -> None: ...
    def on_error(self, message: str) -> None: ...


class NullObserver:
    """Observer that does nothing. Default when no UI is attached."""

    def on_start(self, *, model: str, repo: str, instruction: str) -> None:
        pass

    def on_log_path(self, path: str) -> None:
        pass

    def on_talk(self, *, caller: str, target: str, depth: int) -> None:
        pass

    def on_step(self, *, actor_id: str, step: int, max_steps: int) -> None:
        pass

    def on_reply(self, *, actor_id: str, depth: int) -> None:
        pass

    def on_compact(self, *, actor_id: str, dropped: int) -> None:
        pass

    def on_error(self, message: str) -> None:
        pass
