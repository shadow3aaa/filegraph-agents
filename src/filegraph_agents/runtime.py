from __future__ import annotations

import datetime
import os
import threading
from concurrent.futures import Future
from dataclasses import dataclass, field
import shlex
import subprocess
import uuid
from pathlib import Path
from .actor import BaseActor, FileActor, MainActor, TalkEvent
from .config import FGAConfig
from .errors import PermissionDenied, ToolError
from .llm import ChatModel, LiteLLMModel
from .mailbox import ActorMailbox
from .observer import FGAObserver, NullObserver
from .workspace import Workspace


@dataclass
class FGARuntime:
    """Actor-style runtime for FileGraph Agents v0.

    The runtime is intentionally synchronous and simple. talk() is one-shot and
    returns one reply, but actors are re-entrant because the same FileActor
    instance can receive another talk while an earlier talk is waiting on the
    Python stack. The actor's persistent message history is shared across these
    nested events, so a reentrant talk continues the same ReAct conversation.
    """

    root: str | Path
    model: ChatModel | None = None
    config: FGAConfig = field(default_factory=FGAConfig.from_env)
    observer: FGAObserver = field(default_factory=NullObserver)

    def __post_init__(self) -> None:
        self.workspace = Workspace(self.root)
        if self.model is None:
            self.model = LiteLLMModel(self.config)
        self.actors: dict[str, BaseActor] = {}
        self.mailboxes: dict[str, ActorMailbox] = {}
        self._actors_lock = threading.Lock()
        self._log_lock = threading.Lock()
        # Caps simultaneous in-flight LLM requests. The slot is only held around
        # the actual API call, never while an actor waits on nested talks, so a
        # talk chain longer than the limit cannot deadlock.
        n = self.config.max_parallel_talks
        self._llm_semaphore = threading.Semaphore(n) if n and n > 0 else None
        self.actors["__main__"] = MainActor("__main__", self)
        # Init timestamped log file. FGA_LOG_DIR lets callers (e.g. sandboxed
        # eval harnesses) redirect logs out of the repo so they don't pollute a
        # diff-based submission; defaults to <repo>/outputs.
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_root = os.getenv("FGA_LOG_DIR")
        log_dir = Path(log_root) if log_root else Path(self.root) / "outputs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._log_path: Path = log_dir / f"{ts}.txt"
        self._log_path.write_text("", encoding="utf-8")
        self.observer.on_log_path(str(self._log_path))

    @property
    def main(self) -> MainActor:
        return self.actors["__main__"]  # type: ignore[return-value]

    def ensure_file_actor(self, path: str) -> FileActor:
        rel = self.workspace.rel(path)
        if not self.workspace.exists(rel):
            raise ToolError(f"file actor target does not exist: {rel}")
        with self._actors_lock:
            if rel not in self.actors:
                self.actors[rel] = FileActor(rel, self)
            actor = self.actors[rel]
        if not isinstance(actor, FileActor):
            raise ToolError(f"actor is not a file actor: {rel}")
        return actor

    def stop_file_actor(self, path: str) -> None:
        rel = self.workspace.rel(path)
        with self._actors_lock:
            self.actors.pop(rel, None)
            mb = self.mailboxes.pop(rel, None)
        if mb is not None:
            mb.shutdown()

    def get_actor(self, actor_id: str) -> BaseActor:
        if actor_id == "__main__":
            return self.main
        return self.ensure_file_actor(actor_id)

    def _mailbox_for(self, actor_id: str) -> ActorMailbox:
        actor = self.get_actor(actor_id)
        with self._actors_lock:
            mb = self.mailboxes.get(actor_id)
            if mb is None:
                mb = ActorMailbox(actor)
                self.mailboxes[actor_id] = mb
        return mb

    def _make_event(
        self, *, caller: str, target: str, prompt: str, tx_id: str | None,
        depth: int, ancestors: frozenset[str],
    ) -> TalkEvent:
        if not prompt.strip():
            raise ToolError("talk prompt must not be empty")
        return TalkEvent(
            caller=caller,
            target=target,
            prompt=prompt,
            tx_id=tx_id or str(uuid.uuid4()),
            depth=depth,
            ancestors=ancestors | {caller},
        )

    def talk(
        self,
        *,
        caller: str,
        target: str,
        prompt: str,
        tx_id: str | None = None,
        depth: int = 0,
        ancestors: frozenset[str] = frozenset(),
    ) -> str:
        """Send one message and block for the reply.

        A talk targeting an actor already on this logical call chain (an
        ancestor) is run synchronously in-thread to preserve reentrancy and
        avoid deadlocking on that ancestor's parked mailbox. Any other talk is
        posted to the target's mailbox and awaited via its future, which lets
        multiple talks issued together run on their own actor threads in
        parallel (see talk_many).
        """
        event = self._make_event(
            caller=caller, target=target, prompt=prompt, tx_id=tx_id,
            depth=depth, ancestors=ancestors,
        )
        self.observer.on_talk(caller=caller, target=target, depth=depth)

        if target in event.ancestors or target == caller:
            # Reentrant: the target is waiting up the stack. Run on this thread
            # against the same shared actor object (its worker is parked).
            return self.get_actor(target).handle_talk(event)

        return self._mailbox_for(target).post(event).result()

    def talk_many(
        self,
        *,
        caller: str,
        requests: list[tuple[str, str]],
        tx_id: str | None = None,
        depth: int = 0,
        ancestors: frozenset[str] = frozenset(),
    ) -> list[tuple[str, str]]:
        """Dispatch several talks concurrently and return (target, reply) pairs.

        Reentrant targets are handled synchronously; the rest are posted to
        their mailboxes up front and awaited together, so independent file-agents
        actually work in parallel on their own threads.
        """
        pending: list[tuple[str, Future[str] | None, TalkEvent]] = []
        for target, prompt in requests:
            event = self._make_event(
                caller=caller, target=target, prompt=prompt, tx_id=tx_id,
                depth=depth, ancestors=ancestors,
            )
            self.observer.on_talk(caller=caller, target=target, depth=depth)
            if target in event.ancestors or target == caller:
                pending.append((target, None, event))  # run inline below
            else:
                pending.append((target, self._mailbox_for(target).post(event), event))

        results: list[tuple[str, str]] = []
        for target, fut, event in pending:
            if fut is None:
                results.append((target, self.get_actor(target).handle_talk(event)))
            else:
                results.append((target, fut.result()))
        return results

    def resolve_context_window(self) -> int:
        """Context window (in tokens) for the configured model.

        Prefers an explicit override, then litellm's model catalog, then the
        configured fallback. Cached so the lookup happens at most once.
        """
        cached = getattr(self, "_context_window", None)
        if cached is not None:
            return cached
        if self.config.context_window_override > 0:
            window = self.config.context_window_override
        else:
            window = self._lookup_context_window() or self.config.context_window_fallback
        self._context_window = window
        return window

    def _lookup_context_window(self) -> int | None:
        try:
            import litellm

            model = self.config.model
            for candidate in (model, model.split("/", 1)[-1]):
                try:
                    info = litellm.get_model_info(candidate)
                except Exception:
                    continue
                tokens = info.get("max_input_tokens") or info.get("max_tokens")
                if tokens:
                    return int(tokens)
        except Exception:
            pass
        return None

    def complete_model(self, actor_id, messages, tools):
        """Call the model, bounded by the global in-flight request limit.

        The semaphore slot is held only for the duration of this synchronous
        call, so it is released before the actor blocks on any nested talk.
        """
        if self._llm_semaphore is None:
            return self.model.complete(actor_id, messages, tools)
        with self._llm_semaphore:
            return self.model.complete(actor_id, messages, tools)

    def shutdown(self) -> None:
        with self._actors_lock:
            mailboxes = list(self.mailboxes.values())
            self.mailboxes.clear()
        for mb in mailboxes:
            mb.shutdown()
    def log_raw(self, actor_id: str, step: int, caller: str,
                messages: list[dict], response: object) -> None:
        """Append raw agent conversation to the run log file."""
        from .llm import ModelResponse
        resp_content: str
        resp_tool_calls: list = []
        if isinstance(response, ModelResponse):
            resp_content = response.content or ""
            if response.tool_calls:
                for tc in response.tool_calls:
                    resp_tool_calls.append(f"{tc.name}({tc.arguments})")
        else:
            resp_content = str(response)
        lines = [
            "=" * 60,
            f"agent: {actor_id} | step {step} | caller: {caller}",
            "=" * 60,
        ]
        for msg in messages:
            role = msg.get("role", "?").upper()
            content = msg.get("content", "")
            if role == "ASSISTANT" and msg.get("tool_calls"):
                lines.append(f"--- {role} (tool_calls) ---")
                for tc in msg["tool_calls"]:
                    lines.append(f"  {tc['function']['name']}({tc['function']['arguments']})")
            else:
                lines.append(f"--- {role} ---")
                lines.append(str(content or ""))
            lines.append("")
        if resp_tool_calls:
            lines.append("--- RESPONSE (tool_calls) ---")
            for rc in resp_tool_calls:
                lines.append(f"  {rc}")
        else:
            lines.append("--- RESPONSE ---")
            lines.append(resp_content)
        lines.append("")
        with self._log_lock:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

    def run(self, instruction: str) -> str:
        try:
            return self.main.handle_talk(
                TalkEvent(
                    caller="user",
                    target="__main__",
                    prompt=instruction,
                    tx_id=str(uuid.uuid4()),
                    depth=0,
                    ancestors=frozenset({"user"}),
                )
            )
        finally:
            self.shutdown()

    def shell(self, command: str) -> dict[str, object]:
        command = command.strip()
        if not command:
            raise ToolError("shell command must not be empty")
        self._check_shell_command(command)
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.workspace.root),
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.shell_timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return {
                "ok": False,
                "timeout": True,
                "returncode": None,
                "stdout": (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else "",
                "stderr": (e.stderr or "")[-4000:] if isinstance(e.stderr, str) else "",
            }
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-8000:],
            "stderr": completed.stderr[-8000:],
        }

    def _check_shell_command(self, command: str) -> None:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError as e:
            raise ToolError(f"invalid shell command: {e}") from e
        forbidden = set(self.config.forbidden_shell_commands)
        for tok in tokens:
            base = Path(tok).name
            if base in forbidden:
                raise PermissionDenied(
                    f"shell command '{base}' is forbidden; use FGA tools instead"
                )
