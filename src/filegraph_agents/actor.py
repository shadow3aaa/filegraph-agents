from __future__ import annotations

import time
from dataclasses import dataclass, field
import json
from typing import Any, TYPE_CHECKING

from .config import FGAConfig
from .errors import FGAError, PermissionDenied, ToolError
from .llm import ToolCall
from .prompts import (
    MAIN_SYSTEM,
    SUMMARIZE_SYSTEM,
    VERIFIER_SYSTEM,
    event_user_prompt,
    file_system,
    summarize_user_prompt,
)

if TYPE_CHECKING:
    from .runtime import FGARuntime


@dataclass(slots=True)
class TalkEvent:
    caller: str
    target: str
    prompt: str
    tx_id: str
    depth: int = 0
    # actor_ids currently on the logical call chain leading to this talk.
    # Used to detect reentrancy (a talk targeting an ancestor) so it runs
    # synchronously instead of deadlocking on that ancestor's mailbox.
    ancestors: frozenset[str] = frozenset()


@dataclass
class BaseActor:
    actor_id: str
    runtime: "FGARuntime"
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def config(self) -> FGAConfig:
        return self.runtime.config

    @property
    def system_prompt(self) -> str:
        raise NotImplementedError

    @property
    def default_ls_dir(self) -> str | None:
        return None

    def can_shell(self) -> bool:
        return False

    def can_read_write(self) -> bool:
        return False

    def can_plan(self) -> bool:
        return False

    def can_read(self) -> bool:
        """Read-only permission (VerifierActor has this without can_read_write)."""
        return self.can_read_write()

    def can_delegate(self) -> bool:
        """Whether this actor can talk to others and manage files."""
        return True

    def can_search(self) -> bool:
        """Whether this actor can search file contents (MainActor cannot)."""
        return True

    def _resolve_read_path(self, args: dict[str, Any]) -> str:
        return self.actor_id

    def _ensure_history(self) -> None:
        """Lazily seed the persistent conversation with the system prompt."""
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        chars = 0
        for m in messages:
            chars += len(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                chars += len(fn.get("name", "")) + len(fn.get("arguments", ""))
        return chars // max(1, self.config.chars_per_token)

    @staticmethod
    def _safe_cut_index(messages: list[dict[str, Any]], desired: int) -> int:
        """Return a cut index that does not split a tool-call group.

        The kept tail must begin on a self-contained turn, never on a bare tool
        result. We move the cut *backward* to the start of the group so the
        assistant tool_calls stay attached to their tool results in the tail.
        """
        i = min(desired, len(messages))
        n = len(messages)

        def invalid(idx: int) -> bool:
            if idx >= n:
                return False
            if messages[idx].get("role") == "tool":
                return True  # tail can't start on a bare tool result
            if idx > 0 and messages[idx - 1].get("tool_calls"):
                return True  # would split an assistant call from its results
            return False

        while i > 1 and invalid(i):
            i -= 1
        return i

    def _keep_recent_cut(self, keep_token_budget: int) -> int:
        """Index where the kept tail should start, walking back from the end
        until the tail's tokens exceed keep_token_budget (with a message floor)."""
        floor = self.config.keep_recent_messages_floor
        i = len(self.messages)
        tokens = 0
        kept = 0
        while i > 1:
            tokens += self._estimate_tokens([self.messages[i - 1]])
            kept += 1
            if tokens > keep_token_budget and kept >= floor:
                break
            i -= 1
        return i

    def _compact_history(self) -> None:
        """Summarize older turns when the conversation grows too large.

        The watermark and keep budget are derived from the model's real context
        window: compact when tokens exceed window * context_use_ratio, then keep
        system + the most recent turns fitting window * keep_recent_ratio. The
        middle is replaced with one summary. Tool-call groups are never split.
        """
        window = self.runtime.resolve_context_window()
        watermark = int(window * self.config.context_use_ratio)
        keep_budget = int(window * self.config.keep_recent_ratio)

        if self._estimate_tokens(self.messages) <= watermark:
            return
        if len(self.messages) <= self.config.keep_recent_messages_floor + 2:
            return

        head = self.messages[0:1]  # system
        desired_cut = self._keep_recent_cut(keep_budget)
        cut = self._safe_cut_index(self.messages, max(1, desired_cut))
        older = self.messages[1:cut]
        tail = self.messages[cut:]
        if not older:
            return

        transcript_parts: list[str] = []
        for m in older:
            role = m.get("role", "?")
            if m.get("tool_calls"):
                calls = ", ".join(
                    f"{tc['function']['name']}({tc['function']['arguments']})"
                    for tc in m["tool_calls"]
                )
                transcript_parts.append(f"[{role} tool_calls] {calls}")
            else:
                transcript_parts.append(f"[{role}] {m.get('content') or ''}")
        transcript = self._truncate("\n".join(transcript_parts))

        summary_msgs = [
            {"role": "system", "content": SUMMARIZE_SYSTEM},
            {"role": "user", "content": summarize_user_prompt(transcript)},
        ]
        response = self.runtime.complete_model(self.actor_id, summary_msgs, None)
        summary_text = response.content or "(summary unavailable)"

        self.runtime.observer.on_compact(actor_id=self.actor_id, dropped=len(older))
        self.messages[:] = (
            head
            + [
                {
                    "role": "user",
                    "content": f"Summary of earlier conversation:\n{summary_text}",
                }
            ]
            + tail
        )

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        """Build OpenAI-compatible tool definitions from actor capabilities."""
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "ls",
                    "description": "List files and directories in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (optional)",
                            }
                        },
                    },
                },
            },
        ]

        if self.can_search():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for literal text across the workspace. Returns file paths with match counts, never code content.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Literal text to search for",
                                },
                                "max_results": {
                                    "type": "integer",
                                    "description": "Max results (default 20)",
                                },
                            },
                            "required": ["content"],
                        },
                    },
                },
            )

        if self.can_shell():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "Run shell commands (tests, builds, type checks, lint, or verifier commands only).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Shell command to execute",
                                },
                            },
                            "required": ["command"],
                        },
                    },
                }
            )

        if self.can_delegate():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "delegate",
                        "description": "Delegate a task to a file agent. This transfers ownership — do NOT ask questions, only assign work.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Target file path",
                                },
                                "task": {
                                    "type": "string",
                                    "description": "The task to own and execute",
                                },
                            },
                            "required": ["path", "task"],
                        },
                    },
                },
            )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "create_file",
                        "description": "Create an empty file and spawn its file agent. Then talk to it to write content.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path for the new file",
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
            )
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "delete_file",
                        "description": "Delete a file and stop its file agent.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "Path to delete",
                                },
                            },
                            "required": ["path"],
                        },
                    },
                },
            )

        if self.can_plan():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "plan",
                        "description": "Record your current decomposition plan for later reference. Call this whenever you settle on a multi-step strategy so you can recall it in future steps.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "plan": {
                                    "type": "string",
                                    "description": "The plan text (steps, targets, reasoning)",
                                },
                            },
                            "required": ["plan"],
                        },
                    },
                }
            )

        if self.can_read():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read lines from your own file.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "start_line": {
                                    "type": "integer",
                                    "description": "First line (1-based)",
                                },
                                "offset": {
                                    "type": "integer",
                                    "description": "Number of lines to read",
                                },
                            },
                            "required": ["start_line", "offset"],
                        },
                    },
                }
            )

        if self.can_read_write():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "write",
                        "description": "Write content to your own file. Replaces inclusive [start_line,end_line]. Use end_line=start_line-1 for insertion.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "start_line": {
                                    "type": "integer",
                                    "description": "First line to replace (1-based)",
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": "Last line to replace (inclusive)",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "New content",
                                },
                            },
                            "required": ["start_line", "end_line", "content"],
                        },
                    },
                }
            )

        return tools

    def handle_talk(self, event: TalkEvent) -> str:
        # The actor keeps one persistent ReAct conversation. Each incoming talk
        # is appended as a new user turn to that same history. This is what
        # makes reentrancy safe: a nested talk to this same actor appends to and
        # runs over the very same message list, so there is no context split.
        self._ensure_history()
        self.messages.append(
            {
                "role": "user",
                "content": event_user_prompt(caller=event.caller, task=event.prompt),
            }
        )

        # No per-actor step cap: an actor keeps looping until it produces a
        # reply. The empty-response guard below prevents a no-progress spin, and
        # cyclic talks are broken by reentrancy (ancestors). Context growth is
        # handled by compaction, so a long ReAct loop stays within the window.
        step = 0
        while True:
            step += 1
            self.runtime.observer.on_step(actor_id=self.actor_id, step=step)

            self._compact_history()
            tools = self._get_tool_definitions()
            max_attempts = 10
            for attempt in range(max_attempts):
                try:
                    response = self.runtime.complete_model(
                        self.actor_id, self.messages, tools
                    )
                    break
                except FGAError:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(2 ** attempt)

            # Log raw conversation
            self.runtime.log_raw(
                actor_id=self.actor_id,
                step=step,
                caller=event.caller,
                messages=self.messages,
                response=response,
            )

            # Model replied directly — this is the final answer
            if response.content is not None:
                self.messages.append({"role": "assistant", "content": response.content})
                self.runtime.observer.on_reply(actor_id=self.actor_id)
                return response.content

            # Model wants to call tools
            if response.tool_calls:
                assistant_tc = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": assistant_tc,
                    }
                )

                self.runtime.observer.on_pause(actor_id=self.actor_id)
                results = self._execute_tool_calls(response.tool_calls, event)
                self.runtime.observer.on_resume(actor_id=self.actor_id)
                for tc in response.tool_calls:
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": self._truncate(
                                json.dumps(results[tc.id], ensure_ascii=False)
                            ),
                        }
                    )
                continue

            # Neither content nor tool calls: nothing to do, and looping again
            # would just repeat. Treat it as an empty reply to avoid spinning.
            self.runtime.observer.on_reply(actor_id=self.actor_id)
            return ""

    def _execute_tool_calls(
        self, tool_calls: list[ToolCall], event: TalkEvent
    ) -> dict[str, Any]:
        """Run a step's tool calls, dispatching multiple delegates in parallel.

        When the model emits several delegate() calls in one step, they are sent
        to their target actors together so those file-agents run concurrently on
        their own threads. Non-delegate tools (read/write/shell/ls/...) have
        side effects or touch this actor's own file, so they run serially.
        """
        results: dict[str, Any] = {}
        delegate_calls = [tc for tc in tool_calls if tc.name == "delegate"]

        # Serial tools first (keep deterministic ordering for side effects).
        for tc in tool_calls:
            if tc.name == "delegate":
                continue
            results[tc.id] = self._run_single_tool(tc, event)

        if len(delegate_calls) == 1:
            result = self._run_single_tool(delegate_calls[0], event)
            results[delegate_calls[0].id] = result
        elif delegate_calls:
            try:
                # Ensure each task is non-empty
                for tc in delegate_calls:
                    if not str(tc.arguments.get("task", "")).strip():
                        raise ToolError("delegate task must not be empty")
                requests = [
                    (
                        str(tc.arguments.get("path", "")),
                        str(tc.arguments.get("task", "")),
                    )
                    for tc in delegate_calls
                ]
                replies = self.runtime.talk_many(
                    caller=self.actor_id,
                    requests=requests,
                    tx_id=event.tx_id,
                    depth=event.depth + 1,
                    ancestors=event.ancestors,
                )
            except FGAError as e:
                for tc in delegate_calls:
                    results[tc.id] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            else:
                for tc, (path, reply) in zip(delegate_calls, replies):
                    results[tc.id] = {"ok": True, "path": path, "response": reply}

        return results

    def _run_single_tool(self, tc: ToolCall, event: TalkEvent) -> Any:
        try:
            return self._dispatch_action(tc.name, tc.arguments, event)
        except FGAError as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        except Exception as e:
            return {"ok": False, "error": f"unexpected {type(e).__name__}: {e}"}

    def _truncate(self, text: str) -> str:
        limit = self.config.tool_result_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n<truncated {len(text) - limit} chars>"

    def _dispatch_action(
        self, action: str, args: dict[str, Any], event: TalkEvent
    ) -> Any:
        if action == "ls":
            return self.runtime.workspace.ls(
                args.get("path"), default_dir=self.default_ls_dir
            )
        if action == "search":
            return self.runtime.workspace.search(
                str(args.get("content", "")), max_results=args.get("max_results", 20)
            )
        if action == "create_file":
            path = self.runtime.workspace.create_file(str(args.get("path", "")))
            self.runtime.ensure_file_actor(path)
            return {"ok": True, "path": path}
        if action == "delete_file":
            path = self.runtime.workspace.delete_file(str(args.get("path", "")))
            self.runtime.stop_file_actor(path)
            return {"ok": True, "path": path}
        if action == "delegate":
            path = str(args.get("path", ""))
            task = str(args.get("task", ""))
            if not task.strip():
                raise ToolError("delegate task must not be empty")
            response = self.runtime.talk(
                caller=self.actor_id,
                target=path,
                prompt=task,
                tx_id=event.tx_id,
                depth=event.depth + 1,
                ancestors=event.ancestors,
            )
            return {"ok": True, "path": path, "response": response}
        if action == "plan":
            if not self.can_plan():
                raise PermissionDenied("plan is only available to MainAgent")
            return self.runtime.add_plan(str(args.get("plan", "")))
        if action == "shell":
            if not self.can_shell():
                raise PermissionDenied("shell is only available to MainAgent")
            return self.runtime.shell(str(args.get("command", "")))
        if action == "read":
            if not self.can_read():
                raise PermissionDenied("read is not available for this actor")
            return {
                "ok": True,
                "content": self.runtime.workspace.read_lines(
                    self._resolve_read_path(args),
                    int(args.get("start_line", 1)),
                    int(args.get("offset", 120)),
                ),
            }
        if action == "write":
            if not self.can_read_write():
                raise PermissionDenied("write is only available to FileAgent")
            result = self.runtime.workspace.write_lines(
                self.actor_id,
                int(args.get("start_line", 1)),
                int(args.get("end_line", 0)),
                str(args.get("content", "")),
            )
            return {"ok": True, "result": result}

        raise ToolError(f"unknown action: {action}")


@dataclass
class MainActor(BaseActor):
    @property
    def system_prompt(self) -> str:
        return MAIN_SYSTEM

    def can_shell(self) -> bool:
        return True

    def can_plan(self) -> bool:
        return True

    def can_search(self) -> bool:
        return False


@dataclass
class FileActor(BaseActor):
    @property
    def system_prompt(self) -> str:
        return file_system(self.actor_id)

    @property
    def default_ls_dir(self) -> str | None:
        # List the file's containing directory by default.
        if "/" in self.actor_id:
            return self.actor_id.rsplit("/", 1)[0]
        return "."

    def can_read_write(self) -> bool:
        return True

    def can_read(self) -> bool:
        return True


@dataclass
class VerifierActor(BaseActor):
    @property
    def system_prompt(self) -> str:
        return VERIFIER_SYSTEM

    def can_shell(self) -> bool:
        return True

    def can_read(self) -> bool:
        return True

    def can_delegate(self) -> bool:
        return False

    def _resolve_read_path(self, args: dict[str, Any]) -> str:
        path = args.get("path")
        if not path:
            raise ToolError("path argument is required for VerifierActor read")
        return path

    def _get_tool_definitions(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "ls",
                    "description": "List files and directories in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path (optional)",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search for literal text across the workspace. Returns file paths with match counts, never code content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Literal text to search for",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Max results (default 20)",
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read lines from any file in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "File path to read",
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "First line (1-based)",
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Number of lines to read",
                            },
                        },
                        "required": ["path", "start_line", "offset"],
                    },
                },
            },
        ]
        if self.can_shell():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "description": "Run shell commands (tests, builds, type checks, lint, or verifier commands only).",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "Shell command to execute",
                                },
                            },
                            "required": ["command"],
                        },
                    },
                }
            )
        return tools
