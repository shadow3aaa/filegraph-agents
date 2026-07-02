from __future__ import annotations

import threading
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text
from rich.tree import Tree


@dataclass
class _Node:
    actor_id: str
    depth: int
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    step: int = 0
    max_steps: int = 0
    state: str = "thinking"  # thinking | done


@dataclass
class RichObserver:
    """Live TUI showing the agent network as a talk tree.

    The tree mirrors the talk relationships: one caller may fan out to several
    file-agents (e.g. when the model emits multiple talk tool_calls in a single
    step). Those appear as sibling children under the caller and, with the
    mailbox runtime, actually run in parallel on their own actor threads.

    Observer callbacks arrive from multiple actor threads, so all state mutation
    and Live updates are serialized behind a lock.

    Only the dynamic network state is shown while running; the final reply is
    printed by the CLI after the Live view is torn down.
    """

    console: Console = field(default_factory=Console)
    _live: Live | None = field(default=None, init=False)
    _root: _Node | None = field(default=None, init=False)
    _model: str = field(default="", init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def _short(self, actor_id: str) -> str:
        if actor_id == "__main__":
            return "MainAgent"
        return actor_id.rsplit("/", 1)[-1]

    def _find(self, actor_id: str, node: _Node | None = None) -> _Node | None:
        """Depth-first search for the most relevant node with this actor_id.

        Prefers a still-active (thinking) node so status updates land on the
        live invocation rather than a completed one.
        """
        node = node or self._root
        if node is None:
            return None
        match: _Node | None = None
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.actor_id == actor_id:
                if cur.state == "thinking":
                    return cur
                match = match or cur
            stack.extend(cur.children)
        return match

    def _node_label(self, node: _Node) -> Text:
        label = Text()
        if node.state == "done":
            label.append("● ", style="green")
        else:
            label.append("◆ ", style="yellow")
        label.append(self._short(node.actor_id), style="bold")
        if node.max_steps:
            label.append(f"  step {node.step}/{node.max_steps}", style="dim")
        if node.state == "done":
            label.append("  replied", style="green dim")
        return label

    def _build_tree(self, node: _Node, tree: Tree) -> None:
        branch = tree.add(self._node_label(node))
        for child in node.children:
            self._build_tree(child, branch)

    def _render(self):
        header = Text()
        header.append("FileGraph Agents ", style="bold cyan")
        if self._model:
            header.append(f"· {self._model}", style="dim")

        if self._root is None:
            body: Text | Tree = Text("idle", style="dim")
        else:
            tree = Tree(self._node_label(self._root))
            for child in self._root.children:
                self._build_tree(child, tree)
            body = tree
        return Group(header, Text(""), body)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # --- observer protocol -------------------------------------------------

    def on_start(self, *, model: str, repo: str, instruction: str) -> None:
        self._model = model
        self._root = _Node(actor_id="__main__", depth=0)
        self.console.print(
            Text.assemble(("▸ ", "cyan"), (instruction.strip()[:200], "italic"))
        )
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def on_log_path(self, path: str) -> None:
        pass

    def on_talk(self, *, caller: str, target: str, depth: int) -> None:
        with self._lock:
            if self._root is None:
                self._root = _Node(actor_id="__main__", depth=0)
            parent = self._find(caller) or self._root
            parent.children.append(_Node(actor_id=target, depth=depth, parent=parent))
            self._refresh()

    def on_step(self, *, actor_id: str, step: int, max_steps: int) -> None:
        with self._lock:
            node = self._find(actor_id)
            if node is not None:
                node.step = step
                node.max_steps = max_steps
                node.state = "thinking"
            self._refresh()

    def on_reply(self, *, actor_id: str, depth: int) -> None:
        with self._lock:
            node = self._find(actor_id)
            if node is not None:
                node.state = "done"
            self._refresh()

    def on_compact(self, *, actor_id: str, dropped: int) -> None:
        with self._lock:
            self._refresh()

    def on_error(self, message: str) -> None:
        self.stop()
        self.console.print(Text(f"✗ {message}", style="bold red"))

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def print_final(self, text: str) -> None:
        self.stop()
        from rich.panel import Panel
        from rich.markdown import Markdown

        self.console.print(Panel(Markdown(text), title="Result", border_style="cyan"))
