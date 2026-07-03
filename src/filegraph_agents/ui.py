from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree


@dataclass
class _Node:
    actor_id: str
    depth: int
    parent: "_Node | None" = None
    children: list["_Node"] = field(default_factory=list)
    start_time: float = 0.0
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
    _seen_agents: set[str] = field(default_factory=set, init=False)

    # ------------------------------------------------------------------
    # helpers: agent display / colour
    # ------------------------------------------------------------------

    @staticmethod
    def _agent_kind(actor_id: str) -> str:
        """Return a kind key: 'main', 'verifier', 'file', or 'other'."""
        if actor_id == "__main__":
            return "main"
        if "verifier" in actor_id.lower():
            return "verifier"
        # Likely a file-agent if it looks like a path (contains / or .py)
        if "/" in actor_id or actor_id.endswith(".py"):
            return "file"
        return "other"

    @staticmethod
    def _agent_style(kind: str) -> str:
        return {
            "main": "bold cyan",
            "verifier": "bold magenta",
            "file": "bold green",
            "other": "bold white",
        }.get(kind, "bold white")

    @staticmethod
    def _agent_icon(kind: str) -> str:
        return {
            "main": "◆",
            "verifier": "▲",
            "file": "●",
            "other": "■",
        }.get(kind, "■")

    def _short(self, actor_id: str) -> str:
        """Return a short display label for a tree node."""
        if actor_id == "__main__":
            return "MainAgent"
        # For file paths show basename, keep full path as dim suffix
        if "/" in actor_id:
            parts = actor_id.rsplit("/", 1)
            return parts[-1]
        return actor_id

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

    # ------------------------------------------------------------------
    # tree rendering
    # ------------------------------------------------------------------

    def _is_subtree_done(self, node: _Node) -> bool:
        """Check whether *node* and all its descendants are in 'done' state."""
        if node.state != "done":
            return False
        stack = list(node.children)
        while stack:
            cur = stack.pop()
            if cur.state != "done":
                return False
            stack.extend(cur.children)
        return True

    def _node_label(self, node: _Node, collapsed: bool = False) -> Text:
        kind = self._agent_kind(node.actor_id)
        label = Text()
        icon = self._agent_icon(kind)
        if node.state == "done":
            label.append(f"{icon} ", style="green")
        else:
            label.append(f"{icon} ", style="yellow")

        display = self._short(node.actor_id)
        style = self._agent_style(kind)
        label.append(display, style=style)

        if node.start_time:
            elapsed = time.monotonic() - node.start_time
            if elapsed < 60:
                time_str = f"{elapsed:.1f}s"
            else:
                time_str = f"{elapsed/60:.1f}m"
            label.append(f"  {time_str}", style="dim")

        if collapsed:
            n = len(node.children)
            label.append(f"  [{n} child]", style="dim italic" if n else "dim")
        elif node.state == "done":
            label.append("  ✔", style="green dim")

        return label

    def _build_tree(self, node: _Node, tree: Tree) -> None:
        """Recursively add a node to a Rich Tree, collapsing done subtrees."""
        if self._is_subtree_done(node) and node is not self._root:
            # Collapse: show the node with a child count hint, don't recurse
            branch = tree.add(self._node_label(node, collapsed=True))
            return
        branch = tree.add(self._node_label(node))
        for child in node.children:
            self._build_tree(child, branch)

    def _render_tree(self) -> Panel | Text:
        """Build the tree as a standalone Panel (or "idle" text).

        This is used both by the live display and for static final output.
        """
        if self._root is None:
            return Text("idle", style="dim")
        tree = Tree(self._node_label(self._root))
        for child in self._root.children:
            self._build_tree(child, tree)
        return Panel(
            tree,
            title="Call Tree",
            title_align="center",
            border_style="yellow",
            padding=(0, 1),
        )

    def _active_count(self) -> int:
        """Count nodes whose agent is still thinking (active, not done)."""
        if self._root is None:
            return 0
        count = 0
        stack = [self._root]
        while stack:
            node = stack.pop()
            if node.state == "thinking":
                count += 1
            stack.extend(node.children)
        return count

    def _render(self):
        header = Text()
        header.append("FileGraph Agents ", style="bold cyan")
        active = self._active_count()
        total = len(self._seen_agents)
        if active:
            header.append(f"⚡ {active} active  ", style="yellow")
        if total:
            header.append(f"☰ {total} agents  ", style="dim")
        if self._model:
            header.append(f"· {self._model}", style="dim")

        body = self._render_tree()
        return Group(header, Text(""), body)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._render())

    # --- observer protocol -------------------------------------------------

    def on_start(self, *, model: str, repo: str, instruction: str) -> None:
        self._model = model
        self._seen_agents.add("__main__")
        self._root = _Node(actor_id="__main__", depth=0, start_time=time.monotonic())
        self.console.print(
            Text.assemble(("▸ ", "cyan"), (instruction.strip()[:200], "italic"))
        )
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=12,
            screen=True,
            transient=False,
        )
        self._live.start()

    def on_log_path(self, path: str) -> None:
        pass

    def on_talk(self, *, caller: str, target: str, depth: int) -> None:
        with self._lock:
            self._seen_agents.add(caller)
            self._seen_agents.add(target)
            if self._root is None:
                self._root = _Node(actor_id="__main__", depth=0, start_time=time.monotonic())
            parent = self._find(caller) or self._root
            parent.children.append(
                _Node(actor_id=target, depth=depth, parent=parent, start_time=time.monotonic())
            )
            self._refresh()

    def on_step(self, *, actor_id: str, step: int) -> None:
        with self._lock:
            self._seen_agents.add(actor_id)
            node = self._find(actor_id)
            if node is not None:
                node.start_time = time.monotonic()
                node.state = "thinking"
            self._refresh()

    def on_reply(self, *, actor_id: str) -> None:
        with self._lock:
            self._seen_agents.add(actor_id)
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
            # Print the final tree in the normal terminal buffer so it is
            # visible in scrollback (the alternate-screen Live display is gone).
            tree = self._render_tree()
            if not isinstance(tree, Text):
                self.console.print("")
                self.console.print(tree)
            # Final summary: total unique agents used
            total = len(self._seen_agents)
            self.console.print(
                Text(f"✅ 完成 — 共使用 {total} 个 agent", style="bold green")
            )

    def print_final(self, text: str) -> None:
        self.stop()
        from rich.markdown import Markdown

        self.console.print(Panel(Markdown(text), title="Result", border_style="cyan"))
