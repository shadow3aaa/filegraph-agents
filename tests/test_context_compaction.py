import tempfile
import unittest
from pathlib import Path

from filegraph_agents import FGAConfig, FGARuntime, ScriptedModel
from filegraph_agents.llm import ModelResponse


class ContextCompactionTests(unittest.TestCase):
    def _runtime(self, config):
        td = tempfile.mkdtemp()
        root = Path(td)
        (root / "a.py").write_text("A = 1\n", encoding="utf-8")
        # Summarizer just echoes a fixed summary; it is called with tools=None.
        rt = FGARuntime(
            root,
            model=ScriptedModel(lambda actor_id, messages, tools: ModelResponse(content="SUMMARY")),
            config=config,
        )
        return rt

    def test_no_compaction_below_threshold(self):
        # Window pinned via override; watermark = 10000 * 0.75 = 7500 tokens.
        config = FGAConfig(context_window_override=10_000, chars_per_token=1)
        rt = self._runtime(config)
        actor = rt.get_actor("a.py")
        actor.messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        before = list(actor.messages)
        actor._compact_history()
        self.assertEqual(actor.messages, before)

    def test_compaction_keeps_system_and_recent(self):
        # chars_per_token=1 => 1 char ~ 1 token. window=100 => watermark=75,
        # keep_budget = 100 * 0.4 = 40 tokens of the most recent turns.
        config = FGAConfig(context_window_override=100, chars_per_token=1)
        rt = self._runtime(config)
        actor = rt.get_actor("a.py")
        actor.messages = [{"role": "system", "content": "SYSTEM"}]
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            actor.messages.append({"role": role, "content": f"turn-{i} " + "x" * 20})

        actor._compact_history()

        self.assertEqual(actor.messages[0]["role"], "system")
        self.assertEqual(actor.messages[0]["content"], "SYSTEM")
        # Summary inserted right after system.
        self.assertIn("SUMMARY", actor.messages[1]["content"])
        # Most recent turn preserved verbatim.
        self.assertEqual(actor.messages[-1]["content"], "turn-19 " + "x" * 20)
        # Much smaller than the original 21 messages.
        self.assertLess(len(actor.messages), 21)

    def test_compaction_never_orphans_tool_message(self):
        config = FGAConfig(
            context_window_override=60, keep_recent_ratio=0.05, chars_per_token=1
        )
        rt = self._runtime(config)
        actor = rt.get_actor("a.py")
        # Build history where the desired cut would land on a tool message.
        actor.messages = [{"role": "system", "content": "SYSTEM"}]
        for i in range(6):
            actor.messages.append({"role": "user", "content": f"ask-{i} " + "y" * 30})
            actor.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": f"{i}", "type": "function",
                                "function": {"name": "ls", "arguments": "{}"}}],
            })
            actor.messages.append({"role": "tool", "tool_call_id": f"{i}", "content": "ok"})

        actor._compact_history()

        # The message immediately after the inserted summary must not be an
        # orphaned tool result (a tool message with no preceding assistant call).
        self.assertNotEqual(actor.messages[2]["role"], "tool")
        # Every tool message must be preceded by an assistant carrying tool_calls.
        for idx, m in enumerate(actor.messages):
            if m.get("role") == "tool":
                prev = actor.messages[idx - 1]
                self.assertTrue(
                    prev.get("role") == "assistant" and prev.get("tool_calls"),
                    f"orphan tool message at index {idx}",
                )


if __name__ == "__main__":
    unittest.main()
