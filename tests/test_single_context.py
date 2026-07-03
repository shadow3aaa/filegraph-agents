import tempfile
import unittest
from pathlib import Path

from filegraph_agents import FGAConfig, FGARuntime, ScriptedModel
from filegraph_agents.llm import ModelResponse, ToolCall


class SingleContextTests(unittest.TestCase):
    def test_same_actor_shares_one_context_across_talks(self):
        """Two talks to the SAME file-agent must share one message history,
        never spawn separate context versions (no A0/A1 split)."""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("X = 1\n", encoding="utf-8")

            def handler(actor_id, messages, tools):
                if actor_id == "__main__":
                    tool_rounds = sum(1 for m in messages if m["role"] == "tool")
                    if tool_rounds == 0:
                        return ModelResponse(tool_calls=[
                        ToolCall(id="1", name="delegate",
                                 arguments={"path": "a.py", "task": "first"})
                        ])
                    if tool_rounds == 1:
                        return ModelResponse(tool_calls=[
                        ToolCall(id="2", name="delegate",
                                 arguments={"path": "a.py", "task": "second"})
                        ])
                    return ModelResponse(content="done")

                # a.py: on the SECOND talk, its history must already contain the
                # first exchange. If a fresh context were created per talk, the
                # earlier "first" turn would be missing.
                user_turns = [m for m in messages if m["role"] == "user"]
                if any("second" in (m.get("content") or "") for m in messages):
                    saw_first = any("first" in (m.get("content") or "") for m in messages)
                    assert saw_first, "second talk lost the first talk's context!"
                    return ModelResponse(content=f"a.py saw {len(user_turns)} user turns")
                return ModelResponse(content="a.py handled first")

            rt = FGARuntime(root, model=ScriptedModel(handler), config=FGAConfig())
            self.assertEqual(rt.run("go"), "done")

            # The actor kept exactly one growing message history.
            a = rt.get_actor("a.py")
            contents = [m.get("content") or "" for m in a.messages]
            self.assertTrue(any("first" in c for c in contents))
            self.assertTrue(any("second" in c for c in contents))
            # Both replies live in the same single history.
            self.assertTrue(any("a.py handled first" in c for c in contents))
            self.assertTrue(any("user turns" in c for c in contents))


if __name__ == "__main__":
    unittest.main()
