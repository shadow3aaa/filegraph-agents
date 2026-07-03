import json
import tempfile
import unittest
from pathlib import Path

from filegraph_agents import FGAConfig, FGARuntime, ScriptedModel
from filegraph_agents.llm import ToolCall, ModelResponse


class ReentrantTalkTests(unittest.TestCase):
    def test_a_b_c_a_reentrant_same_actor_memory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "a.py").write_text("A = 1\n", encoding="utf-8")
            (root / "b.py").write_text("B = 1\n", encoding="utf-8")
            (root / "c.py").write_text("C = 1\n", encoding="utf-8")

            def handler(actor_id, messages, tools):
                has_tool_result = any(m["role"] == "tool" for m in messages)
                has_c_asks_a = any("C asks A" in (m.get("content", "") or "") for m in messages)

                if actor_id == "a.py":
                    # Resumed after B returned: A's own talk(B) tool result is now
                    # in the shared history.
                    if has_tool_result:
                        return ModelResponse(content="A got B final")
                    # Reentrant: C asked A while A is still waiting on B. No tool
                    # result yet, but the "C asks A" turn is in the same history.
                    if has_c_asks_a:
                        return ModelResponse(content="A answered C using same local context")
                    return ModelResponse(
                        tool_calls=[ToolCall(id="1", name="delegate", arguments={"path": "b.py", "task": "A asks B"})],
                    )

                if actor_id == "b.py":
                    if has_tool_result:
                        return ModelResponse(content="B got C final")
                    return ModelResponse(
                        tool_calls=[ToolCall(id="2", name="delegate", arguments={"path": "c.py", "task": "B asks C"})]
                    )

                if actor_id == "c.py":
                    if has_tool_result:
                        return ModelResponse(content="C got A final")
                    return ModelResponse(
                        tool_calls=[ToolCall(id="3", name="delegate", arguments={"path": "a.py", "task": "C asks A"})]
                    )

                raise AssertionError(actor_id)

            config = FGAConfig()
            rt = FGARuntime(root, model=ScriptedModel(handler), config=config)
            result = rt.talk(caller="user", target="a.py", prompt="start", depth=0)
            self.assertEqual(result, "A got B final")

            # A used ONE persistent conversation: both the original request-to-B
            # and the reentrant answer-to-C live in the same message history,
            # proving there was no A0/A1 context split.
            a = rt.get_actor("a.py")
            contents = [m.get("content") or "" for m in a.messages]
            self.assertTrue(any("C asks A" in c for c in contents))
            self.assertIn("A answered C using same local context", contents)
            self.assertIn("A got B final", contents)


if __name__ == "__main__":
    unittest.main()
