import tempfile
import threading
import time
import unittest
from pathlib import Path

from filegraph_agents import FGAConfig, FGARuntime, ScriptedModel
from filegraph_agents.llm import ModelResponse, ToolCall


class ParallelTalkTests(unittest.TestCase):
    def test_multiple_talks_run_concurrently(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            targets = ["a.py", "b.py", "c.py", "d.py"]
            for name in targets:
                (root / name).write_text("X = 1\n", encoding="utf-8")

            active = 0
            max_active = 0
            lock = threading.Lock()
            barrier = threading.Barrier(len(targets), timeout=5)

            def handler(actor_id, messages, tools):
                if actor_id == "__main__":
                    # One step emitting four talks at once.
                    if not any(m["role"] == "tool" for m in messages):
                        return ModelResponse(tool_calls=[
                            ToolCall(id=f"t{i}", name="delegate",
                                     arguments={"path": t, "task": "work"})
                            for i, t in enumerate(targets)
                        ])
                    return ModelResponse(content="all done")

                # Each file-agent records concurrency, then waits on a barrier so
                # the test only passes if all four are alive at the same time.
                nonlocal active, max_active
                with lock:
                    active += 1
                    max_active = max(max_active, active)
                try:
                    barrier.wait()
                finally:
                    with lock:
                        active -= 1
                return ModelResponse(content=f"{actor_id} ok")

            config = FGAConfig(max_parallel_talks=4)
            rt = FGARuntime(root, model=ScriptedModel(handler), config=config)
            result = rt.run("fan out")

            self.assertEqual(result, "all done")
            # If execution were serial, the barrier would time out (only 1 active
            # at a time). Reaching 4 concurrent proves real parallelism.
            self.assertEqual(max_active, len(targets))

    def test_parallel_results_map_to_correct_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ["a.py", "b.py"]:
                (root / name).write_text("X = 1\n", encoding="utf-8")

            def handler(actor_id, messages, tools):
                if actor_id == "__main__":
                    if not any(m["role"] == "tool" for m in messages):
                        return ModelResponse(tool_calls=[
                            ToolCall(id="ta", name="delegate", arguments={"path": "a.py", "task": "x"}),
                            ToolCall(id="tb", name="delegate", arguments={"path": "b.py", "task": "x"}),
                        ])
                    # Verify each tool result carries the right file's reply.
                    tool_msgs = [m for m in messages if m["role"] == "tool"]
                    joined = " ".join(m["content"] for m in tool_msgs)
                    assert "reply-from-a.py" in joined and "reply-from-b.py" in joined
                    return ModelResponse(content="ok")
                # Small stagger so ordering can't accidentally line up.
                if actor_id == "a.py":
                    time.sleep(0.02)
                return ModelResponse(content=f"reply-from-{actor_id}")

            rt = FGARuntime(root, model=ScriptedModel(handler), config=FGAConfig())
            self.assertEqual(rt.run("go"), "ok")


if __name__ == "__main__":
    unittest.main()
