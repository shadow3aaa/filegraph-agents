import tempfile
import threading
import time
import unittest
from pathlib import Path

from filegraph_agents import FGAConfig, FGARuntime, ScriptedModel
from filegraph_agents.llm import ModelResponse, ToolCall


class ParallelLimitTests(unittest.TestCase):
    def _run_fanout(self, limit, n_targets):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            targets = [f"f{i}.py" for i in range(n_targets)]
            for name in targets:
                (root / name).write_text("X = 1\n", encoding="utf-8")

            in_flight = 0
            peak = 0
            lock = threading.Lock()

            def handler(actor_id, messages, tools):
                nonlocal in_flight, peak
                with lock:
                    in_flight += 1
                    peak = max(peak, in_flight)
                try:
                    if actor_id == "__main__":
                        if not any(m["role"] == "tool" for m in messages):
                            return ModelResponse(tool_calls=[
                                ToolCall(id=f"t{i}", name="delegate",
                                         arguments={"path": t, "task": "go"})
                                for i, t in enumerate(targets)
                            ])
                        return ModelResponse(content="done")
                    # Hold the slot briefly so overlap is observable.
                    time.sleep(0.02)
                    return ModelResponse(content=f"{actor_id} ok")
                finally:
                    with lock:
                        in_flight -= 1

            config = FGAConfig(max_parallel_talks=limit)
            rt = FGARuntime(root, model=ScriptedModel(handler), config=config)
            result = rt.run("fan out")
            self.assertEqual(result, "done")
            return peak

    def test_semaphore_caps_concurrent_requests(self):
        # 10 file-agents fanned out, but at most 3 LLM calls in flight at once.
        peak = self._run_fanout(limit=3, n_targets=10)
        self.assertLessEqual(peak, 3)

    def test_disabled_limit_allows_full_fanout(self):
        # limit <= 0 disables the cap; all fan-out targets can overlap.
        peak = self._run_fanout(limit=0, n_targets=6)
        self.assertGreater(peak, 1)


if __name__ == "__main__":
    unittest.main()
