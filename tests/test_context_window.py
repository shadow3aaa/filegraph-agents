import tempfile
import unittest
from pathlib import Path

from filegraph_agents import FGAConfig, FGARuntime, ScriptedModel
from filegraph_agents.llm import ModelResponse


def _runtime(config):
    td = tempfile.mkdtemp()
    root = Path(td)
    (root / "a.py").write_text("A = 1\n", encoding="utf-8")
    return FGARuntime(
        root,
        model=ScriptedModel(lambda *a: ModelResponse(content="x")),
        config=config,
    )


class ContextWindowTests(unittest.TestCase):
    def test_override_wins(self):
        rt = _runtime(FGAConfig(model="gpt-4o", context_window_override=5000))
        self.assertEqual(rt.resolve_context_window(), 5000)

    def test_catalog_lookup(self):
        # gpt-4o is in litellm's catalog (128k input tokens).
        rt = _runtime(FGAConfig(model="gpt-4o"))
        self.assertGreaterEqual(rt.resolve_context_window(), 100_000)

    def test_catalog_lookup_strips_provider_prefix(self):
        rt = _runtime(FGAConfig(model="openai/gpt-4o"))
        self.assertGreaterEqual(rt.resolve_context_window(), 100_000)

    def test_fallback_for_unknown_model(self):
        rt = _runtime(
            FGAConfig(model="totally-made-up-model-xyz", context_window_fallback=17000)
        )
        self.assertEqual(rt.resolve_context_window(), 17000)

    def test_result_is_cached(self):
        rt = _runtime(FGAConfig(model="gpt-4o"))
        first = rt.resolve_context_window()
        self.assertEqual(rt._context_window, first)
        self.assertEqual(rt.resolve_context_window(), first)


if __name__ == "__main__":
    unittest.main()
