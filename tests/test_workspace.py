import tempfile
import unittest
from pathlib import Path

from filegraph_agents.workspace import Workspace


class WorkspaceTests(unittest.TestCase):
    def test_read_write_search_ls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src" / "a.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
            ws = Workspace(root)

            self.assertEqual(ws.read_lines("src/a.py", 2, 2), "   2: two\n   3: three")
            result = ws.write_lines("src/a.py", 2, 2, "TWO")
            self.assertIn("src/a.py", result)
            self.assertIn("TWO", (root / "src" / "a.py").read_text(encoding="utf-8"))

            matches = ws.search("TWO")
            self.assertEqual(matches[0]["path"], "src/a.py")

            items = ws.ls("src")
            self.assertEqual(items, [{"path": "src/a.py", "type": "file"}])

    def test_create_delete(self):
        with tempfile.TemporaryDirectory() as td:
            ws = Workspace(td)
            path = ws.create_file("pkg/new.py")
            self.assertEqual(path, "pkg/new.py")
            self.assertTrue((Path(td) / "pkg" / "new.py").exists())
            deleted = ws.delete_file("pkg/new.py")
            self.assertEqual(deleted, "pkg/new.py")
            self.assertFalse((Path(td) / "pkg" / "new.py").exists())
    def test_search_max_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(3):
                (root / f"file{i}.txt").write_text(f"testfile {i}", encoding="utf-8")
            ws = Workspace(root)
            results = ws.search("testfile", max_results=1)
            self.assertLessEqual(len(results), 1)
            self.assertEqual(len(results), 1)

    def test_search_default_max_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(25):
                (root / f"file{i}.txt").write_text("common word", encoding="utf-8")
            ws = Workspace(root)
            results = ws.search("common")
            self.assertEqual(len(results), 20)

if __name__ == "__main__":
    unittest.main()
