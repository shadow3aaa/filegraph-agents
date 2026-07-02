import unittest
from unittest.mock import MagicMock

from filegraph_agents.actor import BaseActor, MainActor, FileActor
from filegraph_agents.errors import PermissionDenied


class TestShellPermissions(unittest.TestCase):
    def setUp(self):
        self.mock_runtime = MagicMock()
        self.mock_runtime.model = MagicMock()
        self.mock_runtime.workspace = MagicMock()
        self.mock_runtime.shell = MagicMock(return_value="ok")
        self.mock_event = MagicMock()  # Fake TalkEvent

    def test_base_actor_cannot_shell(self):
        actor = BaseActor("base", self.mock_runtime)
        self.assertFalse(actor.can_shell())
        with self.assertRaises(PermissionDenied):
            actor._dispatch_action("shell", {"command": "ls"}, self.mock_event)

    def test_file_actor_cannot_shell(self):
        actor = FileActor("file.py", self.mock_runtime)
        self.assertFalse(actor.can_shell())
        with self.assertRaises(PermissionDenied):
            actor._dispatch_action("shell", {"command": "ls"}, self.mock_event)

    def test_main_actor_can_shell(self):
        actor = MainActor("__main__", self.mock_runtime)
        self.assertTrue(actor.can_shell())
        result = actor._dispatch_action("shell", {"command": "echo hi"}, self.mock_event)
        self.assertEqual(result, "ok")
        self.mock_runtime.shell.assert_called_once_with("echo hi")


if __name__ == "__main__":
    unittest.main()
