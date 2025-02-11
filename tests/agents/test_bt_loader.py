import os
import unittest

from mk_ai.agents.BT.nodes import Node, Selector, Sequence, Action, Condition
from mk_ai.agents.BT.loader import BTLoader 
from mk_ai.agents.BT.conditions import is_close_to_enemy, is_enemy_to_the_left, is_enemy_to_the_right

class TestBTLoader(unittest.TestCase):
    def setUp(self):
        self.condition_map = {
            "is_enemy_to_the_right": is_enemy_to_the_right,
            "is_enemy_to_the_left": is_enemy_to_the_left,
            "is_close_to_enemy": is_close_to_enemy
        }
        self.action_map = {
            "NEUTRAL_ID": 0,
            "MOVE_RIGHT_ID": 1,
            "MOVE_LEFT_ID": 2,
            "JUMP_ID": 3
        }
        self.loader = BTLoader(self.condition_map, self.action_map)
        # Define the fixtures directory path relative to this test file.
        self.fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")

    def _get_fixture_path(self, filename: str) -> str:
        """Helper method to construct the full path to a fixture file."""
        return os.path.join(self.fixtures_dir, filename)

    def test_valid_bt(self):
        """Test that a valid YAML specification produces the correct Behavior Tree."""
        file_path = self._get_fixture_path("valid_bt.yaml")
        bt_root = self.loader.gen_bt(file_path)

        self.assertIsInstance(bt_root, Selector)
        self.assertEqual(bt_root.name, "Approach Enemy or Jump")
        self.assertEqual(len(bt_root.children), 1)

        seq_node = bt_root.children[0]
        self.assertIsInstance(seq_node, Sequence)
        self.assertEqual(len(seq_node.children), 2)

        condition_node = seq_node.children[0]
        action_node = seq_node.children[1]
        
        self.assertIsInstance(condition_node, Condition)
        self.assertIsInstance(action_node, Action)
        self.assertEqual(action_node.frames_needed, 5)

    def test_invalid_node_type(self):
        """Test that an unknown node type in the YAML raises a ValueError."""
        file_path = self._get_fixture_path("invalid_node_type.yaml")
        with self.assertRaises(ValueError) as context:
            self.loader.gen_bt(file_path)
        self.assertIn("Unknown node type", str(context.exception))

    def test_invalid_frames_needed(self):
        """Test that an Action node with an invalid (negative) frames_needed value raises a ValueError."""
        file_path = self._get_fixture_path("invalid_frames_needed.yaml")
        with self.assertRaises(ValueError) as context:
            self.loader.gen_bt(file_path)
        self.assertIn("invalid 'frames_needed' value", str(context.exception))

    def test_missing_condition(self):
        """Test that a Condition node with a non-existent condition mapping raises a ValueError."""
        file_path = self._get_fixture_path("missing_condition.yaml")
        with self.assertRaises(ValueError) as context:
            self.loader.gen_bt(file_path)
        self.assertIn("not defined", str(context.exception))


if __name__ == '__main__':
    unittest.main()