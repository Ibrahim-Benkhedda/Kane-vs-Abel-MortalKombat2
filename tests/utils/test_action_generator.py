import unittest
import os
import sys

from src.mk_ai.utils import ActionGenerator

class TestActionGenerator(unittest.TestCase):

    def setUp(self):
        self.buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self.action_generator = ActionGenerator(buttons=self.buttons)

    def test_init(self):
        self.assertEqual(self.action_generator.buttons, self.buttons)
        self.assertEqual(self.action_generator.actions, [])

    def test_add_single_action(self):
        self.action_generator.add_action(["B", "DOWN"])
        self.assertEqual(self.action_generator.actions, [["B", "DOWN"]])

    def test_add_multiple_actions(self):
        self.action_generator.add_actions([["B", "DOWN"], ["UP", "RIGHT"]])
        self.assertEqual(self.action_generator.actions, [["B", "DOWN"], ["UP", "RIGHT"]])

    def test_invalid_single_action(self):
        with self.assertRaises(ValueError):
            self.action_generator.add_action("B, DOWN")

    def test_invalid_multiple_actions(self):
        with self.assertRaises(ValueError):
            self.action_generator.add_actions([["B", "DOWN"], "UP, RIGHT"])

    def test_generate_action_mapping(self):
        # Equivalent to the old test logic for generate_action_mapping().
        self.action_generator.add_actions([["B", "DOWN"]])
        self.action_generator.add_actions([["UP", "RIGHT"]])
        
        # Now we call build() to produce the binary mapping internally.
        self.action_generator.build()
        action_mapping = self.action_generator.binary_mapping  # replaced old function return

        self.assertEqual(len(action_mapping), 2)
        self.assertEqual(action_mapping[0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(action_mapping[1], [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

    def test_generate_action_mapping_with_lookup(self):
        # Equivalent to the old test logic for generate_action_mapping_with_lookup().
        self.action_generator.add_actions([["B", "DOWN"], ["UP", "RIGHT"]])
        
        # build() also populates combo_to_id
        self.action_generator.build()
        action_mapping = self.action_generator.binary_mapping
        
        # Check if the action mapping is correct
        self.assertEqual(len(action_mapping), 2)
        self.assertEqual(action_mapping[0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(action_mapping[1], [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
        
        # Check if the combo_to_id mapping is correct (using property instead of _combo_to_id)
        self.assertEqual(self.action_generator.combo_to_id[("B", "DOWN")], 0)
        self.assertEqual(self.action_generator.combo_to_id[("UP", "RIGHT")], 1)

    def test_get_action_id(self):
        self.action_generator.add_actions([["B", "DOWN"], ["UP", "RIGHT"]])
        
        # build() to populate combo_to_id
        self.action_generator.build()
        
        # Check if the correct ID is returned for a given combo
        self.assertEqual(self.action_generator.get_action_id(["B", "DOWN"]), 0)
        self.assertEqual(self.action_generator.get_action_id(["UP", "RIGHT"]), 1)
        
        # Check if KeyError is raised for a non-existent combo
        with self.assertRaises(KeyError):
            self.action_generator.get_action_id(["A", "LEFT"])


if __name__ == '__main__':
    unittest.main()
