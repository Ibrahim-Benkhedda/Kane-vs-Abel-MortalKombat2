import unittest
import os 
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # updates the path to include the parent directory

from action_generator import ActionGenerator

class TestActionGenerator(unittest.TestCase):

    def setUp(self):
        self.buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        self.action_generator = ActionGenerator(self.buttons)

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
        self.action_generator.add_actions([["B", "DOWN"]])
        self.action_generator.add_actions([["UP", "RIGHT"]])
        action_mapping = self.action_generator.generate_action_mapping()
        self.assertEqual(len(action_mapping), 2)
        self.assertEqual(action_mapping[0], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(action_mapping[1], [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])

if __name__ == '__main__':
    unittest.main()