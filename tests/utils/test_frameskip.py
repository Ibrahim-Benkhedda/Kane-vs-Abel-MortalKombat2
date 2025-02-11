import unittest
import numpy as np
import os 
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # updates the path to include the parent directory

from unittest.mock import MagicMock
from src.utils import DeterministicFrameSkip

class TestFrameSkip(unittest.TestCase):
    def setUp(self):
        # Create a mock environment
        self.mock_env = MagicMock()
        self.n = 4
        self.wrapper = DeterministicFrameSkip(self.mock_env, n=self.n)

    def test_reset_with_stack(self):
        """Test if reset calls the underlying env's reset with frame stack"""
        expected_obs = np.zeros((84, 84, 4))  # Assuming frame stack of 4
        self.mock_env.reset.return_value = expected_obs

        obs = self.wrapper.reset()
        
        self.mock_env.reset.assert_called_once()
        np.testing.assert_array_equal(obs, expected_obs)

    def test_step_normal_with_stack(self):
        """Test if step properly accumulates rewards over skip frames with frame stack"""
        # Mock returns for each step
        mock_returns = [
            (np.zeros((84, 84, 4)), 1.0, False, False, {}),
            (np.zeros((84, 84, 4)), 2.0, False, False, {}),
            (np.zeros((84, 84, 4)), 3.0, False, False, {}),
            (np.ones((84, 84, 4)), 4.0, False, False, {})
        ]
        self.mock_env.step.side_effect = mock_returns

        obs, reward, done, truncated, info = self.wrapper.step(1)  # action = 1

        # Check if step was called correct number of times
        self.assertEqual(self.mock_env.step.call_count, self.n)
        # Check if rewards were accumulated correctly
        self.assertEqual(reward, 10.0)  # 1 + 2 + 3 + 4 = 10
        # Check if final observation is returned
        np.testing.assert_array_equal(obs, np.ones((84, 84, 4)))

    def test_step_early_done_with_stack(self):
        """Test if step breaks early when done is True with frame stack"""
        # Mock returns where done=True on second step
        mock_returns = [
            (np.zeros((84, 84, 4)), 1.0, False, False, {}),
            (np.ones((84, 84, 4)), 2.0, True, False, {}),
            (np.zeros((84, 84, 4)), 3.0, False, False, {}),
            (np.zeros((84, 84, 4)), 4.0, False, False, {})
        ]
        self.mock_env.step.side_effect = mock_returns

        obs, reward, done, truncated, info = self.wrapper.step(1)

        # Check if step was called only twice (should break after done=True)
        self.assertEqual(self.mock_env.step.call_count, 2)
        # Check if rewards were accumulated correctly until done
        self.assertEqual(reward, 3.0)  # 1 + 2 = 3
        # Check if done is True
        self.assertTrue(done)
        # Check if final observation before done is returned
        np.testing.assert_array_equal(obs, np.ones((84, 84, 4)))

    def test_step_with_truncated_with_stack(self):
        """Test if step breaks early when truncated is True with frame stack."""
        # Mock returns where truncated=True on second step
        mock_returns = [
            (np.zeros((84, 84, 4)), 1.0, False, False, {}),
            (np.ones((84, 84, 4)), 2.0, False, True, {}),
            (np.zeros((84, 84, 4)), 3.0, False, False, {}),
            (np.zeros((84, 84, 4)), 4.0, False, False, {})
        ]
        self.mock_env.step.side_effect = mock_returns

        obs, reward, done, truncated, info = self.wrapper.step(1)

        # Check if step was called only twice (should break after truncated=True)
        self.assertEqual(self.mock_env.step.call_count, 2)
        # Check if rewards were accumulated correctly until truncated
        self.assertEqual(reward, 3.0)  # 1 + 2 = 3
        # Check if truncated is True
        self.assertTrue(truncated)
        # Check if final observation before truncated is returned
        np.testing.assert_array_equal(obs, np.ones((84, 84, 4)))

    def test_reset_no_stack(self):
        """Test if reset calls the underlying env's reset without frame stack"""
        expected_obs = np.zeros((84, 84))  # Assuming no frame stack
        self.mock_env.reset.return_value = expected_obs

        obs = self.wrapper.reset()
        
        self.mock_env.reset.assert_called_once()
        np.testing.assert_array_equal(obs, expected_obs)

    def test_step_normal_no_stack(self):
        """Test if step properly accumulates rewards over skip frames without frame stack"""
        # Mock returns for each step
        mock_returns = [
            (np.zeros((84, 84)), 1.0, False, False, {}),
            (np.zeros((84, 84)), 2.0, False, False, {}),
            (np.zeros((84, 84)), 3.0, False, False, {}),
            (np.ones((84, 84)), 4.0, False, False, {})
        ]
        self.mock_env.step.side_effect = mock_returns

        obs, reward, done, truncated, info = self.wrapper.step(1)  # action = 1

        # Check if step was called correct number of times
        self.assertEqual(self.mock_env.step.call_count, self.n)
        # Check if rewards were accumulated correctly
        self.assertEqual(reward, 10.0)  # 1 + 2 + 3 + 4 = 10
        # Check if final observation is returned
        np.testing.assert_array_equal(obs, np.ones((84, 84)))

    def test_step_early_done_no_stack(self):
        """Test if step breaks early when done is True without frame stack"""
        # Mock returns where done=True on second step
        mock_returns = [
            (np.zeros((84, 84)), 1.0, False, False, {}),
            (np.ones((84, 84)), 2.0, True, False, {}),
            (np.zeros((84, 84)), 3.0, False, False, {}),
            (np.zeros((84, 84)), 4.0, False, False, {})
        ]
        self.mock_env.step.side_effect = mock_returns

        obs, reward, done, truncated, info = self.wrapper.step(1)

        # Check if step was called only twice (should break after done=True)
        self.assertEqual(self.mock_env.step.call_count, 2)
        # Check if rewards were accumulated correctly until done
        self.assertEqual(reward, 3.0)  # 1 + 2 = 3
        # Check if done is True
        self.assertTrue(done)
        # Check if final observation before done is returned
        np.testing.assert_array_equal(obs, np.ones((84, 84)))

    def test_step_with_truncated_no_stack(self):
        """Test if step breaks early when truncated is True without frame stack."""
        # Mock returns where truncated=True on second step
        mock_returns = [
            (np.zeros((84, 84)), 1.0, False, False, {}),
            (np.ones((84, 84)), 2.0, False, True, {}),
            (np.zeros((84, 84)), 3.0, False, False, {}),
            (np.zeros((84, 84)), 4.0, False, False, {})
        ]
        self.mock_env.step.side_effect = mock_returns

        obs, reward, done, truncated, info = self.wrapper.step(1)

        # Check if step was called only twice (should break after truncated=True)
        self.assertEqual(self.mock_env.step.call_count, 2)
        # Check if rewards were accumulated correctly until truncated
        self.assertEqual(reward, 3.0)  # 1 + 2 = 3
        # Check if truncated is True
        self.assertTrue(truncated)
        # Check if final observation before truncated is returned
        np.testing.assert_array_equal(obs, np.ones((84, 84)))

if __name__ == '__main__':
    unittest.main()