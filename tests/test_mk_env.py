import unittest
from unittest.mock import patch, MagicMock
from mk_env import MkEnvWrapper

class TestMkEnvWrapper(unittest.TestCase):

    def setUp(self):
        # Patch RetroEnv's __init__ method
        patcher1 = patch('retro.RetroEnv.__init__', return_value=None)
        self.addCleanup(patcher1.stop)
        patcher1.start()

        # Initialize MkEnvWrapper
        self.env = MkEnvWrapper(game='MortalKombatII-Genesis')

        # Patch action generator and action space
        self.env.action_generator = patch('action_generator.ActionGenerator').start()
        self.addCleanup(patch.stopall)
        self.env.action_generator.generate_action_mapping.return_value = [0, 1, 2]
        self.env._action_mapping = [0, 1, 2]
        self.env.action_space = patch('gymnasium.spaces.Discrete').start()

    def test_initialization(self):
        self.assertEqual(self.env.max_health, 120)
        self.assertEqual(self.env.win_reward, 10.0)
        self.assertEqual(self.env.lose_penalty, -10)
        self.assertIsNone(self.env.player_health_prev)
        self.assertIsNone(self.env.enemy_health_prev)

    def test_reset(self):
        with patch('retro.RetroEnv.reset', return_value=(None, {'health': 100, 'enemy_health': 100})):
            obs, info = self.env.reset()
            self.assertEqual(self.env.player_health_prev, 100)
            self.assertEqual(self.env.enemy_health_prev, 100)

    def test_step(self):
        with patch('retro.RetroEnv.step', return_value=(None, 0, False, False, {'health': 90, 'enemy_health': 80, 'rounds_won': 1, 'enemy_rounds_won': 0})):
            obs, reward, done, truncated, info = self.env.step(0)
            self.assertEqual(self.env.player_health_prev, 90)
            self.assertEqual(self.env.enemy_health_prev, 80)
            self.assertEqual(reward, 10.0)  # win_reward

    def test_compute_reward(self):
        reward = self.env._compute_reward(90, 80, 1, 0)
        self.assertEqual(reward, 10.0)  # win_reward

    def test_step_with_damage(self):
        with patch('retro.RetroEnv.step', return_value=(None, 0, False, False, {'health': 80, 'enemy_health': 70, 'rounds_won': 0, 'enemy_rounds_won': 0})):
            obs, reward, done, truncated, info = self.env.step(0)
            self.assertEqual(self.env.player_health_prev, 80)
            self.assertEqual(self.env.enemy_health_prev, 70)
            self.assertEqual(reward, 0.08333333333333333)  # custom reward calculation

    def test_step_with_round_loss(self):
        with patch('retro.RetroEnv.step', return_value=(None, 0, False, False, {'health': 80, 'enemy_health': 70, 'rounds_won': 0, 'enemy_rounds_won': 1})):
            obs, reward, done, truncated, info = self.env.step(0)
            self.assertEqual(self.env.player_health_prev, 80)
            self.assertEqual(self.env.enemy_health_prev, 70)
            self.assertEqual(reward, -9.916666666666666)  # lose_penalty

if __name__ == '__main__':
    unittest.main()