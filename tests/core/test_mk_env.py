import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from src.mk_ai.wrappers.mk_env import MkEnvWrapper

class TestMkEnvWrapper(unittest.TestCase):

    def setUp(self):
        # Patch RetroEnv's __init__ method to avoid its original initialization.
        patcher_init = patch('retro.RetroEnv.__init__', return_value=None)
        self.mock_retro_init = patcher_init.start()
        self.addCleanup(patcher_init.stop)

        # Create the environment instance (bypassing RetroEnv init).
        self.env = MkEnvWrapper(game='MortalKombatII-Genesis', state="dummy.state")

        # Patch load_state so it doesn't actually try to load a state.
        patcher_load_state = patch.object(self.env, 'load_state', return_value=None)
        self.mock_load_state = patcher_load_state.start()
        self.addCleanup(patcher_load_state.stop)

        # Patch the ActionGenerator inside the environment.
        # We'll mock out build() and set a fake binary_mapping.
        self.env.action_generator = MagicMock()
        self.env.action_generator.build.return_value = None
        self.env.action_generator.binary_mapping = [0, 1, 2]  # Our fake discrete action mapping
        # Re-assign environment attributes accordingly
        self.env._action_mapping = [0, 1, 2]

        # Mock out the action space too.
        patcher_discrete = patch('gymnasium.spaces.Discrete')
        self.mock_discrete = patcher_discrete.start()
        self.addCleanup(patcher_discrete.stop)

        # Initialize health tracking with default numbers
        self.env.player_health_prev = 100
        self.env.enemy_health_prev = 100

    def get_dummy_obs(self):
        """
        Create a dummy observation with a shape that _preprocess_frame expects,
        e.g. a 240x320 RGB image.
        """
        return np.zeros((240, 320, 3), dtype=np.uint8)

    def test_initialization(self):
        # Check if environment is initialized with correct default values
        self.assertEqual(self.env.max_health, 120)
        self.assertEqual(self.env.player_health_prev, 100)
        self.assertEqual(self.env.enemy_health_prev, 100)

    def test_reset(self):
        """
        Test that reset sets up health correctly and
        processes the returned observation into shape (84, 84, 1).
        """
        dummy_obs = self.get_dummy_obs()
        with patch('retro.RetroEnv.reset', return_value=(dummy_obs, {'health': 100, 'enemy_health': 100})):
            obs, info = self.env.reset()
            self.assertEqual(self.env.player_health_prev, 100)
            self.assertEqual(self.env.enemy_health_prev, 100)
            self.assertEqual(obs.shape, (84, 84, 1))

    def test_step(self):
        """
        Test step with a typical transition: 
        player's health goes 100 -> 90, enemy's 100 -> 80, round win for player.
        We expect the custom reward to reflect damage and round bonus.
        """
        dummy_obs = self.get_dummy_obs()
        # Mock RetroEnv.step to return dummy obs, 0 reward, done=False, truncated=False,
        # and updated health info (90 / 80).
        with patch('retro.RetroEnv.step', 
                   return_value=(dummy_obs, 0, False, False,
                                 {'health': 90, 'enemy_health': 80, 'rounds_won': 1, 'enemy_rounds_won': 0})):
            obs, reward, done, truncated, info = self.env.step(0)  # 0 is the discrete action index
            self.assertEqual(self.env.player_health_prev, 90)
            self.assertEqual(self.env.enemy_health_prev, 80)
            self.assertFalse(done)
            self.assertFalse(truncated)
            # Based on _compute_reward formula:
            #   Damage inflicted = 100 -> 80 => +20
            #   Damage taken = 100 -> 90 => -10 => net +10
            #   Round won => + (120 ^ ((90+1)/(120+1))) ~ +36.6169
            #   Total ~ 46.6169
            self.assertAlmostEqual(reward, 46.61691598268281, places=4)

    def test_compute_reward(self):
        """
        Specifically test _compute_reward logic from 100->90 (player) and 100->80 (enemy),
        plus 1 round won.
        """
        self.env.player_health_prev = 100
        self.env.enemy_health_prev = 100
        reward = self.env._compute_reward(90, 80, 1, 0)
        self.assertAlmostEqual(reward, 46.61691598268281, places=4)

    def test_step_with_damage(self):
        """
        Player 100->80, enemy 100->70, no rounds won/lost. Expect net +10 reward.
        """
        dummy_obs = self.get_dummy_obs()
        with patch('retro.RetroEnv.step',
                   return_value=(dummy_obs, 0, False, False,
                                 {'health': 80, 'enemy_health': 70, 'rounds_won': 0, 'enemy_rounds_won': 0})):
            obs, reward, done, truncated, info = self.env.step(1)  # action index 1
            self.assertEqual(self.env.player_health_prev, 80)
            self.assertEqual(self.env.enemy_health_prev, 70)
            # +30 for dealing damage (100->70), -20 for taking damage (100->80) => +10 total
            self.assertAlmostEqual(reward, 10.0, places=4)
            self.assertFalse(done)

    def test_step_with_round_loss(self):
        """
        Player 100->80, enemy 100->70, enemy round_won=1 => additional penalty.
        """
        dummy_obs = self.get_dummy_obs()
        with patch('retro.RetroEnv.step',
                   return_value=(dummy_obs, 0, False, False,
                                 {'health': 80, 'enemy_health': 70, 'rounds_won': 0, 'enemy_rounds_won': 1})):
            obs, reward, done, truncated, info = self.env.step(2)  # action index 2
            self.assertEqual(self.env.player_health_prev, 80)
            self.assertEqual(self.env.enemy_health_prev, 70)
            # Normal damage-based reward: +30 - 20 => +10
            # Round lost penalty => - (120 ^ ((70+1)/(120+1))) ~ -16.59646
            # Net = +10 - 16.59646 => ~ -6.59646
            self.assertAlmostEqual(reward, -6.59646, places=4)
            self.assertFalse(done)

if __name__ == '__main__':
    unittest.main()
