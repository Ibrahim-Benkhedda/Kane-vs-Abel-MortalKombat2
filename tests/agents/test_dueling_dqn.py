import unittest
import torch as th
import numpy as np
import os 
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # updates the path to include the parent directory

from dueling_dqn import DuelingCnnPolicy, DuelingQNetwork, DuelingDQN
from stable_baselines3.common.torch_layers import NatureCNN
from gymnasium.spaces import Box, Discrete

class TestDuelingArchitecture(unittest.TestCase):
    def test_network_arch(self):
        # Mock parameters
        obs_space = Box(low=0, high=255, shape=(1, 84, 84), dtype=np.uint8)
        act_space = Discrete(n=6)
        features_extractor = NatureCNN(obs_space, features_dim=512)
        net_arch = [256, 256]

        model = DuelingQNetwork(
            observation_space=obs_space,
            action_space=act_space,
            features_extractor=features_extractor,
            features_dim=512,
            net_arch=net_arch,
            activation_fn=th.nn.ReLU,
        )

        # Check the internal submodules
        self.assertIsNotNone(model.advantage_net)
        self.assertIsNotNone(model.value_net)

        # Make sure advantage_net outputs #actions, value_net outputs 1
        dummy_features = th.randn(2, 512)  # pretend CNN output for batch_size=2
        advantage_out = model.advantage_net(dummy_features)
        value_out = model.value_net(dummy_features)

        # Suppose the action_space.n = 4
        self.assertEqual(advantage_out.shape, (2, act_space.n))
        self.assertEqual(value_out.shape, (2, 1))

        # Next, test the final forward pass (which merges them)
        # We must provide a mock "obs" that the features_extractor can handle
        # example would be if it's an Atari shape of (4, 84, 84):
        dummy_obs = th.randn(2, 1, 84, 84)
        q_values = model(dummy_obs)  # [batch_size, action_dim]

        self.assertEqual(q_values.shape, (2, act_space.n))

if __name__ == "__main__":
    unittest.main()