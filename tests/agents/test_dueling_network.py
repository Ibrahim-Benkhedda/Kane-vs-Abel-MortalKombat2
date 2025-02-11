import unittest
import torch as th
import numpy as np
import os 
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # updates the path to include the parent directory

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from dueling_dqn import DuelingQNetwork

class DummyFeatureExtractor(BaseFeaturesExtractor):
    """
    Dummy CNN feature extractor for testing.
    Converts input tensor to a flattened tensor with a fixed dimension.
    """
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        self.conv = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(np.prod(observation_space.shape), features_dim),
            th.nn.ReLU(),
        )

    def forward(self, observations):
        return self.conv(observations)


class TestDuelingQNetwork(unittest.TestCase):
    def setUp(self):
        # Define mock observation and action spaces
        self.observation_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)  # Two discrete actions
        self.features_dim = 512
        self.net_arch = [256, 256]
        self.activation_fn = th.nn.ReLU

        # Create DuelingQNetwork with a dummy feature extractor
        self.model = DuelingQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=DummyFeatureExtractor(self.observation_space, self.features_dim),
            features_dim=self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )

        print(self.model)

    def test_forward_shapes(self):
        # Create a dummy observation batch
        batch_size = 128
        dummy_obs = th.rand(batch_size, *self.observation_space.shape)

        # Pass through the network
        q_values = self.model(dummy_obs)

        # Assert output shapes
        self.assertEqual(q_values.shape, (batch_size, self.action_space.n), "Q-values shape is incorrect")
        print("Forward pass shapes are correct.")

    def test_q_value_computation(self):
        # Create a dummy observation batch
        batch_size = 16
        dummy_obs = th.rand(batch_size, *self.observation_space.shape)

        # Forward pass
        features = self.model.extract_features(dummy_obs, self.model.features_extractor)
        advantages = self.model.advantage_net(features)
        values = self.model.value_net(features)

        # Manually compute Q-values
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        expected_q_values = values + (advantages - advantages_mean)
        actual_q_values = self.model(dummy_obs)

        # Assert correctness
        self.assertTrue(th.allclose(expected_q_values, actual_q_values, atol=1e-5), "Q-value computation is incorrect")
        print("Q-value computation is correct.")

    def test_predict(self):
        # Create a dummy observation batch
        batch_size = 32
        dummy_obs = th.rand(batch_size, *self.observation_space.shape)

        # Forward pass
        q_values = self.model(dummy_obs)

        # Predict actions
        predicted_actions = self.model._predict(dummy_obs, deterministic=True)

        # Assert the predicted actions are the argmax of Q-values
        expected_actions = q_values.argmax(dim=1)
        self.assertTrue(th.equal(predicted_actions, expected_actions), "_predict method is incorrect")
        print("Action prediction is correct.")

if __name__ == "__main__":
    unittest.main()
