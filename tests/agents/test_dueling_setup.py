import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Updates the path to include the parent directory

from dueling_dqn import DuelingDQN
import gymnasium as gym
from stable_baselines3.common.torch_layers import NatureCNN
from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
    GrayscaleObservation,
)

def make_cartpole_pixel_env():
    """
    Create a pixel-based CartPole environment with resized and grayscale observations.
    """
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    # Add rendered images as the primary observation
    env = AddRenderObservation(env, render_only=True)
    # Resize to 84x84
    env = ResizeObservation(env, (84, 84))
    # Convert to grayscale
    env = GrayscaleObservation(env, keep_dim=True)
    return env

# Create the environment
env = make_cartpole_pixel_env()

# Print the observation space to verify the setup
print("Observation space:", env.observation_space)

# Create the DuelingDQN model
model = DuelingDQN(
    env=env,
    learning_rate=3e-4,
    buffer_size=100_000,  # Reduced buffer size
    batch_size=64,
    learning_starts=1000,
    target_update_interval=1000,
    train_freq=4,
    gradient_steps=1,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.02,
    verbose=1,  # For detailed logs
    policy_kwargs={"features_extractor_class": NatureCNN}
)

# Perform a short training session to ensure everything works
model.learn(1000)

# Print the policy details
print("Available attributes/methods:", dir(model))
print("Policy attribute present?:", hasattr(model, "policy"))

if hasattr(model, "policy"):
    print("Policy object:", model.policy)
    if hasattr(model.policy, "q_net"):
        print("Features extractor:", model.policy.q_net.features_extractor)
