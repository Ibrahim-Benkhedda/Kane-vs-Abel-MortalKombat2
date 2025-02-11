import gymnasium as gym
import numpy as np
import torch as th 
import torch.optim as optim
import os 
import sys 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # updates the path to include the parent directory

from dueling_dqn import DuelingDQN
from double_dqn import DoubleDQN
from dueling_ddqn import DuelingDoubleDQN
from frameskip import DeterministicFrameSkip

from stable_baselines3 import DQN 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.torch_layers import NatureCNN

from gymnasium.wrappers import (
    AddRenderObservation,
    ResizeObservation,
    GrayscaleObservation,
)

GAMMA = 0.99
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.05
REPLAY_MEMORY = 200000
BATCH = 128
NUM_ENVS = 8



def make_cartpole_pixel_env(num_stack=4, image_size=(84, 84)):
    """
    Create a CartPole-v1 environment that returns image observations
    so we can feed them into a CNN. Steps:

    1) `PixelObservationWrapper` => adds "pixels" key to the obs dict
    2) `pixels_only=True` => the observation is only the pixels
    3) `ResizeObservation` => reshape to (84, 84) or any desired size
    4) `GrayScaleObservation` => (optional) converts to grayscale
    5) `FrameStack` => (optional) stack 4 frames (channel dim is increased)
    6) `Monitor` => record stats like episode length, reward, etc.
    
    Returns:
        Gymnasium env that produces observations of shape [num_stack, 84, 84] in grayscale.
    """
    def _init_env():    
        env = gym.make("CartPole-v1", render_mode="rgb_array")
        # 1) Convert environment so that it returns pixel observations in 'pixels' key
        env = AddRenderObservation(env, render_only=True)
        # 2) Resize to something typical (84x84)
        env = ResizeObservation(env, image_size)
        # 3) Convert to grayscale (keeps a singleton channel dim: shape (1, 84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        # 4) Monitor to log data
        env = Monitor(env)
        return env
    return _init_env


def test_dueling_cartpole():
    # 1) Create training environment with SubprocVecEnv and VecFrameStack
    env_fns = [make_cartpole_pixel_env() for _ in range(NUM_ENVS)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecFrameStack(train_env, n_stack=4)

    # 2) Create evaluation environment with DummyVecEnv and VecFrameStack
    eval_env = DummyVecEnv([make_cartpole_pixel_env()])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # 3) Create EvalCallback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs_dueling_cartpole/best_model",
        log_path="./logs_dueling_cartpole/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # 4) Create Dueling DQN model
    model = DuelingDQN(
        env=train_env,
        learning_rate=3e-4,
        buffer_size=REPLAY_MEMORY,
        batch_size=BATCH,
        exploration_fraction=0.1,
        exploration_initial_eps=INITIAL_EPSILON,
        exploration_final_eps=FINAL_EPSILON,
        policy_kwargs={"features_extractor_class": NatureCNN},
        verbose=1,
        tensorboard_log="./logs_dueling_cartpole/",
        device="cuda"  # or "cpu" if no GPU
    )

    # 5) Train the model
    model.learn(total_timesteps=200_000, callback=eval_callback)

    # 6) Save the final model
    model.save("dueling_cartpole_pixel.zip")
    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    test_dueling_cartpole()
