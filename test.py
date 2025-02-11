import argparse
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from mk_ai.wrappers import MkEnvWrapper
from mk_ai.utils import DeterministicFrameSkip
from mk_ai.agents import DuelingDoubleDQN, DoubleDQN, DuelingDQN

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# -----------------------------------------------------------------------
# 1. Environment Setup
# -----------------------------------------------------------------------
def make_test_env(
    game: str = "MortalKombatII-Genesis",
    state: str = "Level1.LiuKangVsJax",
    states: list = None,
    render_mode: str = "human",
    num_stack: int = 4,
    num_skip: int = 4, 
) -> VecFrameStack:
    """
    Create a vectorized environment wrapped in FrameStack for testing.

    Parameters:
        game (str): Name of the game/ROM.
        state (str): Specific state to load (e.g., 'Level1.LiuKangVsJax').
        render_mode (str): Rendering mode, 'human' for on-screen display.
        num_stack (int): Number of frames to stack.

    Returns:
        VecFrameStack: A frame-stacked vectorized environment.
    """
    def _init_env():
        env = MkEnvWrapper(
            game=game,
            state=state,
            states=states,
            render_mode=render_mode,
        )

        env = DeterministicFrameSkip(env, n=num_skip)
        return env

    # DummyVecEnv accepts a list of environment callables
    vec_env = DummyVecEnv([_init_env])
    stacked_env = VecFrameStack(vec_env, n_stack=num_stack)
    return stacked_env


# -----------------------------------------------------------------------
# 2. Evaluation
# -----------------------------------------------------------------------
def evaluate_agent(model, env, num_episodes: int = 10) -> Tuple[float, float]:
    """
    Evaluate a given RL agent in the provided environment over multiple episodes.

    Parameters:
        model: A Stable-Baselines3 model (DQN, PPO, etc.).
        env: A vectorized environment to test in.
        num_episodes (int): Number of evaluation episodes.

    Returns:
        Tuple[float, float]: (average_reward, std_reward)
    """
    all_episode_rewards = []

    for episode_idx in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            # Model predicts best action
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)

            plt.imshow(obs[0])
            plt.show(block=False)
            
            episode_reward += reward[0]  # reward is an array of length num_envs

        all_episode_rewards.append(episode_reward)
        print(f"Episode {episode_idx + 1}/{num_episodes} reward: {episode_reward}")

    avg_reward = float(np.mean(all_episode_rewards))
    std_reward = float(np.std(all_episode_rewards))
    print(f"Evaluation over {num_episodes} episodes: "
          f"Average Reward = {avg_reward:.2f}, Std = {std_reward:.2f}")

    return avg_reward, std_reward


# -----------------------------------------------------------------------
# 3. Model Loading
# -----------------------------------------------------------------------
def load_model(model_path: str, model_type: str = "DQN"):
    """
    Load a pre-trained SB3 model (DQN, DDQN ...).

    Parameters:
        model_path (str): Path to the saved model (e.g. 'Kane.zip').
        model_type (str): The type of SB3 model: 'DQN' or 'DDQN' or 'PPO'.

    Returns:
        An instance of the loaded SB3 model.
    """
    if model_type.upper() == "DQN":
        return DQN.load(model_path, device="cuda")
    elif model_type.upper() == "DDQN":
        return DoubleDQN.load(model_path, device="cuda")
    elif model_type.upper() == "DUELINGDDQN":
        return DuelingDoubleDQN.load(model_path, device="cuda")
    elif model_type.upper() == "PPO":
        return PPO.load(model_path, device="cuda")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


# -----------------------------------------------------------------------
# Main - Testing/Evaluation Script Entry
# -----------------------------------------------------------------------
def main(args):
    """
    Main function to load and evaluate a pre-trained agent.

    Parameters:
        args: Command line arguments from argparse.
    """
    print(f"Loading the model from {args.model_path} (type: {args.model_type})...")
    model = load_model(args.model_path, args.model_type)

    print("Creating test environment...")
    env = make_test_env(
        game=args.game,
        state=args.state,
        states=args.states,
        render_mode=args.render_mode,
        num_stack=args.num_stack,
        num_skip=args.num_skip
    )

    print("Starting evaluation...")
    evaluate_agent(model, env, num_episodes=args.num_episodes)

    print("Evaluation complete. Closing environment.")
    env.close()


# -----------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained agent on Mortal Kombat II: Genesis.")
    parser.add_argument("--model_path", type=str, default="Kane.zip", help="Path to the saved model file.")
    parser.add_argument("--model_type", type=str, default="DQN", help="Type of the model: 'DQN' or 'PPO'.")
    parser.add_argument("--game", type=str, default="MortalKombatII-Genesis", help="Name of the ROM/game.")
    parser.add_argument("--state", type=str, default="Level1.LiuKangVsJax", help="Game state to load.")
    parser.add_argument("--states", type=list, default=None, help="List of game states to load.")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode: 'human' or 'rgb_array' or 'none'.")
    parser.add_argument("--num_stack", type=int, default=4, help="Number of frames to stack in the environment.")
    parser.add_argument("--num_skip",  type=int, default=4, help="Number of frames to skip in the environment.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run for evaluation.")

    args = parser.parse_args()
    main(args)
