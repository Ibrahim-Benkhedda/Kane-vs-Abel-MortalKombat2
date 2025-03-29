import argparse
import csv
import os
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
            record="replays/"
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
    Records each episode's reward and win flag (assumed win when rounds_won equals 2)
    and then saves these results to a CSV file.
    
    Parameters:
        model: A Stable-Baselines3 model (e.g., DQN, PPO).
        env: A vectorized environment.
        num_episodes (int): Number of episodes for evaluation.
    
    Returns:
        Tuple[float, float]: (average_reward, std_reward)
    """
    # -----------------------------------------------------------------------
# 2. Evaluation
# -----------------------------------------------------------------------
def evaluate_agent(model, env, num_episodes: int = 10) -> Tuple[float, float, list, list]:
    """
    Evaluate a given RL agent in the provided environment over multiple episodes.
    Records each episode's reward and win flag (assumed win when rounds_won equals 2).

    Parameters:
        model: A Stable-Baselines3 model (e.g., DQN, PPO).
        env: A vectorized environment.
        num_episodes (int): Number of episodes for evaluation.

    Returns:
        Tuple[float, float, list, list]:
            - average_reward: float
            - std_reward: float
            - list of episode rewards
            - list of win flags (True if agent won the episode)
    """
    all_episode_rewards = []
    episode_wins = []  # True if agent won the episode, False otherwise

    for episode_idx in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        rounds_won = 0

        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]  # vectorized env returns reward as a list
            rounds_won = info[0].get("rounds_won", rounds_won)

        agent_won = (rounds_won == 2)
        all_episode_rewards.append(episode_reward)
        episode_wins.append(agent_won)
        print(f"Episode {episode_idx + 1}/{num_episodes} reward: {episode_reward}, won: {agent_won}")

    avg_reward = float(np.mean(all_episode_rewards))
    std_reward = float(np.std(all_episode_rewards))
    print(f"Evaluation over {num_episodes} episodes: Average Reward = {avg_reward:.2f}, Std = {std_reward:.2f}")

    return avg_reward, std_reward, all_episode_rewards, episode_wins

# -----------------------------------------------------------------------
#  3. Evaluation Helper - Save Results
# -----------------------------------------------------------------------
def save_results_csv(
        episode_rewards: list,
        episode_wins: list,
        avg_reward: float,
        std_reward: float,
        base_folder: str = "notebooks/evaluation_results",
        filename: str = None,
        state: str = None
    ):
    """
    Save the per-episode rewards and win flags along with overall statistics to a CSV file.

    Parameters:
        episode_rewards (list): List of rewards for each episode.
        episode_wins (list): List of booleans indicating if the episode was won.
        avg_reward (float): The overall average reward.
        std_reward (float): The overall reward standard deviation.
        base_folder (str): Folder to save the CSV file.
        filename (str): Name of the CSV file; if not provided, one is generated.
        state (str): Optional state name used to generate a state-specific filename.
    """
    if filename is None:
        if state is not None:
            filename = f"evaluation_results_{state}.csv"
        else:
            filename = "evaluation_results.csv"
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    
    file_path = os.path.join(base_folder, filename)
    
    with open(file_path, mode="w", newline="") as csv_file:
        fieldnames = ["episode", "reward", "won"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for i, (reward, win) in enumerate(zip(episode_rewards, episode_wins), 1):
            writer.writerow({"episode": i, "reward": reward, "won": win})
        writer.writerow({"episode": "average", "reward": avg_reward, "won": ""})
        writer.writerow({"episode": "std", "reward": std_reward, "won": ""})
    
    print(f"Results saved to {file_path}")


# -----------------------------------------------------------------------
# 4. Model Loading
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
    Depending on the --individual_eval flag, either evaluate each provided state individually
    (saving separate CSV files) or use a fixed state or list (for random sampling) via MkEnvWrapper.

    Parameters:
        args: Command line arguments.
    """
    print(f"Loading the model from {args.model_path} (type: {args.model_type})...")
    model = load_model(args.model_path, args.model_type)

    if args.individual_eval and args.states:
        for state in args.states:
            print(f"\nEvaluating state individually: {state}")
            env = make_test_env(
                game=args.game,
                state=state,    # use the individual state for fixed evaluation
                states=None,    # do not pass the list; fixed state evaluation
                render_mode=args.render_mode,
                num_stack=args.num_stack,
                num_skip=args.num_skip
            )
            avg_reward, std_reward, episode_rewards, episode_wins = evaluate_agent(model, env, num_episodes=args.num_episodes)
            save_results_csv(episode_rewards, episode_wins, avg_reward, std_reward, state=state)
            env.close()
    else:
        # Use a fixed state or a list for random sampling (if states provided)
        print("Creating test environment...")
        env = make_test_env(
            game=args.game,
            state=args.state,
            states=args.states,  # if provided and not individual_eval, MkEnvWrapper samples randomly
            render_mode=args.render_mode,
            num_stack=args.num_stack,
            num_skip=args.num_skip
        )
        print("Starting evaluation...")
        avg_reward, std_reward, episode_rewards, episode_wins = evaluate_agent(model, env, num_episodes=args.num_episodes)
        # Save results using the default state (or you may pass args.state)
        save_results_csv(episode_rewards, episode_wins, avg_reward, std_reward, state=args.state)
        env.close()

    print("Evaluation complete.")


# -----------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained agent on Mortal Kombat II: Genesis.")
    parser.add_argument("--model_path", type=str, default="Kane.zip", help="Path to the saved model file.")
    parser.add_argument("--model_type", type=str, default="DQN", help="Type of the model: 'DQN', 'DDQN', 'DUELINGDDQN', or 'PPO'.")
    parser.add_argument("--game", type=str, default="MortalKombatII-Genesis", help="Name of the ROM/game.")
    parser.add_argument("--state", type=str, default="Level1.LiuKangVsJax", help="Game state to load if not using --states.")
    parser.add_argument("--states", type=lambda s: s.split(','), default=None,
                        help="Comma-separated list of game states to load. Use with --individual_eval for individual evaluation.")
    parser.add_argument("--individual_eval", action="store_true", help="If set, evaluate each provided state individually and save separate results.")
    parser.add_argument("--render_mode", type=str, default="human", help="Render mode: 'human', 'rgb_array', or 'none'.")
    parser.add_argument("--num_stack", type=int, default=4, help="Number of frames to stack in the environment.")
    parser.add_argument("--num_skip",  type=int, default=4, help="Number of frames to skip in the environment.")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run for evaluation.")

    args = parser.parse_args()
    main(args)
