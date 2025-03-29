import os
import numpy as np
from typing import Callable, List

from mk_ai.wrappers import MkEnvWrapper
from mk_ai.utils import DeterministicFrameSkip, Schedules
from mk_ai.agents import DuelingDoubleDQN
from mk_ai.callbacks import CustomEvalCallback

from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList

# Directories and fine tuning timesteps
MODEL_DIR: str = "models"
LOG_DIR: str = "./experiminets_finals/fine_tuned"
FINE_TUNE_TIMESTEPS: int = 2_000_000
NUM_ENVS: int = 8

def make_env(states: List[str], render_mode: str = "none") -> Callable[[], MkEnvWrapper]:
    """
    Factory function to create an environment using a list of game states.
    
    Parameters:
        states (List[str]): List of game states to use in the environment.
        render_mode (str): Rendering mode (set to "none" for training).
    
    Returns:
        A callable that creates a wrapped environment.
    """
    def _init_env() -> MkEnvWrapper:
        env = MkEnvWrapper(
            game="MortalKombatII-Genesis",
            states=states,
            render_mode=render_mode,
            record="datad"
        )

        env = DeterministicFrameSkip(env, n=4)
        env = Monitor(env)
        return env
    return _init_env

def fine_tune():
    """
    Fine tunes a pre-trained model using transfer learning on challenging states
    from tier 2 and tier 3, excluding the tier 1 states.
    
    This function:
      1. Creates a vectorized fine tuning environment using SubprocVecEnv.
      2. Loads a pre-trained model from curriculum training.
      3. (Optionally) freezes early layers of the policy.
      4. Adjusts the learning rate for fine tuning.
      5. Sets up an evaluation callback.
      6. Continues training for a defined number of timesteps.
      7. Saves the fine tuned model.
    """
    challenging_states: List[str] = [
        "VeryEasy.LiuKang-04",
        "VeryEasy.LiuKang-05",
        "VeryEasy.LiuKang-06",
        "VeryEasy.LiuKang-07",
        "VeryEasy.LiuKang-08"
    ]

    # Create the fine tuning training environment using SubprocVecEnv
    env = SubprocVecEnv([make_env(challenging_states, render_mode="none") for _ in range(NUM_ENVS)])
    env = VecFrameStack(env, n_stack=4)

    # Load the pre-trained model (from curriculum training for 16M timesteps)
    pretrained_model_path = os.path.join(MODEL_DIR, "kane", "DuellingDDQN_curriculum_16M_VeryEasy_3_Tiers")
    print(f"Loading pre-trained model from {pretrained_model_path}...")
    model = DuelingDoubleDQN.load(pretrained_model_path, env=env, device="cuda")

    # Optionally freeze early layers (e.g., feature extractor) if desired.
    # for param in model.policy.features_extractor.parameters():
    #     param.requires_grad = False

    # Adjust learning rate for fine tuning with a more gradual decay.
    fine_tune_lr = Schedules.linear_decay(3.16e-4, 1e-5)
    model.learning_rate = fine_tune_lr

    # Create an evaluation environment using DummyVecEnv for consistency in evaluation
    eval_env = DummyVecEnv([make_env(challenging_states, render_mode="none")])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Set up an evaluation callback to monitor fine tuning progress
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(LOG_DIR, "fine_tuned_best_transfer"),
        log_path=os.path.join(LOG_DIR, "fine_tune_eval_transfer"),
        eval_freq=62_500,  # Evaluate every total 500K (8 sub processes)steps
        n_eval_episodes=10,
        deterministic=False,
        render=False,
        verbose=1
    )

    # Fine tune the model for a defined number of timesteps
    print("Starting fine tuning...")
    model.learn(
        total_timesteps=FINE_TUNE_TIMESTEPS,
        reset_num_timesteps=False,
        callback=eval_callback,
    )

    # Save the fine-tuned model
    fine_tuned_model_path = os.path.join(MODEL_DIR, "kane", "DuellingDDQN_fine_tuned_transfer")
    model.save(fine_tuned_model_path)
    print(f"Fine tuned model saved to {fine_tuned_model_path}")

    # Close the environments
    env.close()
    eval_env.close()

if __name__ == "__main__":
    fine_tune()
