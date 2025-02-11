import numpy as np

from typing import Callable, List, Dict, Any
from retro import RetroEnv

from mk_ai.wrappers import MkEnvWrapper
from mk_ai.agents import DuelingDoubleDQN, DoubleDQN, DuelingDQN
from mk_ai.utils import DeterministicFrameSkip, StochasticFrameSkip, Schedules
from mk_ai.callbacks import CurriculumCallback, CustomEvalCallback

from stable_baselines3 import DQN, PPO 
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

MODEL_DIR: str = "models"
LOG_DIR: str = "logs"
TOTAL_TIMESTEPS: str = 20_000_000


# ========================================================================================
# Environment Creation
# ========================================================================================
def make_env(states: list[str]) -> Callable[[], MkEnvWrapper]:
    """
    Factory function to create an environment using a list of possible states.
    """
    # 1. create the environment using lambda to pass it to DummyVecEnv/SubprocVecEnv
    def _init_env() -> MkEnvWrapper:
        env = MkEnvWrapper(
            game="MortalKombatII-Genesis",
            states=states,
            render_mode="none",
            record="replays"
        )

        env = DeterministicFrameSkip(env, n=4)
        env = Monitor(env)
        return env 
    return _init_env

# ======================
# Training
# ======================
def train() -> None:
    """
    Train a DQN model on the Mortal Kombat II retro environment.

    This function:
      1. Defines tiered game state lists for curriculum training if used.
      2. Creates a vectorized training environment using SubprocVecEnv.
      3. Stacks frames for the environment using VecFrameStack.
      4. Initializes a DQN model with an scheduler learning rate.
      5. Sets up evaluation (and optionally curriculum) callbacks.
      6. Trains the model for a specified total number of timesteps.
      7. Saves the trained model and closes the environment.
    """

    # Define the states for each tier.
    tier1_states: List[str] = ["Level1.LiuKangVsJax"]
    tier2_states: List[str] = [
        "Level1.LiuKangVsJax", "VeryEasy.LiuKang-02", "VeryEasy.LiuKang-03",
        "VeryEasy.LiuKang-04", "VeryEasy.LiuKang-05" , 
    ]
    tier3_states: List[str] = [
        "Level1.LiuKangVsJax", "VeryEasy.LiuKang-02", "VeryEasy.LiuKang-03",
        "VeryEasy.LiuKang-04", "VeryEasy.LiuKang-05", "VeryEasy.LiuKang-06",
        "VeryEasy.LiuKang-07", "VeryEasy.LiuKang-08"
    ]
    tier4_states: List[str] = ["LiuKangVsBaraka_VeryHard_01", "LiuKangVsReptile_VeryHard_02", "LiuKangVsJax_VeryHard_03"]

    tiered_states: List[List[str]] = [tier1_states, tier2_states, tier3_states, tier4_states]


    # Create the main training environment using SubprocVecEnv with num_envs of processes.
    num_envs: int = 8
    venv = SubprocVecEnv([make_env(tier1_states) for _ in range(num_envs)])

    # Stack N frames together for the vectorized environment
    stacked_env = VecFrameStack(venv, 4)

    # Initialize learning rate schedules.
    lr_schedule = Schedules.linear_decay(3.16e-4, 1e-5)
    exp_decay_lr = Schedules.exponential_decay(3.16e-4, 0.295)

    # Initialize a DQN model
    model = DuelingDoubleDQN(
        # policy="CnnPolicy",
        env=stacked_env,
        verbose=1,
        device="cuda",
        buffer_size=200000,
        batch_size=32,
        gamma=0.948594691717984,
        learning_rate=exp_decay_lr,
        exploration_fraction=0.2992653791117977,
        exploration_initial_eps=0.9251553723683702,
        exploration_final_eps=0.07284636692467307,
        # train_freq=4,  
        # gradient_steps=2, 
        # target_update_interval=5000,
        tensorboard_log="./experiments/DuelingDDQN/"
    )

    # Optionally, create a curriculum callback (currently commented out)
    # curriculum_callback = CurriculumCallback(
    #     vec_env=venv,
    #     tiered_states=tiered_states,
    #     verbose=1,
    #     buffer_size=25
    # )

    # Define the desired evaluation interval in environment steps.
    desired_eval_interval: int = 500_000  # evaluation every defined number ENVIRONMENT steps
    eval_freq: int = max(desired_eval_interval // num_envs, 1)  # 500_000 // 8 = 62,500

    # Create the evaluation environment using DummyVecEnv.
    eval_env = DummyVecEnv([make_env(tier1_states)])
    eval_env = VecFrameStack(eval_env, 4)

    # Create the custom evaluation callback.
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        best_model_save_path="./experiments/DuelingDDQN/best_model-2",
        log_path="./experiments/DuelingDDQN/eval_logs-2",
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=False,  
        render=False,
        verbose=1
    )

    # Create a callback list. (Here, only the evaluation callback is used right now)
    callback_list = CallbackList([eval_callback])

    # Train the model for the total timesteps.
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=True,
        callback=callback_list
    )

    # Save the final trained model
    model.save("DueilingDoubleDQN_without_curriculum_4M_VeryHardVsJax_Exp_2")

    # Close the vectorized environment.
    venv.close()
    print(f"[MAIN] Training complete.")


# ======================
# CLI Entry Point
# ======================
if __name__ == "__main__":
    train()

