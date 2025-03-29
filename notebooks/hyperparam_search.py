import optuna
import os
import pandas as pd 

from mk_env import MkEnvWrapper
from utils import DeterministicFrameSkip, Schedules
from agents import DuelingDoubleDQN, DoubleDQN, DuelingDQN

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from gymnasium.wrappers import TimeLimit

LOG_DIR = "experiments/hpo/logs" 
OPT_DIR = "experiments/hpo/models"
TOTAL_TIMESTEPS = 500_000
N_EVAL_EPISODES = 10

# function to retrun tested hyperparameters
def optimize_hyperparameters(trial):
    return {
        "initial_lr": trial.suggest_loguniform("initial_lr", 1e-5, 1e-3),
        "decay_rate": trial.suggest_uniform("decay_rate", 0.1, 0.9),
        "buffer_size" : trial.suggest_int("buffer_size", 100000, 200000, step=100000),
        "batch_size" : trial.suggest_categorical("batch_size", [32, 64]),
        "exploration_fraction" : trial.suggest_uniform("exploration_fraction", 0.1, 0.5),
        "exploration_initial_eps" : trial.suggest_uniform("exploration_initial_eps", 0.8, 1.0),
        "exploration_final_eps" : trial.suggest_uniform("exploration_final_eps", 0.01, 0.1),
        "gamma": trial.suggest_loguniform("gamma", 0.9, 0.9999),
    }

# run a training loop and return the average reward 
def optimize_agent(trial):
    try: 
        # get the set of hyperparameters
        model_params = optimize_hyperparameters(trial)

        # Extract lr schedule params and remove from model_params so that we dont pass it to DQN params.
        initial_lr = model_params.pop("initial_lr")
        decay_rate = model_params.pop("decay_rate")
        lr_schedule = Schedules.exponential_decay(initial_lr, decay_rate)

        # create the env 
        def make_env():
            env = MkEnvWrapper(
                game="MortalKombatII-Genesis",
                states=["Level1.LiuKangVsJax", "VeryEasy.LiuKang-02", "VeryEasy.LiuKang-03", "VeryEasy.LiuKang-04"],
                render_mode="none",
                record="replays"
            )
            env = DeterministicFrameSkip(env, n=4)
            env = TimeLimit(env, max_episode_steps=4500)
            env = Monitor(env, LOG_DIR)
            return env
        num_envs = 8
        venv = SubprocVecEnv([make_env for _ in range(num_envs)])

        stacked_env = VecFrameStack(venv, 4)

        # setup the DQN model
        model = DuelingDoubleDQN(
            env=stacked_env,
            tensorboard_log=LOG_DIR,
            verbose=0,
            learning_rate=lr_schedule,
            **model_params
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        # Evaluation environment (DummyVecEnv)
        eval_env = DummyVecEnv([lambda: make_env()])
        eval_stacked_env = VecFrameStack(eval_env, 4)
        
        mean_reward, _ = evaluate_policy(model, eval_stacked_env, n_eval_episodes=N_EVAL_EPISODES)

        stacked_env.close()

        SAVE_PATH = os.path.join(OPT_DIR, f"DuelingDoubleDQN_{trial.number}")
        model.save(SAVE_PATH)

        return mean_reward
    
    except Exception as e:
        print(f"Error: {e}")
        return -9999.0
    

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=10, n_jobs=1)
    print("Number of finished trials: ", len(study.trials))

    # convert trials to DataFrame
    results_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    # validate 'experiments/hpo' directory exists before saving
    csv_dir = os.path.join("experiments", "hpo")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "optuna_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")