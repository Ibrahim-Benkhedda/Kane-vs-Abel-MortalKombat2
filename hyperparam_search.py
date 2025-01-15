import optuna
import os
import pandas as pd 

from mk_env import MkEnvWrapper

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium.wrappers import TimeLimit

LOG_DIR = "experiments/hpo/logs" 
OPT_DIR = "experiments/hpo/models"
TOTAL_TIMESTEPS = 200000
N_EVAL_EPISODES = 5

# function to retrun tested hyperparameters
def optimize_hyperparameters(trial):
    return {
        "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
        "buffer_size" : trial.suggest_int("buffer_size", 50000, 500000, step=50000),
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

        # create the env 
        def make_env():
            env = MkEnvWrapper(
                game="MortalKombatII-Genesis",
                state="Level1.LiuKangVsJax",
                render_mode=None,
            )
            
            env = TimeLimit(env, max_episode_steps=4500)
            env = Monitor(env, LOG_DIR)
            return env
        
        venv = DummyVecEnv([make_env])

        stacked_env = VecFrameStack(venv, 4)

        # setup the DQN model
        model = DQN(
            "CnnPolicy",
            stacked_env,
            tensorboard_log=LOG_DIR,
            verbose=0,
            **model_params
        )

        model.learn(total_timesteps=TOTAL_TIMESTEPS)

        mean_reward, _ = evaluate_policy(model, stacked_env, n_eval_episodes=N_EVAL_EPISODES)

        stacked_env.close()

        SAVE_PATH = os.path.join(OPT_DIR, f"DQN_{trial.number}")
        model.save(SAVE_PATH)

        return mean_reward
    
    except Exception as e:
        return -9999.0
    

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optimize_agent, n_trials=15, n_jobs=1)
    print("Number of finished trials: ", len(study.trials))

    # convert trials to DataFrame
    results_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

    # validate 'experiments/hpo' directory exists before saving
    csv_dir = os.path.join("experiments", "hpo")
    os.makedirs(csv_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "optuna_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")