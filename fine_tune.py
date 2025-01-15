from mk_env import MkEnvWrapper

from stable_baselines3 import DQN
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor


MODEL_PATH = "models/Kane_DQN.zip"  
CHECKPOINT_DIR = "models/checkpoints" 
TOTAL_TIMESTEPS = 1000000  
SAVE_EVERY_STEPS = 10000 
MODEL_DIR = "models"
LOG_DIR = "logs"

# 1. create the environment using lambda to pass it to DummyVecEnv
def make_env():
    env = MkEnvWrapper(
        game="MortalKombatII-Genesis",
        state="Level1.LiuKangVsJax",
        render_mode="human",
        record="replays"
    )

    env = Monitor(env)
    return env 

def train():
    # 2. vectorize the environment
    venv = DummyVecEnv([make_env])

    # 3. stack 4 frames together for the vectorized environment
    stacked_env = VecFrameStack(venv, 4)

    # 4. Initialize the DQN model
    model = DQN(
        "CnnPolicy",
        stacked_env,
        verbose=1,
        device="cuda",
        buffer_size=100000,
        batch_size=32,
        gamma=0.94,
        learning_starts=100,
        learning_rate=1e-3,
        # exploration_fraction=0.2,  
        # exploration_initial_eps=1.0,  
        # exploration_final_eps=0.05,  
        # train_freq=4,  
        # gradient_steps=2, 
        # target_update_interval=5000,
        tensorboard_log="./expirements/logs/"
    )

    # model = PPO(
    #     "CnnPolicy",
    #     stacked_env,
    #     device="cuda", 
    #     verbose=1,
    #     n_steps=512,
    #     batch_size=512,
    #     n_epochs=4,
    #     gamma=0.94,
    #     tensorboard_log="./Prototype/logs/"
    # )

    # action_logging_callback = ActionFrequencyLoggingCallback(log_freq=1_000, verbose=1)

    # 5. Train the model for a total number of timesteps
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=True,
        # callback=action_logging_callback
    )

    # 6. Save the trained model
    model.save("Kane_DQN.zip")

    # 7. Close the environment
    venv.close()
    print("Training complete.")


if __name__ == "__main__":
    train()

