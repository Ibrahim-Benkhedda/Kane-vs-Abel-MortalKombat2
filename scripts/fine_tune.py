from mk_env import MkEnvWrapper
from agents import DuelingDoubleDQN, DoubleDQN, DuelingDQN
from utils.schedulers import exponential_decay_lr, linear_decay_lr
from callbacks import CurriculumCallback, CustomEvalCallback

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

# ======================
# Configuration
# ======================        
MODEL_PATH = "models/Kane_DQN.zip"  
CHECKPOINT_DIR = "models/checkpoints" 
TOTAL_TIMESTEPS = 1000000  
SAVE_EVERY_STEPS = 10000 
MODEL_DIR = "models"
LOG_DIR = "logs"
NUM_ENVS = 8

# 1. create the environment using lambda to pass it to DummyVecEnv
def make_env():
    env = MkEnvWrapper(
        game="MortalKombatII-Genesis",
        state="Level1.LiuKangVsJax",
        render_mode="human",
        record="none"
    )

    env = Monitor(env)
    return env 

def train():
    # 2. vectorize the environment
    venv = SubprocVecEnv([make_env for _ in range(NUM_ENVS)])

    # 3. stack 4 frames together for the vectorized environment
    stacked_env = VecFrameStack(venv, 4)

    # 4. load pretrained model
    model = DQN.load(MODEL_PATH, env=stacked_env)
    
    # 5 . Train the model for a total number of timesteps
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

