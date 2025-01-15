from mk_env import MkEnvWrapper
from frameskip import DeterministicFrameSkip, StochasticFrameSkip
from double_dqn import DoubleDQN

from stable_baselines3 import DQN
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

MODEL_DIR = "models"
LOG_DIR = "logs"
TOTAL_TIMESTEPS = 4000000

def exponential_decay_lr(initial_lr: float, decay_rate: float):
    # progress_remaining is 1 -> 0, so we can invert it if we want time steps
    # e.g., a quick hack is to do: steps = (1 - progress_remaining) * total_timesteps
    # NOTE: but typically you'd do something simpler or just do linear for RL.
    def lr_schedule(progress_remaining: float) -> float:
        return initial_lr * (decay_rate ** (1.0 - progress_remaining))
    return lr_schedule

def linear_decay_lr(initial_lr, final_lr):
    """
    Linearly decay from initial_lr (at progress=1) to final_lr (at progress=0).
    """
    def lr_schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return lr_schedule

# 1. create the environment using lambda to pass it to DummyVecEnv
def make_env():
    env = MkEnvWrapper(
        game="MortalKombatII-Genesis",
        state="Level1.LiuKangVsJax",
        render_mode="none",
        record="replays"
    )

    env = DeterministicFrameSkip(env, n=4)
    env = Monitor(env)
    return env 

def train():
    # 2. vectorize the environment
    venv = DummyVecEnv([make_env])

    # 3. stack 4 frames together for the vectorized environment
    stacked_env = VecFrameStack(venv, 4)
    # 4. Initialize the DQN model
    # lr_schedule = linear_decay_lr(3.16e-4, 1e-5)
    exp_decay_lr = exponential_decay_lr(3.16e-4, 0.295)
    model = DoubleDQN(
        "CnnPolicy",
        stacked_env,
        verbose=1,
        device="cuda",
        buffer_size=200000,
        batch_size=32,
        gamma=0.948594691717984,
        learning_rate=exp_decay_lr,
        # learning_rate=0.00031637183030275843,
        exploration_fraction=0.2992653791117977,
        exploration_initial_eps=0.9251553723683702,
        exploration_final_eps=0.07284636692467307,
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
    model.save("DDQN_4M_with_lrSchedule_frameskip.zip")

    # 7. Close the environment
    venv.close()
    print("Training complete.")


if __name__ == "__main__":
    train()

