import retro
import time
import gymnasium as gym 
import numpy as np 

class DeterministicFrameSkip(gym.Wrapper):
    """
    A wrapper for deterministic frame skipping, where the given action
    is repeated for a fixed number of frames.
    """
    def __init__(self, env, n) -> None:
        """
        Initialize the DeterministicFrameSkip wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            n (int): The number of frames to skip.
        """
        super().__init__(env) 
        self.n = n
        self.supports_want_render = hasattr(env, "supports_want_render")
        self.total_env_steps = 0
        self.total_skipped_steps = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Reset the environment and clear the current action.

        Args:
            **kwargs: Additional arguments for the reset method.

        Returns:
            observation: The initial observation of the environment.
            info: Additional reset information.
        """
        return self.env.reset(**kwargs)

    def step(self, action: list[str]) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform the given action for `n` frames.

        Args:
            action: The action to execute.

        Returns:
            observation: The observation after the final skipped frame.
            total_reward: The accumulated reward over the skipped frames.
            terminated: Whether the episode has terminated.
            truncated: Whether the episode has been truncated.
            info: Additional information from the environment.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        observation = None

        for i in range(self.n):
            if self.supports_want_render and i < self.n - 1:
                obs, reward, terminated, truncated, info = self.env.step(action, want_render=False)
            else:
                obs, reward, terminated, truncated, info = self.env.step(action)

            # for updating the tracking variables for debugging purposes.
            self.total_env_steps +=1
            if i > 0:
                self.total_skipped_steps +=1

            total_reward += reward
            observation = obs  

            if terminated or truncated:
                break
        
        # self.print_steps()
        
        return observation, total_reward, terminated, truncated, info
    
    def print_steps(self) -> None:
        print(f"Total environment steps: {self.total_env_steps}")
        print(f"Total skipped steps: {self.total_skipped_steps}")
        print(f"Effective agent steps: {self.total_env_steps - self.total_skipped_steps}")

        
class StochasticFrameSkip(gym.Wrapper):
    """
    This was taken from the Stable-Retro example:
    https://stable-retro.farama.org/index.html
    """
    def __init__(self, env, n, stickprob) -> None:
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac) -> tuple[np.ndarray, float, bool, bool, dict]:
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info
        