# =============================================================================
# Env Model (Encapsulates the Environment)
# =============================================================================

class EnvModel:
    """
    Encapsulates the game environment logic.
    """
    def __init__(self, env_wrapper):
        self.env = env_wrapper

        self.obs = None
        self.info = None
        self.reset()
    
    def reset(self):
        """Reset the environment and retrieve the initial observation."""
        self.obs, self.info = self.env.reset()
    
    def step(self, actions):
        """
        Step through the environment using the provided actions.
        If the episode ends, the environment is reset.
        """
        self.obs, reward, done, truncated, self.info = self.env.step(tuple(actions))
        if done or truncated:
            self.reset()
        return reward 

    def close(self):
        self.env.close()