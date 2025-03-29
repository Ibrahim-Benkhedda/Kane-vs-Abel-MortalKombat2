# =============================================================================
# Env Model (Encapsulates the Environment)
# =============================================================================

class EnvModel:
    """
    Encapsulates the game environment logic.
    """
    def __init__(self, env_wrapper, elo_manager, agent_ids):
        self.env = env_wrapper
        self.elo_manager = elo_manager
        self.agent_ids = agent_ids or ["Agent1", "Agent2"]
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

        # Update Elo ratings if the episode is done
        if done or truncated:
            # Determine winner based on rounds won.
            if self.info.get("rounds_won", 0) == 2:
                winner = self.agent_ids[0]
                loser = self.agent_ids[1]
            elif self.info.get("enemy_rounds_won", 0) == 2:
                winner = self.agent_ids[1]
                loser = self.agent_ids[0]
            else:
                winner = None

            # Update Elo ratings if the Elo manager is available.
            if winner and self.elo_manager:
                self.elo_manager.update_ratings(winner, loser, winner)

            self.reset()

        return reward 

    def close(self):
        """Close the environment."""
        self.env.close()