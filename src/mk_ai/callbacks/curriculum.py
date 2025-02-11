from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class CurriculumCallback(BaseCallback):
    """
    Switch from:
    - Tier 1 states to Tier states if average episode reward is above a threshold
    - Tier 2 states to Tier 3 states if average episode reward is above a threshold
    """
    def __init__(self, vec_env, tiered_states, verbose=0, buffer_size=20):
        super().__init__(verbose)
        self.vec_env = vec_env
        # list of states for each tier, something like [[tier1_states], [tier2_states], [tier3_states]]
        self.tiered_states = tiered_states
        self.current_tier_idx = 0
        self.buffer_size = buffer_size
        self.episode_rewards = deque(maxlen=self.buffer_size)
        self.episode_count = 0

    def _on_training_start(self):
        """
        When training starts, set the states to the tier 1 states
        """
        init_states = self.tiered_states[self.current_tier_idx]
        self.vec_env.env_method("set_states", init_states)

    def _on_step(self):
        """
        On each step, update the episode reward buffer and check if the average reward is above the threshold
        """
        self.episode_rewards.append(self.training_env.get_episode_rewards())

    def _on_step(self) -> bool:
        # "dones" is an array of booleans for each parallel environment
        dones = self.locals.get("dones", [])
        rewards = self.locals.get("rewards", [])

        if not hasattr(self, "episode_returns"):
            # track partial returns for each sub-env
            self.episode_returns = [0.0] * self.training_env.num_envs

        # Update partial returns and detect end of episodes
        for i, done in enumerate(dones):
            self.episode_returns[i] += rewards[i]
            if done:
                # Episode finished for environment i
                final_return = self.episode_returns[i]
                self.episode_rewards.append(final_return)
                self.episode_returns[i] = 0.0
                self.episode_count += 1

        # Every time an episode ends, check rolling average reward
        if self.episode_count > 0 and (any(dones)):
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)

            # if Tier 1, check if avg_reward > 50 => move to Tier 2
            if self.current_tier_idx == 0 and avg_reward > 50:
                self.current_tier_idx = 1
                print(f"[Callback] Switching to Tier 2, avg_reward={avg_reward:.2f}")
                self._update_env_states()

            # If Tier 2, check if avg_reward > 150 => move to Tier 3
            elif self.current_tier_idx == 1 and avg_reward > 150:
                self.current_tier_idx = 2
                print(f"[Callback] Switching to Tier 3, avg_reward={avg_reward:.2f}")
                self._update_env_states()

            # If Tier 3, check if avg_reward > 250 => move to Tier 4
            elif self.current_tier_idx == 2 and avg_reward > 250:
                self.current_tier_idx = 3
                print(f"[Callback] Switching to Tier 4, avg_reward={avg_reward:.2f}")
                self._update_env_states()

        return True

    def _update_env_states(self):
        """Update the env states to the next tier."""
        new_states = self.tiered_states[self.current_tier_idx]
        self.vec_env.env_method("set_states", new_states)