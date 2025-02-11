from .mk_env import MkEnvWrapper
from typing import Tuple

class MultiAgentMkEnvWrapper(MkEnvWrapper):
    """Handles 2-player action combination and info tracking."""

    def __init__(self, *args, **kwargs):
        """
        Initialize the MultiAgentMkEnvWrapper.

        Parameters:
            - *args: Positional arguments for the MkEnvWrapper.
            - **kwargs: Keyword arguments for the MkEnvWrapper.
        """
        super().__init__(*args, **kwargs)
        self.current_info = None  # Track current environment info
        self.original_obs = None  # Stores current raw observation

    def step(self, actions: Tuple[int, int]):
        """
        Perform a step in the environment with combined actions.

        Parameters:
            - actions (Tuple[int, int]): Actions for both players.

        Returns:
            - Tuple: Processed observation, reward, done, truncated, and info.
        """
        p1_array = self._action_mapping[actions[0]]
        p2_array = self._action_mapping[actions[1]]
        combined = p1_array + p2_array

        raw_obs, reward, done, truncated, info = super(MkEnvWrapper, self).step(combined)

        self.original_obs = raw_obs
        processed_obs = self._preprocess_frame(self.original_obs)

        self.current_info = info

        return processed_obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment and get the initial observation.

        Parameters:
            - **kwargs: Additional arguments for the reset method.

        Returns:
            - Tuple: Initial observation and info.
        """
        obs, info = super().reset(**kwargs)
        self.current_info = info
        return obs, info
    

# ===================================================================================================================
# The code below im using it for testing only where im comparing implentation between encapsulation and inheritance 
# ===================================================================================================================

# class MultiAgentMkEnvWrapper:
#     """
#     A multi-agent environment wrapper that composes an instance of MkEnvWrapper.
#     It handles combining actions from two agents and tracking additional info.
#     """
#     def __init__(self, mk_env: MkEnvWrapper):
#         """
#         Initialize the multi-agent wrapper.

#         Parameters:
#             mk_env (MkEnvWrapper): An instance of the single-agent environment wrapper.
#         """
#         self.env = mk_env  # Composition: our multi-agent wrapper "has-a" MkEnvWrapper
#         self.current_info = None  # To track additional environment info
#         self.original_obs = None  # To store the raw observation

#     def combine_actions(self, action1, action2):
#         """
#         Combine the actions for two players.
        
#         Parameters:
#             action1: Action (or array) for player 1.
#             action2: Action (or array) for player 2.
            
#         Returns:
#             The combined action in the format expected by the underlying environment.
#         """
#         # For example, if actions are lists or numpy arrays, you might choose to concatenate them:
#         return action1 + action2  # Adjust if you need different logic (e.g., bitwise OR)

#     def step(self, actions: Tuple[int, int]):
#         """
#         Step the environment using a tuple of two action indices.

#         Parameters:
#             actions (Tuple[int, int]): Actions for player 1 and player 2.

#         Returns:
#             Tuple containing processed observation, reward, done, truncated, and info.
#         """
#         # Map indices to actual actions using the underlying environment's mapping.
#         p1_action = self.env._action_mapping[actions[0]]
#         p2_action = self.env._action_mapping[actions[1]]
        
#         # Combine actions in a way that the underlying environment understands.
#         combined = self.combine_actions(p1_action, p2_action)
        
#         # Step the underlying environment.
#         raw_obs, reward, done, truncated, info = self.env.step(combined)
        
#         # Save the raw observation.
#         self.original_obs = raw_obs
        
#         # Preprocess the observation using the underlying environment's method.
#         processed_obs = self.env._preprocess_frame(raw_obs)
        
#         # Update our info tracking.
#         self.current_info = info
        
#         return processed_obs, reward, done, truncated, info

#     def reset(self, **kwargs):
#         """
#         Reset the underlying environment.

#         Parameters:
#             **kwargs: Additional arguments to pass to the underlying environment's reset method.

#         Returns:
#             Tuple of the initial observation and info.
#         """
#         obs, info = self.env.reset(**kwargs)
#         self.current_info = info
#         return obs, info

#     def close(self):
#         """Close the underlying environment."""
#         self.env.close()