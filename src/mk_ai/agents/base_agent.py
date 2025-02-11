import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, Any

class Agent(ABC):
    """Base class for all agent types."""
    
    @abstractmethod
    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Select an action based on the observation and additional info.

        Parameters:
            obs (np.ndarray): The current observation from the environment.
            info (Dict[str, Any]): Additional information from the environment.

        Returns:
            int: The selected action.
        """
        pass