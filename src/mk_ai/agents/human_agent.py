import pyglet
import numpy as np

from pyglet.window import key as keycodes
from typing import List, Tuple, Dict, Any, Optional, Set
from mk_ai.agents import Agent
from mk_ai.configs import P1_KEY_MAP, P2_KEY_MAP

class HumanAgent(Agent):
    """Handles human input through keyboard."""
    
    def __init__(self, action_mapping: List[List[int]], buttons: List[str], player_num: int = 1):
        """
        Initialize a HumanAgent.

        Parameters:
            - action_mapping (List[List[int]]): Mapping of actions to button presses.
            - buttons (List[str]): List of button names.
            - player_num (int): Player number (1 or 2) to determine key mapping.
        """
        self.action_mapping: List[List[int]] = action_mapping
        self.buttons: List[str] = buttons
        self.pressed_keys: Set[int] = set()
        self.key_map = P1_KEY_MAP if player_num == 1 else P2_KEY_MAP

    def update_keys(self, pressed_keys: set):
        """
        Update the set of currently pressed keys.

        Parameters:
            - pressed_keys (set): Set of currently pressed keys.
        """
        self.pressed_keys = pressed_keys

    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Select an action based on the current pressed keys.

        Parameters:
            - obs (np.ndarray): The current observation from the environment.
            - info (Dict[str, Any]): Additional information from the environment.

        Returns:
            - int: The selected action.
        """
        action_array = [0] * len(self.buttons)
        for key in self.pressed_keys:
            if key in self.key_map: 
                button = self.key_map[key]
                if button in self.buttons:
                    action_array[self.buttons.index(button)] = 1
        
        # Debug: Print generated action array
        print(f"[HumanAgent] Player {1 if self.key_map is P1_KEY_MAP else 2} action array:", action_array)

        # Find matching action ID
        for action_id, arr in enumerate(self.action_mapping):
            if arr == action_array:
                return action_id
        return 0