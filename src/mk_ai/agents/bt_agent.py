import numpy as np
import os

from typing import List, Tuple, Dict, Any, Optional, Callable
from mk_ai.agents import Agent
from mk_ai.agents.BT.nodes import NodeStatus, Node
from mk_ai.agents.BT.game_context import GameStateContext
from mk_ai.agents.BT.loader import BTLoader
from mk_ai.agents.BT.conditions import ConditionsProvider
from mk_ai.utils import ActionGenerator
from mk_ai.configs import BUTTONS


class BTAgent(Agent):
    """
    Encapsulates the Behavior Tree and a blackboard (GameStateContext).
    """
    def __init__(self, buttons: List[str], bt_file_path: Optional[str] = None) -> None:
        """
        Paramters:
            buttons (List[str]): The list of environment buttons.
            bt_file_path (Optional[str]): Path to the behavior tree YAML file.
        """
        self.buttons = buttons
        self.context = GameStateContext()
        
        

        # file path to the action config and behavior tree files
        action_config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "env_config.yaml")

        # Default behavior tree path if none provided
        if bt_file_path is None:
            bt_file_path = os.path.join(os.path.dirname(__file__), "BT", "default_bt.yaml")

        # Map condition names from YAML to actual functions.
        condition_map = ConditionsProvider.gen_condition_map()

        print(f"[BT::BTAgent] Loading BT from {bt_file_path}")
        print(f"[BT::BTAgent] Available conditions: {condition_map.keys()}")
        print(f"[BT::BTAgent] Available actions: {self.buttons}")

        # Load the action generator and build the action map
        action_gen = ActionGenerator(filename=action_config_file)
        action_gen.build()

        # Map action placeholder names to actual action IDs.
        action_map: Dict[str, int] = action_gen.action_map

        # Load and Generate the Behavior Tree from the YAML file
        bt_loader: BTLoader = BTLoader(condition_map, action_map)
        self.bt_root: Node = bt_loader.gen_bt(yaml_file=bt_file_path)

    def action_to_env(self, actions: List[str]) -> List[int]:
        """
        Converts a list of button names to the Retro-compatible
        binary array [0,0,1,...] of length len(self.buttons).
        """
        action_array: List[int] = [0] * len(self.buttons)
        for btn in actions:
            if btn in self.buttons:
                idx: int = self.buttons.index(btn)
                action_array[idx] = 1
        return action_array

    def update_context(self, info: dict) -> None:
        """
        Copy relevant data from the provided game state information (RAM data)
        into the blackboard

        Parameters:
            info (Dict[str, Any]): Dictionary containing game state RAM information.
        """
        self.context.player_x = info.get("x_position", 0)
        self.context.enemy_x  = info.get("enemy_x_position", 0)
        self.context.player_y = info.get("y_position", 0)
        self.context.enemy_y  = info.get("enemy_y_position", 0)

        # print(f"[BT::BTAgent] self.context.get_distance_x")

    def tick(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Update the context and tick the Behavior Tree.
        If the tree's branch completes (SUCCESS or FAILURE), its state is reset.

        Parameters:
            obs (np.ndarray): The current observation from the environment (unused here).
            info (Dict[str, Any]): Additional game state information.
        
        Returns:
            int: The action ID selected by the Behavior Tree.
        """
        self.update_context(info)

        # Tick the behavior tree using the current game state context
        status: NodeStatus = self.bt_root.tick(self.context)

        # Get the current action ID from the BT
        action_id: Optional[int] = self.bt_root.get_action_id()

        # If the BT returns a terminal status, reset its state.
        if status in (NodeStatus.SUCCESS, NodeStatus.FAILURE):
            self.bt_root.reset()

        # If the BT failed or no valid action was found, return the fallback action (NEUTRAL)
        if status == NodeStatus.FAILURE or action_id is None:
            return 0  # Fallback/no-op action.
        return action_id
    
    def select_action(self, obs: np.ndarray, info: Dict[str, Any]) -> int:
        """
        Evaluate the tree. If the root is RUNNING or SUCCESS, 
        we can retrieve the current action_id from the root via get_action_id().
        If the root fails => fallback to 0 or some no-op.
        """
        return self.tick(obs, info)
    
if __name__ == "__main__":
    agent = BTAgent(buttons=BUTTONS)
    # Simulate some game state info
    test_info = {
        "x_position": 100,
        "enemy_x_position": 200,
        "y_position": 0,
        "enemy_y_position": 0,
    }
    action = agent.select_action(None, test_info)
    print("Selected action:", action)
