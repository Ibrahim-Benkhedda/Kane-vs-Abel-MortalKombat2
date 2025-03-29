import yaml 
from mk_ai.agents.BT.nodes import *
from mk_ai.agents.BT.conditions import is_close_to_enemy, is_enemy_to_the_left, is_enemy_to_the_right, is_long_range_enemy

class BTLoader:
    def __init__(self, condition_map: dict, action_map: dict) -> None:
        """
        Initialize the loader with the given mappings of actions and conditions.

        Paramters: 
            condition_map (dict): Maps condition names to actual functions.
            action_map (dict): Maps action ID names to actual numeric IDs.
        """

        self.condition_map = condition_map
        self.action_map = action_map


    def gen_node(self, node_spec: dict) -> Node:
        """
            Recursively create a Behaviour tree node from YAML 

            Parameters: 
                node_spec (dict): A dictionary representing the node specification loaded from YAML

            Returns:
                Node: An instance of one of the Behavior Tree node classes (Selector, Sequence,
                    Action, or Condition) based on the 'type' field in the specification.
        """

        # extract the node type, name and its properties from the specification
        node_type = node_spec.get("type")
        name = node_spec.get("name")
        properties = node_spec.get("properties", {})

        # recursively create child nodes from the children list
        children_specs = node_spec.get("children", [])
        children = [self.gen_node(child) for child in children_specs]

        # based on the node type, create its corresponding Behaviour Tree node
        if node_type == "Selector":
            # create a Selector node with the given name and its child nodes
            return Selector(name=name, children=children)
        
        elif node_type == "Sequence":
            # create a Sequence node with the given name and its child nodes
            return Sequence(name=name, children=children)
        
        elif node_type == "Action":
            # get the action_id placeholder string from the properties
            action_id_name = properties.get("action_id")
            # map the placeholder to the actual action ID using ACTION_ID_MAP
            action_id = self.action_map.get(action_id_name, 0)
            # get the number of frames needed; default to 1 if not specified
            frames_needed = properties.get("frames_needed", 1)

            # validate that frames_needed is a positive integer 
            if not isinstance(frames_needed, int) or frames_needed < 1:
                raise ValueError(f"[BT::Loader] Action node '{name}' has an invalid 'frames_needed' value: {frames_needed}")
            
            # create and return the Action node
            return Action(name=name, action_id=action_id, frames_needed=frames_needed)
        
        elif node_type == "Condition":
            # get the condition name from the properties
            condition_name = properties.get("condition")
            # map the condition name to the actual function using a conditions map
            condition = self.condition_map.get(condition_name)

            # raise an error if the condition function does not exist
            if condition is None:
                raise ValueError(f"[BT::Loader] Condition '{condition_name}' not defined in CONDITION_MAP")
            
            # create and return the Condition node
            return Condition(name=name, condition=condition)
        
        else:
            raise ValueError(f"[BT::Loader] Unknown node type: {node_type}")
        
    def gen_bt(self, yaml_file: str) -> Node:
        """
        Loads a Behaviour Tree from a YAML file

        Parameters:
            yaml_file (str): The path to the YAML file that contains the behavior tree specification

        Returns:
            Node: The root node of the generated Behavior Tree
        """
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        # the top-level key "node" in the YAML file holds the root node specification
        return self.gen_node(data["node"])

if __name__ == "__main__":
    # this for testing only
    # map condition names from yaml file to actual function  
    condition_map = {
        "is_enemy_to_the_right": is_enemy_to_the_right,
        "is_enemy_to_the_left": is_enemy_to_the_left,
        "is_close_to_enemy": is_close_to_enemy,
        "is_long_range_enemy": is_long_range_enemy,
    }
    action_map = {
        "NEUTRAL_ID": 0,
        "MOVE_RIGHT_ID": 1,
        "MOVE_LEFT_ID": 2,
        "JUMP_ID": 3
    }
    
    btloader = BTLoader(condition_map, action_map)
    bt_root = btloader.gen_bt("bt.yaml")
    print("BT generated successfully")