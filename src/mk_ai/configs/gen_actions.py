import yaml
from mk_ai.utils import ActionGenerator
from typing import List

# def gen_action_mapping(actions: List[str], buttons: List[str]):
#     """
#     Uses ActionGenerator to dynamically generate a mapping of discrete actions.
    
#     Returns:
#         tuple:
#             - mapping (dict): A mapping from a string key (e.g., "LEFT_DOWN") to a numeric ID.
#             - binary_mapping (list): A list of binary arrays representing each action.
#     """

#     # init the generator with a buttons array 
#     action_gen = ActionGenerator(buttons=buttons)
#     # add actions 
#     action_gen.add_actions(actions=actions)
#     # generate binary mapping and populate the reverse lookup 
#     binary_mapping = action_gen.generate_action_mapping_with_lookup()

#     # create a dict mapping 
#     mapping = {}
#     for combo, idx in action_gen.combo_to_id.items():
#         # Create a key by joining the combo with underscores.
#         key = "_".join(combo) if combo else "NEUTRAL"
#         mapping[key] = idx
    
#     return mapping, binary_mapping

# def load_env_actions(filename: str):
#     """
#     Loads action space from a defined file (Yaml) and returns them as tuple
#     """
#     with open(filename, "r") as f:
#         data = yaml.safe_load(f)
#     return data["buttons"], data["actions"]



if __name__ == "__main__":
    action_gen = ActionGenerator(filename="env_config.yaml")
    action_gen.build()

    BUTTONS = action_gen.buttons 
    ACTIONS = action_gen.actions 
    ACTION_MAP = action_gen.action_map
    ACTIONS_BINARY = action_gen.binary_mapping
    
    print(f" [ENV_ACTIONS] Generated Buttons from YAML: {BUTTONS}")
    print(f" [ENV_ACTIONS] Generated Actions from YAML: {ACTIONS}")

    print(f" [ENV_ACTIONS] Generated ACTION_MAP: {ACTION_MAP}")
    print(f" [ENV_ACTIONS] Generated ACTION_BINARY_MAPPING: {ACTIONS_BINARY}")