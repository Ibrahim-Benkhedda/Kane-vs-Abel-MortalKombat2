from mk_ai.utils import ActionGenerator
from typing import List

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