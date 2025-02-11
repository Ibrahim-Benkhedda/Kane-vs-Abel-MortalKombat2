from pyglet.window import key as keycodes
from mk_ai.utils import ActionGenerator
# Button mappings for the Sega Genesis controller
BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]

# Define key mapping for Player 1
P1_KEY_MAP = {
    keycodes.UP: "UP",
    keycodes.DOWN: "DOWN",
    keycodes.LEFT: "LEFT",
    keycodes.RIGHT: "RIGHT",
    keycodes.Z: "A",
    keycodes.X: "B",
    keycodes.C: "C",
    keycodes.ENTER: "START",
}

# Define key mapping for Player 2
P2_KEY_MAP = {
    keycodes.W: "UP",
    keycodes.S: "DOWN",
    keycodes.A: "LEFT",
    keycodes.D: "RIGHT",
    keycodes.T: "A",
    keycodes.Y: "B",
    keycodes.U: "C",
    keycodes.ENTER: "START",
}

# Predefined actions as button combinations
ACTIONS = [
    [], ['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['LEFT', 'UP'], ['RIGHT', 'UP'],
    ['DOWN', 'B'], ['LEFT', 'UP'], ['RIGHT', 'DOWN', 'B'], ['RIGHT', 'DOWN', 'A'],
    ['RIGHT', 'UP', 'B'], ['RIGHT', 'UP', 'A'], ['RIGHT', 'UP', 'C'],
    ['LEFT', 'UP', 'B'], ['LEFT', 'UP', 'A'], ['LEFT', 'UP', 'C'],
    ['C'], ['START'], ['B'], ['Y'], ['X'], ['Z'], ['A'], ['UP'], ['MODE']
]

# Minimal actions
ACTIONS_MINIMAL = [ ['LEFT'], ['RIGHT'], ['A'], ['B'], ['START'] ]

# Minimal actions with jump attacks
ACTIONS_MINIMAL_WITH_JUMP = [ ['LEFT'], ['RIGHT'], ['A'], ['B'], ['START'],
            ['RIGHT', 'DOWN', 'B'],  ['RIGHT', 'UP', 'C'], ['LEFT', 'UP', 'C'] 
]

# Minimal actions with jump and crouch attacks
ACTIONS_MINIMAL_WITH_JUMP_AND_PUNCH = [ ['LEFT'], ['RIGHT'], ['A'], ['B'], ['START'],
            ['RIGHT', 'DOWN', 'B'],  ['RIGHT', 'UP', 'C'], ['LEFT', 'UP', 'C'], ['C']
]


if __name__ == "__main__":
    actionGenerator = ActionGenerator(buttons=BUTTONS)
    actionGenerator.add_actions(ACTIONS)

    actionGenerator.generate_action_mapping_with_lookup()

    action_mapping = {}
    for combo, idx in actionGenerator._combo_to_id.items():
        key = "_".join(combo) if combo else "NEUTRAL"
        action_mapping[key] = idx 

    print("Dynamic action mapping:", action_mapping)

