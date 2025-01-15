class ActionGenerator: 
    """
    This class is used to generate predefined actions for retro
    environment based on button combinations
    """
    def __init__(self, buttons):
        """
        initializes the class with a list of buttons

        """
        self.buttons = buttons
        self.actions = []

    def add_action(self, action):
        """
        Add a single action to the action list.
        :param action: A single action (e.g., ["B", "DOWN"]).
        """
        if not isinstance(action, list):
            raise ValueError("Action must be a list of button names.")
        self.actions.append(action)

    def add_actions(self, actions):
        """
        Add a new action as a combination of buttons.

        Params:
            action: List of button names (e.g. ["LEFT", "DOWN"])
        """
        if not all(isinstance(action, list) for action in actions):
            raise ValueError("Each action in actions must be a list of button names.")
        self.actions.extend(actions)

    def generate_action_mapping(self):
        """
        generates all possible combinations of actions based on the buttons

        Params:
            - return a list of binary arrays where each array is a button combination
        """
        action_mapping = []
        for action in self.actions:
            # create a binary array for each action 
            binary_action = [0] * len(self.buttons)
            for button in action:
                binary_action[self.buttons.index(button)] = 1
            action_mapping.append(binary_action)
        return action_mapping
    