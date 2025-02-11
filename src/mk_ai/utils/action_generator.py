from typing import Optional

# file: mk_ai/action_generator.py

from typing import List, Dict, Tuple, Optional
import yaml

class ActionGenerator:
    """
    Manages the definition of buttons and actions, either loaded from a YAML file
    or directly assigned. Generates both a dictionary (action_map) and a binary
    representation (binary_mapping) for each action.

    Workflow:
      1. Instantiate with either a YAML filename or direct lists of buttons/actions.
      2. (Optional) call load_from_yaml(...) if you want to load or overwrite data.
      3. Call build() to generate action_map, binary_mapping, and combo_to_id.
      4. Access them via properties action_map, binary_mapping, and combo_to_id.
    """

    def __init__(
        self,
        filename: Optional[str] = None,
        buttons: Optional[List[str]] = None,
        actions: Optional[List[List[str]]] = None
    ) -> None:
        """
        Initialize the ActionGenerator, optionally from a YAML file or direct parameters.

        Args:
            filename (str, optional): Path to a YAML file with "buttons" and "actions" keys.
            buttons (List[str], optional): List of button names (e.g. ["LEFT", "RIGHT", "A"]).
            actions (List[List[str]], optional): Each element is a list of button names (e.g. ["LEFT", "DOWN"]).
        """
        self._buttons: List[str] = []
        self._actions: List[List[str]] = []

        # Internal structures (generated after build() is called)
        self._action_map: Dict[str, int] = {}
        self._binary_mapping: List[List[int]] = []
        self._combo_to_id: Dict[Tuple[str, ...], int] = {}

        # If a filename is provided, load from YAML
        if filename is not None:
            self.load_from_yaml(filename)
        else:
            # Otherwise, if direct data is provided, use it
            if buttons is not None:
                self._buttons = buttons
            if actions is not None:
                self._actions = actions

    def load_from_yaml(self, filename: str) -> None:
        """
        Load button and action data from a YAML file that must contain keys "buttons" and "actions".

        Args:
            filename (str): Path to the YAML file.

        Raises:
            FileNotFoundError: If the file cannot be opened.
            KeyError: If the file doesn't contain 'buttons' or 'actions' keys.
            yaml.YAMLError: If there's an error parsing the YAML file.
        """
        with open(filename, "r") as f:
            data = yaml.safe_load(f)

        if "buttons" not in data or "actions" not in data:
            raise KeyError("The YAML file must have 'buttons' and 'actions' keys.")

        self._buttons = data["buttons"]
        self._actions = data["actions"]
        # Reset previous results if we are re-loading
        self._action_map.clear()
        self._binary_mapping.clear()
        self._combo_to_id.clear()

    def add_action(self, action: List[str]) -> None:
        """
        Add a single action (list of button names).

        Args:
            action (List[str]): A single action (e.g., ["LEFT", "DOWN"]).

        Raises:
            ValueError: If the provided action is not a list of strings.
        """
        if not isinstance(action, list):
            raise ValueError("Action must be a list of button names.")
        self._actions.append(action)

    def add_actions(self, actions: List[List[str]]) -> None:
        """
        Add multiple actions at once.

        Args:
            actions (List[List[str]]): A list of actions, each an array of button names.

        Raises:
            ValueError: If any item in 'actions' is not a list of strings.
        """
        if not all(isinstance(a, list) for a in actions):
            raise ValueError("Each action in 'actions' must be a list of button names.")
        self._actions.extend(actions)

    def build(self) -> None:
        """
        Build or rebuild the action mappings. This method generates:
          - self._binary_mapping: A list of binary arrays for each action.
          - self._combo_to_id: A dict mapping from the tuple of button combos -> discrete ID.
          - self._action_map: A dict mapping from a string key (e.g., 'LEFT_DOWN') -> discrete ID.

        Must be called after setting or loading the buttons and actions.
        """
        # Clear any existing data
        self._binary_mapping.clear()
        self._combo_to_id.clear()
        self._action_map.clear()

        for idx, action in enumerate(self._actions):
            # Create a binary array for this particular combo of buttons
            binary_array = [0] * len(self._buttons)
            for btn in action:
                if btn in self._buttons:
                    binary_array[self._buttons.index(btn)] = 1

            # Store in _binary_mapping
            self._binary_mapping.append(binary_array)
            # Also store reverse lookup from combo -> ID
            self._combo_to_id[tuple(action)] = idx

        # Now build the "action_map" (string version -> ID)
        for combo, idx in self._combo_to_id.items():
            if combo:
                key = "_".join(combo)
            else:
                key = "NEUTRAL"  # For empty combos
            self._action_map[key] = idx

    @property
    def buttons(self) -> List[str]:
        """List of button names used for building actions."""
        return self._buttons

    @property
    def actions(self) -> List[List[str]]:
        """List of button combos (actions)."""
        return self._actions

    @property
    def action_map(self) -> Dict[str, int]:
        """
        Dictionary mapping an underscore-joined combo string (e.g., "LEFT_DOWN") to a discrete ID.
        Available after calling build().
        """
        return self._action_map

    @property
    def binary_mapping(self) -> List[List[int]]:
        """
        A list of binary arrays where each array corresponds to an action.
        Available after calling build().
        """
        return self._binary_mapping

    @property
    def combo_to_id(self) -> Dict[Tuple[str, ...], int]:
        """
        Reverse lookup that maps a tuple of button combos (e.g. ("LEFT", "DOWN")) to a discrete ID.
        Available after calling build().
        """
        return self._combo_to_id

    def get_action_id(self, combo: List[str]) -> int:
        """
        Return the discrete ID for a given combo of buttons.

        Args:
            combo (List[str]): The list of button names representing the action.

        Returns:
            int: Discrete ID assigned to this combo.

        Raises:
            KeyError: If the combo does not exist in the built dictionary.
        """
        return self._combo_to_id[tuple(combo)]

    