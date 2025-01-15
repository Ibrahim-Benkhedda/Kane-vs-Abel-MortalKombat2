## Packages 
- Stable-Retro: Provides the environment (e.g., the game).
- Gymnasium: Acts as the interface for interacting with the environment (reset, step, render, etc.).
- Stable-Baselines3: Provides the RL algorithms to train an agent in the environment.


## Mortal Kombat II: Genesis Action Space 
 Since Mortal Kombat uses a 12-button controller, and each button can either be pressed (1) or not pressed (0), the total number of possible combinations of button states is:

2
12
=
4096
2 
12
 =4096
However, many of these combinations are invalid or nonsensical for gameplay. For example:

Pressing UP and DOWN simultaneously doesn't make sense.
Pressing START and other buttons together may not trigger meaningful gameplay actions.
So, while 4096 is the theoretical action space size, the effective or valid action space is much smaller.

reates a Set of Valid Combinations:

Instead of letting the agent explore all 4096 possible combinations (most of which are useless or invalid), the example defines a curated list of meaningful actions.
These actions are defined as combinations of buttons, e.g., ['LEFT', 'DOWN'] or ['RIGHT', 'UP', 'B'].
Maps Action Names to Button Combinations:

For each action (like ['LEFT', 'DOWN']), it creates a binary array of size 12 where:
Each element corresponds to whether a button is pressed (True) or not (False).
The position of each element matches the index of the button in the buttons list.
Example:
['LEFT', 'DOWN'] → [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]
Changes the Action Space:

Replaces the original MultiDiscrete action space (4096 combinations) with a Discrete action space.
In the new space:
Each discrete action corresponds to one of the predefined valid combinations.
For example, Action 0 might map to [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] (pressing LEFT and DOWN), and Action 1 might map to [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] (pressing B).
Overrides the action() Method:

The action() method takes a single Discrete action index from the agent and maps it to the corresponding binary array of button presses.
This binary array is passed to the underlying environment, which interprets it as the buttons being pressed for that step.
