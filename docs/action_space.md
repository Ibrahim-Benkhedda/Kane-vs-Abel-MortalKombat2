# Understanding Action Spaces in Reinforcement Learning (Discrete vs. MultiDiscrete)

In reinforcement learning environments like those in **Gym**, action spaces define the range and structure of possible actions an agent can take. This document explains the key differences between **Discrete** and **MultiDiscrete** action spaces and provides examples to help clarify their use cases.

---

## **1. Discrete Action Space**
### Definition
A **Discrete** action space represents a single choice from a set of predefined actions. The agent selects one action per step, identified by an integer index.

### Key Characteristics
- **Single action per step**: The agent chooses one action at each timestep.
- **Size**: The total number of possible actions is equal to the number of indices in the space.
- **Structure**: Actions are indexed by integers starting from `0`.

### Example
Consider a simple game with three actions: `LEFT`, `RIGHT`, and `JUMP`.
```python
import gym

# Define the action space
action_space = gym.spaces.Discrete(3)  # 3 possible actions

# Possible actions:
# 0 -> LEFT
# 1 -> RIGHT
# 2 -> JUMP
```
#### How It Works
At each step, the agent outputs a single integer to indicate its action:
```python
action = 1  # The agent chooses action 'RIGHT'
```

---

## **2. MultiDiscrete Action Space**
### Definition
A **MultiDiscrete** action space represents multiple independent discrete actions, allowing the agent to choose a combination of actions at each step.

### Key Characteristics
- **Multiple simultaneous actions**: The agent selects a vector of actions, where each element corresponds to a discrete action in a specific dimension.
- **Size**: The total action space size is the product of possibilities across all dimensions.
- **Structure**: An array of integers, where each dimension is independent.

### Example
Consider a game with three independent actions:
- Action 1 (`PUNCH`) has 2 options: `[0, 1]`
- Action 2 (`KICK`) has 2 options: `[0, 1]`
- Action 3 (`JUMP`) has 2 options: `[0, 1]`

```python
import gym
import numpy as np

# Define the action space
action_space = gym.spaces.MultiDiscrete([2, 2, 2])

# Example actions:
# [0, 0, 0] -> No actions
# [1, 0, 1] -> PUNCH + JUMP
# [0, 1, 0] -> KICK
```
#### How It Works
At each step, the agent outputs an array where each element represents the state of a button or action:
```python
action = np.array([1, 0, 1])  # The agent chooses PUNCH and JUMP
```

---

## **3. Key Differences Between Discrete and MultiDiscrete**

| Feature                | Discrete                          | MultiDiscrete                      |
|------------------------|------------------------------------|-------------------------------------|
| **Structure**          | Single integer                   | Array/vector of integers           |
| **Action per Step**    | One action                       | Multiple simultaneous actions       |
| **Action Space Size**  | Number of actions                | Product of possibilities across dimensions |
| **Use Case**           | Simple games (e.g., arcade games) | Complex games (e.g., fighting games) or robotics |
| **Example**            | `0` (e.g., `LEFT`)               | `[1, 0, 1]` (e.g., `PUNCH` + `JUMP`) |

---

## **4. Practical Use Cases**
### When to Use Discrete
- Games or tasks where the agent selects **one action at a time**.
- Example: Moving a character left, right, or jumping.

### When to Use MultiDiscrete
- Games or tasks where the agent performs **multiple simultaneous actions**.
- Example: Fighting games like Mortal Kombat, where moves involve combinations of buttons (e.g., `DOWN + B`).

---

## **5. Example: Custom Action Wrapper**
In games like **Mortal Kombat**, you may want to define a curated set of valid actions to simplify the agent's decision-making. Here's an example of a custom action wrapper:

### Problem
By default, the action space for a 12-button controller is **MultiDiscrete([2, 2, 2, ..., 2])** (one dimension per button), resulting in \(2^{12} = 4096\) combinations. Many of these combinations are invalid or redundant.

### Solution
Reduce the action space by defining meaningful button combinations and mapping them to a **Discrete** space:

```python
import gym
import numpy as np

class PlayerOneNetworkControllerWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(PlayerOneNetworkControllerWrapper, self).__init__(env)
        
        # Define buttons
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        
        # Define valid actions (button combinations)
        actions = [
            ["LEFT"], ["RIGHT"], ["DOWN", "B"], ["UP", "A"], ["LEFT", "DOWN"], ["RIGHT", "UP", "B"],
            ["C"], ["START"], ["A", "B"],
        ]
        
        # Map actions to binary arrays
        self._actions = []
        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        
        # Redefine action space as Discrete
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, action):
        # Map discrete action index to button combination
        return self._actions[action].copy()
```

### How It Works
- The wrapper reduces the 4096 possible combinations to a small, meaningful set of valid actions.
- Converts the **MultiDiscrete** space into a **Discrete** space for efficient learning.

---

## **6. Summary**
- **Discrete**: Use for tasks with a single action per step.
- **MultiDiscrete**: Use for tasks with multiple simultaneous actions.
- For complex environments, you can simplify the action space by creating a custom wrapper to map valid button combinations to a **Discrete** space.

This approach improves learning efficiency and makes it easier to train agents for games like Mortal Kombat.

---

Keep this document handy for quick reference while working with action spaces in Gym-based environments!


# Visualizing the Flow

+------------------+          +------------------+          +--------------------+
|  Agent's Action  |   --->   |  Action Mapping  |   --->   |  Game Environment  |
| (Discrete Index) |          | (Binary Array)   |          | (Button Presses)   |
+------------------+          +------------------+          +--------------------+
|        0         |          | [0, 0, 0, ...]  |          | 'LEFT'             |
|        1         |          | [0, 0, 0, ...]  |          | 'RIGHT' + 'DOWN'   |
|        2         |          | [0, 0, 0, ...]  |          | 'UP' + 'B'         |
+------------------+          +------------------+          +--------------------+


+------------------+          +---------------------------+          +-----------------------+
|  Agent's Action  |   --->   |      Action Mapping       |   --->   |   Game Environment    |
| (Discrete Index) |          |    (Binary Array)         |          |    (Button Presses)   |
+------------------+          +---------------------------+          +-----------------------+
|        0         |          | [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] |         'LEFT'         |
|        1         |          | [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] |        'RIGHT'         |
|        2         |          | [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0] |    'LEFT' + 'DOWN'     |
|        3         |          | [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0] |   'RIGHT' + 'DOWN'     |
|        4         |          | [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] |      'UP' + 'B'        |
|        5         |          | [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] |     'DOWN' + 'A'       |
|        6         |          | [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] |          'C'           |
|        7         |          | [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] |        'START'         |
+------------------+          +---------------------------+          +-----------------------+


