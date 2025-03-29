# Action Space in Kane vs Abel Mortal Kombat II

This document explains how the action system works in the Mortal Kombat II environment, including action generation, mapping, and how different agents interface with it.

## Action Space Representations

The action system uses three core representations that are translated between different components:

1. **Button Names**: Human-readable commands (`LEFT`, `RIGHT`, `A`, `B`, etc.)
2. **Binary Arrays**: Stable-Retro's internal format (`[0,0,1,0,0,0,0,0]`)
3. **Action IDs**: Integer indices used to reference specific actions (`0`, `1`, `2`, etc.)

## Action Generator

The `ActionGenerator` class creates and manages mappings between these representations:

```python
from mk_ai.utils import ActionGenerator

# Load from YAML configuration
action_gen = ActionGenerator(filename="env_config.yaml")
action_gen.build()  # Must be called explicitly to create mappings

# Access the generated mappings
action_map = action_gen.action_map  # Maps names to IDs: {"LEFT": 1, "RIGHT": 2}
```

### YAML Configuration

The env_config.yaml file defines the buttons and valid action combinations:

```yaml
buttons:
  - B
  - A
  - MODE
  - START
  - UP
  - DOWN
  - LEFT
  - RIGHT
  - C
  - Y
  - X
  - Z

actions:
  - []  # Neutral
  - [LEFT]
  - [RIGHT]
  - [LEFT, DOWN]
  - [RIGHT, DOWN]
  # ...more combinations
```

### Generated Mappings

The `ActionGenerator` creates several key mappings:

```python
# Button list - individual inputs
buttons = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']

# Action combinations with auto-generated names
action_map = {
    'NEUTRAL': 0,
    'LEFT': 1,
    'RIGHT': 2,
    'LEFT_DOWN': 3,
    'RIGHT_DOWN': 4,
    # ...
}

# Binary representation for each action
binary_mapping = {
    0: [0,0,0,0,0,0,0,0,0,0,0,0],  # NEUTRAL
    1: [0,0,0,0,0,0,1,0,0,0,0,0],  # LEFT
    2: [0,0,0,0,0,0,0,1,0,0,0,0],  # RIGHT
    # ...
}
```

## Agent Action Processing

### RL Agents (DQN, Double DQN, Dueling DQN)

RL agents directly output action IDs:

```python
# Agent internally selects action from discrete space
action_id = agent.select_action(observation, info)  # Returns integer ID

# MkEnvWrapper handles conversion to binary format for Stable-Retro
binary_action = env_wrapper.action_mapping[action_id]  # e.g., [0,0,1,0,0,0,0,0]
```

### Behavior Tree Agents

BT agents process actions through multiple steps:

1. **YAML Definition**: Actions are defined with string names in the BT file:
   ```yaml
   - type: Action
     name: "Jump Attack"
     properties:
       action_id: UP_A  # String name matching action_map keys
       frames_needed: 5
   ```

2. **Conversion in BTLoader**: String names are converted to action IDs during BT construction:
   ```python
   # In BTLoader.gen_node():
   action_id_name = properties.get("action_id")  # "UP_A" from YAML
   action_id = self.action_map.get(action_id_name, 0)  # Maps to integer ID
   ```

3. **Action Selection**: The BT returns the action ID of the active node:
   ```python
   # In BTAgent.tick():
   status = self.bt_root.tick(self.context)
   action_id = self.bt_root.get_current_action()
   return action_id  # Returns integer for environment
   ```

4. **Binary Conversion**: The environment wrapper converts the ID to binary format

### Human Agents

Human agents convert keyboard inputs to action IDs:

```python
# 1. Key pressed → button name via key_map
button_name = self.key_map.get(key)  # e.g., LEFT, RIGHT

# 2. Build binary array from pressed buttons
binary_array = [0] * len(self.buttons)
for button in pressed_buttons:
    if button in self.buttons:
        idx = self.buttons.index(button)
        binary_array[idx] = 1

# 3. Find matching action ID from the binary array
for action_id, action_binary in enumerate(self.action_mapping):
    if action_binary == binary_array:
        return action_id
```

## Complete Action Flow

```
╔════════════════════════╗                   ╔════════════════════╗
║     Agent Selection     ║                   ║  Environment Input ║
╠════════════════════════╣                   ╠════════════════════╣
║                        ║                   ║                    ║
║  ┌─────────────────┐   ║                   ║  ┌──────────────┐  ║
║  │ Human:          │   ║                   ║  │              │  ║
║  │ Keyboard Input  │──>║───┐               ║  │ Stable-Retro │  ║
║  └─────────────────┘   ║   │               ║  │ Environment  │  ║
║                        ║   │               ║  │              │  ║
║  ┌─────────────────┐   ║   │  ┌────────┐   ║  │              │  ║
║  │ RL:             │   ║   └─>│        │   ║  │              │  ║
║  │ Neural Network  │──>║────>│ Action │──>║──>│              │  ║
║  └─────────────────┘   ║   ┌─>│   ID   │   ║  │              │  ║
║                        ║   │  │        │   ║  │              │  ║
║  ┌─────────────────┐   ║   │  └────────┘   ║  │              │  ║
║  │ BT:             │   ║   │               ║  │              │  ║
║  │ Decision Tree   │──>║───┘               ║  │              │  ║
║  └─────────────────┘   ║                   ║  └──────────────┘  ║
║                        ║                   ║                    ║
╚════════════════════════╝                   ╚════════════════════╝
           ▲                                           ▲
           │                                           │
           │                                           │
           │       ┌─────────────────────────┐        │
           │       │                         │        │
           └───────┤    Action Conversion    ├────────┘
                   │    ID <=> Binary        │
                   │                         │
                   └─────────────────────────┘
```

## Visualizing the Flow

```
+------------------+          +---------------------------+          +-----------------------+
|  Agent's Action  |   --->   |      Action Mapping       |   --->   |   Game Environment    |
| (Discrete Index) |          |    (Binary Array)         |          |    (Button Presses)   |
+------------------+          +---------------------------+          +-----------------------+
|        0         |          | [0, 0, 0, 0, 0, 0, 0, 0]  |          |        NEUTRAL        |
|        1         |          | [0, 0, 0, 0, 0, 0, 0, 0]  |          |         'LEFT'        |
|        2         |          | [0, 0, 0, 0, 0, 0, 0, 0]  |          |        'RIGHT'        |
|        3         |          | [0, 0, 0, 0, 0, 0, 0, 0]  |          |    'LEFT' + 'DOWN'    |
|        4         |          | [0, 0, 0, 0, 0, 0, 1, 0]  |          |    'RIGHT' + 'DOWN'   |
+------------------+          +---------------------------+          +-----------------------+
```

## Implementation Details

### Shared Action Space

All agents share the same action space defined in env_config.yaml. This ensures:

- Consistent behavior across agent types
- Fair comparisons between different agents
- Compatibility with the Stable-Retro environment

### Adding Custom Actions

To add a new action:

1. Add the button combination to env_config.yaml:
   ```yaml
   actions:
     # Existing actions...
     - [DOWN]  # Add standalone crouch action
   ```

2. Rebuild the action mappings and update affected agents.

## Debugging Tips

- Check the `action_map` to see available actions: `print(action_gen.action_map)`
- Ensure BT YAML files use action names that match the keys in `action_map`
- If actions aren't working, verify they exist in env_config.yaml
- Use `print(binary_mapping[action_id])` to see the binary representation

## Related Files

- `mk_ai/utils/action_generator.py`: Contains ActionGenerator class
- `mk_ai/configs/env_config.yaml`: Defines buttons and valid actions
- `mk_ai/agents/bt_agent.py`: BT agent implementation
- `mk_ai/agents/BT/loader.py`: Loads and processes BT actions
