# Human Agent Controls and Input Customization

This document explains how the human agent works in Kane vs Abel Mortal Kombat II, how keyboard input is processed, and how to customize or extend the input system.

## Human Agent Overview

The `HumanAgent` class provides an interface for human players to participate in matches through keyboard inputs. It translates key presses into game actions compatible with the Stable-Retro environment.

## Default Controls

### Player 1 Controls
| Action | Default Key |
|--------|-------------|
| **Movement** |  |
| UP | ↑ (Up Arrow) |
| DOWN | ↓ (Down Arrow) |
| LEFT | ← (Left Arrow) |
| RIGHT | → (Right Arrow) |
| **Attacks** |  |
| A | Z |
| B | X |
| C | C |
| START | Enter |
| MODE | Tab |

### Player 2 Controls
| Action | Default Key |
|--------|-------------|
| **Movement** |  |
| UP | W |
| DOWN | S |
| LEFT | A |
| RIGHT | D |
| **Attacks** |  |
| A | T |
| B | Y |
| C | U |
| START | Enter |
| MODE | Tab |

## Input Processing Pipeline

The input system follows this flow:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Key Press      │     │  InputHandler   │     │  HumanAgent     │     │  Game           │
│  (Keyboard)     │────>│  (Key Tracking) │────>│  (Action        │────>│  Environment    │
└─────────────────┘     └─────────────────┘     │  Conversion)    │     └─────────────────┘
                                                └─────────────────┘
```

1. Key presses are detected by pyglet's event system
2. `InputHandler` maintains sets of pressed keys for each player
3. `HumanAgent` converts key presses to binary action arrays
4. Actions are passed to the game environment

## How Input Works

### Input Detection

The `InputHandler` class listens for key events on the game window:

```python
def on_key_press(self, symbol, modifiers):
    # Track which player's key was pressed
    if symbol in P1_KEY_MAP:
        self.p1_pressed.add(symbol)
    elif symbol in P2_KEY_MAP:
        self.p2_pressed.add(symbol)
```

### Key-to-Action Conversion

The `HumanAgent` converts tracked key presses to game actions:

```python
def select_action(self, obs, info) -> int:
    # Convert pressed keys to binary action array
    action_array = [0] * len(self.buttons)
    for key in self.pressed_keys:
        if key in self.key_map:
            button = self.key_map[key]
            if button in self.buttons:
                action_array[self.buttons.index(button)] = 1
    
    # Find matching action ID
    for action_id, arr in enumerate(self.action_mapping):
        if arr == action_array:
            return action_id
    return 0  # Return NEUTRAL if no match
```

## Customizing Key Mappings

You can modify key mappings by editing the key map constants in the project:

```python
# In mk_ai/configs/__init__.py or similar file
from pyglet.window import key as keycodes

P1_KEY_MAP = {
    keycodes.UP: "UP",
    keycodes.DOWN: "DOWN",
    keycodes.LEFT: "LEFT",
    keycodes.RIGHT: "RIGHT",
    keycodes.Z: "A",
    keycodes.X: "B",
    keycodes.C: "C",
    # Add custom mappings here
}

P2_KEY_MAP = {
    keycodes.W: "UP",
    keycodes.S: "DOWN", 
    keycodes.A: "LEFT",
    keycodes.D: "RIGHT",
    keycodes.T: "A",
    keycodes.Y: "B",
    keycodes.U: "C",
    # Add custom mappings here
}
```

## Adding New Input Methods

### Gamepad Support

To add gamepad support:

1. Create a gamepad handler class that inherits from `Agent`:

```python
class GamepadAgent(Agent):
    def __init__(self, action_mapping, buttons, gamepad_id=0):
        self.action_mapping = action_mapping
        self.buttons = buttons
        self.gamepad_id = gamepad_id
        
        # Initialize gamepad library (e.g., using pyglet or pygame)
        self.gamepad = self.initialize_gamepad(gamepad_id)
    
    def initialize_gamepad(self, id):
        # Implementation depends on the gamepad library you're using
        pass
        
    def select_action(self, obs, info):
        # Read gamepad state
        button_states = self.read_gamepad_state()
        
        # Convert to action array similar to HumanAgent
        action_array = [0] * len(self.buttons)
        for button_name, is_pressed in button_states.items():
            if is_pressed and button_name in self.buttons:
                action_array[self.buttons.index(button_name)] = 1
        
        # Find matching action ID
        for action_id, arr in enumerate(self.action_mapping):
            if arr == action_array:
                return action_id
        return 0
```

2. Update `AgentFactory` to support the new agent type:

```python
elif config.agent_type == "gamepad":
    return GamepadAgent(
        action_mapping=action_mapping,
        buttons=buttons,
        gamepad_id=config.gamepad_id
    )
```

### Network Input (Online Play)

For networked multiplayer:

1. Create a `NetworkAgent` class that receives inputs from a remote player
2. Implement a simple socket server to relay player inputs
3. Update the `AgentFactory` to support the new agent type

## Troubleshooting Input Issues

### Common Issues and Solutions

1. **Keys not responding:**
   - Check that the key mapping exists in `P1_KEY_MAP` or `P2_KEY_MAP`
   - Verify the window has focus when keys are pressed

2. **Multiple keys not working together:**
   - Some keyboards have limited key rollover. Try pressing fewer keys simultaneously
   - Check if the key combination exists in env_config.yaml

3. **Key stuck on:**
   - Ensure `on_key_release` is being called properly
   - Add manual cleanup code to reset keys when losing window focus

## Example: Adding WASD Controls for Player 1

```python
from pyglet.window import key as keycodes

# Modified key mapping that uses WASD for P1 movement
P1_KEY_MAP = {
    # WASD movement
    keycodes.W: "UP",
    keycodes.A: "LEFT",
    keycodes.S: "DOWN",
    keycodes.D: "RIGHT",
    # Attack buttons
    keycodes.J: "A",
    keycodes.K: "B",
    keycodes.L: "C",
    keycodes.ENTER: "START",
    keycodes.SPACE: "MODE",
}
```

## Advanced: Creating Custom Input Handlers

For specialized input devices or complex control schemes, you can create custom input handlers:

1. Create a new class that inherits from `InputHandler`
2. Override the key event methods to support your devices
3. Update your custom `HumanAgent` to work with your handler

```python
class CustomInputHandler(InputHandler):
    def __init__(self, window):
        super().__init__(window)
        # Initialize your custom input devices
        
    def setup_window_events(self):
        super().setup_window_events()
        # Add custom event handlers
        
    def update(self):
        # Poll custom input devices and update key states
        pass
```

Then modify your game loop to use this custom handler.

Similar code found with 1 license type