# Creating Custom Behavior Trees

This document explains how to create your own behavior trees for AI agents in the Kane vs Abel Mortal Kombat II environment.

## Introduction to Behavior Trees

Behavior Trees (BTs) are a hierarchical structure of nodes that control decision making for AI agents. They offer several advantages:

- Modular, reusable behavior components
- Visual representation of decision logic
- Readable and maintainable AI designs
- Effective for implementing complex game AI

## Behavior Tree Structure

A behavior tree is defined in YAML format and consists of different node types arranged in a hierarchical structure:

```yaml
node:
  type: Selector  # Root node type
  name: "Root"    # Node name for debugging
  children:       # Child nodes
    - type: Sequence
      name: "Attack Sequence"
      children:
        # More nodes...
```

## Node Types

### Selector Node

Executes children from left to right until one succeeds. If all fail, the selector fails.

```yaml
type: Selector
name: "Choose Attack"
children:
  - type: Action
    name: "High Attack"
    properties:
      action_id: A
      frames_needed: 3
  - type: Action
    name: "Low Attack"
    properties:
      action_id: DOWN_A
      frames_needed: 3
```

### Sequence Node

Executes children from left to right until all succeed or one fails. If any fail, the sequence fails.

```yaml
type: Sequence
name: "Jump and Attack"
children:
  - type: Action
    name: "Jump"
    properties:
      action_id: UP
      frames_needed: 2
  - type: Action
    name: "Attack"
    properties:
      action_id: A
      frames_needed: 1
```

### Condition Node

Tests a condition and returns success or failure based on the result.

```yaml
type: Condition
name: "Is Enemy Close"
properties:
  condition: "is_close_to_enemy"  # Must match a function in ConditionsProvider
```

### Action Node

Performs a game action for a specified number of frames.

```yaml
type: Action
name: "Block"
properties:
  action_id: B        # Must match an action in action_map
  frames_needed: 5    # Number of frames to hold the action
```

## Creating Your BT File

### File Structure

Create a YAML file with a single root node:

```yaml
node:
  type: Selector
  name: "Fighter Behavior"
  children:
    # Add your behavior tree nodes here
```

### Available Conditions

The following conditions are available from `ConditionsProvider`:

| Condition Name | Function | Description |
|----------------|----------|-------------|
| `is_enemy_to_the_right` | Returns true when the enemy is to the right of the player | Used for directional awareness |
| `is_enemy_to_the_left` | Returns true when the enemy is to the left of the player | Used for directional awareness |
| `is_close_to_enemy` | Returns true when the enemy is within 50 units | Used for close-range actions |
| `is_long_range_enemy` | Returns true when the enemy is beyond 50 units | Used for ranged attacks |
| `is_medium_range_enemy` | Returns true when the enemy is between 50-120 units | Good for jump-in attacks |

### Available Actions

Actions are defined in env_config.yaml and mapped to names in the ActionGenerator. Common actions include:

| Action Name | Description | Button Combination |
|-------------|-------------|-------------------|
| `NEUTRAL` | No buttons pressed | None |
| `LEFT` | Move left | LEFT |
| `RIGHT` | Move right | RIGHT |
| `UP` | Jump | UP |
| `DOWN` | Crouch | DOWN |
| `A` | A button | A |
| `B` | B button | B |
| `LEFT_DOWN` | Crouch + move left | LEFT+DOWN |
| `RIGHT_DOWN` | Crouch + move right | RIGHT+DOWN |
| `UP_A` | Jump + A | UP+A |
| `DOWN_B` | Crouch + B | DOWN+B |

## Complete Example: Aggressive Fighter

```yaml
node:
  type: Selector
  name: "Aggressive Fighter"
  children:
    # Close Range Attack
    - type: Sequence
      name: "Close Range Attack"
      children:
        - type: Condition
          name: "Enemy is Close"
          properties:
            condition: "is_close_to_enemy"
        - type: Selector
          name: "Choose Close Attack"
          children:
            - type: Action
              name: "Leg Sweep"
              properties:
                action_id: DOWN_B
                frames_needed: 3
            - type: Action
              name: "Low Punch"
              properties:
                action_id: DOWN_A
                frames_needed: 3
                
    # Medium Range Jump-In
    - type: Sequence
      name: "Jump Attack Approach"
      children:
        - type: Condition
          name: "Enemy is Medium Range"
          properties:
            condition: "is_medium_range_enemy"
        - type: Selector
          name: "Jump Direction"
          children:
            - type: Sequence
              name: "Jump Right Attack"
              children:
                - type: Condition
                  name: "Enemy to Right"
                  properties:
                    condition: "is_enemy_to_the_right"
                - type: Action
                  name: "Jump Forward Attack"
                  properties:
                    action_id: RIGHT_UP_A
                    frames_needed: 5
            - type: Sequence
              name: "Jump Left Attack"
              children:
                - type: Condition
                  name: "Enemy to Left"
                  properties:
                    condition: "is_enemy_to_the_left"
                - type: Action
                  name: "Jump Forward Attack"
                  properties:
                    action_id: LEFT_UP_A
                    frames_needed: 5
    
    # Long Range Fireball
    - type: Sequence
      name: "Long Range Attack"
      children:
        - type: Condition
          name: "Enemy is Far"
          properties:
            condition: "is_long_range_enemy"
        - type: Action
          name: "Fireball Special"
          properties:
            action_id: RIGHT_UP_B
            frames_needed: 7
            
    # Approach - Movement
    - type: Selector
      name: "Movement"
      children:
        - type: Sequence
          name: "Move Right"
          children:
            - type: Condition
              name: "Enemy to Right"
              properties:
                condition: "is_enemy_to_the_right"
            - type: Action
              name: "Move Right"
              properties:
                action_id: RIGHT
                frames_needed: 3
        - type: Sequence
          name: "Move Left"
          children:
            - type: Condition
              name: "Enemy to Left"
              properties:
                condition: "is_enemy_to_the_left"
            - type: Action
              name: "Move Left"
              properties:
                action_id: LEFT
                frames_needed: 3
```

## Best Practices

1. **Prioritize actions properly**: Place the most important behaviors at the top of selectors
2. **Balance reactivity**: Don't set `frames_needed` too high or your agent will be unresponsive
3. **Test incrementally**: Start with a simple BT and gradually add complexity
4. **Use meaningful node names**: They help when debugging behaviors
5. **Consider opponent position**: Always check enemy position before directional actions
6. **Add fallback behaviors**: Ensure your tree has default actions if no conditions match

## Debugging BTs

1. **Check action names**: Ensure they match exactly with keys in `action_map`
2. **Verify conditions**: Make sure condition names match functions in `ConditionsProvider`
3. **Start simple**: Begin with a few nodes and gradually expand
4. **Test against stationary opponents** first to verify behavior
5. **Watch for indentation errors** in YAML that may break the tree structure

## Using Your Custom BT

Run the arena with your custom BT file:

```bash
python arena.py --p1-type bt --p1-bt-file path/to/your_bt.yaml
```

Or for player 2:

```bash
python arena.py --p2-type bt --p2-bt-file path/to/your_bt.yaml
```

## Further Customization

To add custom conditions:
1. Add your function to the `ConditionsProvider` class in `conditions.py`
2. Use your new condition name in the BT YAML file