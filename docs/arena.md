# Kane vs Abel Mortal Kombat II Arena

## Overview

The Mortal Kombat II Arena is a framework for training, evaluating, and comparing different AI agents in the Mortal Kombat II environment. It supports various agent types including human players, DQN-based reinforcement learning agents, and behavior tree agents.

## System Architecture

The arena implements a Model-View-Controller (MVC) architecture:

1. **Model (`EnvModel`)**: Manages the game environment state and interactions
2. **View (`Renderer`)**: Handles visualization of the game state
3. **Controller (`MortalKombatArena`)**: Coordinates inputs, agent actions, and game updates
4. **Input Handler (`InputHandler`)**: Processes keyboard inputs for human players

## Agent Types

The arena supports the following agent types:

- **Human**: Controlled via keyboard inputs
- **DQN**: Deep Q-Network reinforcement learning agent
- **Double DQN**: Double Deep Q-Network for more stable learning
- **Dueling DDQN**: Dueling architecture with Double DQN for better value estimation
- **BT**: Behavior Tree-based agents that use predefined decision trees

## Command Line Interface

Run the arena using the following command structure:

```bash
python arena.py [options]
```

### Common Options

| Option | Description |
|--------|-------------|
| `--p1-type {human,dqn,double_dqn,dueling_ddqn,bt}` | Agent type for Player 1 |
| `--p1-model PATH` | Path to the model file for Player 1 (for RL agents) |
| `--p1-bt-file PATH` | Path to behavior tree YAML file for Player 1 (for BT agents) |
| `--p1-username NAME` | Username for Player 1 (for human players) |
| `--p2-type {human,dqn,double_dqn,dueling_ddqn,bt}` | Agent type for Player 2 |
| `--p2-model PATH` | Path to the model file for Player 2 (for RL agents) |
| `--p2-bt-file PATH` | Path to behavior tree YAML file for Player 2 (for BT agents) |
| `--p2-username NAME` | Username for Player 2 (for human players) |
| `--window-size WIDTH HEIGHT` | Window dimensions (default: 640 480) |
| `--fps FPS` | Target frames per second (default: 60) |
| `--switch` | Swap Player 1 and Player 2 positions |

## Configuration

### Agent Configuration

Each agent is configured through an `AgentConfig` object with these properties:
- `agent_type`: The type of agent ("human", "dqn", "double_dqn", "dueling_ddqn", "bt")
- `model_path`: Path to the model file (for RL agents)
- `bt_file_path`: Path to the behavior tree file (for BT agents)
- `player_num`: Player number (1 or 2)

### Arena Configuration

The arena is configured through an `ArenaConfig` object with these properties:
- `game`: Game name ("MortalKombatII-Genesis")
- `state`: Initial game state ("Level1.LiuKangVsJax.2P")
- `players`: Number of players (2)
- `window_size`: Window dimensions for rendering
- `fps`: Target frames per second
- `p1_agent`: Agent configuration for Player 1
- `p2_agent`: Agent configuration for Player 2

## ELO Rating System

The arena includes an ELO rating system to track and compare agent performance:

- Each agent type has its own rating tracked in `elo_ratings.json`
- Ratings are updated after each match based on game outcome
- Default starting rating is 1500
- Human players are tracked by their usernames
- AI agents are tracked by their uppercase agent types

## Behavior Trees

BT agents use YAML files to define their decision trees:

1. Each BT file defines a tree of nodes (Selectors, Sequences, Conditions, Actions)
2. Conditions map to functions in the `ConditionsProvider` class
3. Actions map to game inputs defined in `env_config.yaml`

### Example BT Structure

```yaml
node:
  type: Selector
  name: "Root"
  children:
    - type: Sequence
      name: "Attack Sequence"
      children:
        - type: Condition
          name: "Is Close"
          properties:
            condition: "is_close_to_enemy"
        - type: Action
          name: "Attack"
          properties:
            action_id: "A"
            frames_needed: 5
```

## Examples

### Human vs. Double DQN

```bash
python arena.py \
    --p1-type human \
    --p2-type double_dqn \
    --p2-model "models/kane/DDQN_4M_with_lrSchedule_frameskip_24_actions.zip"
```

### Dueling DQN vs. Double DQN

```bash
python arena.py \
    --p1-type dueling_ddqn \
    --p1-model "models/DuelingDQN_without_curriculum_4M_VeryEasyVsJax" \
    --p2-type double_dqn \
    --p2-model "models/DoubleDQN_without_curriculum_4M_VeryEasyVsJax_Exp_A"
```

### BT vs. BT with Custom Trees

```bash
python arena.py \
    --p1-type bt \
    --p1-bt-file "bt_files/aggressive.yaml" \
    --p2-type bt \
    --p2-bt-file "bt_files/defensive.yaml"
```

### Switching Player Positions

```bash
python arena.py \
    --p1-type human \
    --p2-type double_dqn \
    --p2-model "models/kane/model.zip" \
    --switch
```

## Controls

### Player 1 Controls
- **Arrow Keys**: Movement
- **Z**: A button
- **X**: B button
- **C**: C button
- **Enter**: Start button

### Player 2 Controls
- **W, A, S, D**: Movement
- **T**: A button
- **Y**: B button
- **U**: C button
- **Enter**: Start button