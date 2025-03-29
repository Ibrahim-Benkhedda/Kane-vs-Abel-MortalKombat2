# Adding New Games to the Kane vs Abel Arena Framework

This guide outlines the workflow for integrating new Stable-Retro compatible games into the Arena framework.

## Overview

The Arena framework was designed for Mortal Kombat II but can be extended to other 2-player games. This document outlines the key steps and files to modify for successful integration.

## Integration Workflow

### 1. Import Game ROM into Stable-Retro

```bash
# Import your ROM
python -m retro.import /path/to/game.md
```

Verify import:
```bash
# Verify game appears in list
python -c "import retro; print([g for g in retro.data.list_games() if 'YourGame' in g])"
```

### 2. Create Game States

- Launch the game with Stable-Retro UI: `python -m retro.ui.play --game YourGame-SystemName`
- Press F9 at meaningful starting points (character select, match start)
- States are saved in `~/.local/share/stable-retro/data/YourGame-SystemName/states/`

### 3. Define Game Memory Mapping

Create or modify:
- `~/.local/share/stable-retro/data/YourGame-SystemName/data.json`

Key variables to map:
- Player health
- Enemy health
- Player position
- Enemy position
- Round counters
- Any game-specific state information

### 4. Configure Button Layout

Create a new config:
- `src/mk_ai/configs/your_game_config.yaml`

Define:
- Available buttons
- Valid button combinations/actions

### 5. Create Game-Specific Environment Wrapper

Modify or create:
- `src/mk_ai/wrappers/your_game_env.py`

Implement:
- Game-specific observation processing
- Custom reward functions
- State tracking relevant to your game

### 6. Extend MultiAgent Support

Modify or create:
- `src/mk_ai/wrappers/multiagent_your_game.py` 

Implement:
- Proper handling of two-player actions
- Player-specific observation processing
- Game-specific multiagent interactions

### 7. Update Arena Configuration

Modify:
- arena_config.py

Add:
- Support for your game in the config class
- Game state selection options

### 8. Update Command Line Interface

Modify:
- arena.py

Add:
- Game selection parameters
- State selection parameters
- Game-specific options

## Testing Your Integration

1. Start with basic functionality:
```bash
python arena.py --game YourGame-SystemName --state YourState
```

2. Test human controls:
```bash
python arena.py --game YourGame-SystemName --p1-type human --p2-type human
```

3. Test with AI agents:
```bash
python arena.py --game YourGame-SystemName --p1-type bt --p1-bt-file your_bt.yaml
```

## Troubleshooting Tips

- **Memory addresses incorrect**: Use RAM watch in Stable-Retro UI to find correct addresses
- **Game not launching**: Check ROM import and state file existence
- **Actions not working**: Verify button configuration and action mapping
- **Agent not responding**: Ensure observation processing works properly

## Key Files Reference

| File | Purpose | What to Modify |
|------|---------|----------------|
| env_config.yaml | Button/action definitions | Add game-specific actions |
| `mk_env.py` | Base environment wrapper | Extend for game-specific behavior |
| `multiagent_mk_env.py` | Multiagent support | Extend for 2-player support |
| `arena_config.py` | Framework configuration | Add game to available options |
| arena.py | Main application | Add game-specific CLI options |

## Example Workflow

For a new fighting game:

1. Import ROM and create starting state at match beginning
2. Map health, position and round variables in data.json
3. Define basic moves in a new config YAML file
4. Create environment wrapper handling game-specific rewards
5. Extend multiagent support for 2-player mode
6. Update arena configuration and CLI
7. Test with human controls first, then with agents

By following this workflow, you can integrate most Stable-Retro compatible games into the Arena framework for AI research and evaluation.