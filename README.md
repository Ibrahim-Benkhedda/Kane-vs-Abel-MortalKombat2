# Overview

This repository contains the code for training an agent to play Mortal Kombat II using reinforcement learning.

# Packages 
- Stable-Retro: Provides the environment (e.g., the game).
- Gymnasium: Acts as the interface for interacting with the environment (reset, step, render, etc.).
- Stable-Baselines3: Provides the RL algorithms to train an agent in the environment.

# Example: Kane (Double DQN) vs Very Easy Opponent
The following example shows **Kane**, using Double DQN architecture playing with **Liu Kang** (Left character), competing against a very easy CPU opponent in **Mortal Kombat II (Genesis version)**.
![Kane vs Very Easy Opponent](data/replays/LiuKang-VeryEasy.gif) ![Kane vs Very Easy Opponent](data/replays/LiuKang-Jax-VeryEasy.gif)

# Folder Structure Overview

This repository is organized to clearly separate its various components, making it easy to develop, test, and extend. Below is an explanation of the key directories and files included in the project:

## Folder Structure
```
.
├── arena.py                 # interface: integrates game logic, rendering, and input handling for evaluating agents against each other.
├── docs/                    # Documentation files
├── experiments/             # TensorBoard logs and experiment configurations for RL training runs
├── models/                  # Pre-trained models and BT config files
│   ├── kane/                # Learning-based agents  DQN, Double DQN, Dueling DDQN models)
│   └── abel/                # Behaviour tree configs for BT agents in YAML format
├── notebooks/               # Jupyter notebooks and reports for evaluation, analysis, and hyperparameter optimization
├── scripts/                 # Utility scripts for hyperparam search, fine-tuning, and match simulations
├── src/mk_ai/               # Core source code of the project
│   ├── agents/              # Agent implementations:
│   │   ├── BT/              # Behaviour tree implementation (BT logic and YAML configurations)
│   │   └── DQN/             # Deep Q-Network variants implementation (DQN, Double DQN, Dueling DDQN implementations)
│   ├── core/                # Game environment logic (arena configuration, input handling, rendering)
│   ├── configs/             # Configuration files for the environment and agents
│   ├── callbacks/           # Training callbacks (curriculum learning, custom evaluations)
│   ├── utils/               # Helper functions and utility modules
│   └── wrappers/            # Gymnasium wrappers for the Mortal Kombat II Genesis environment
├── train.py                 # Script for training RL agents in Mortal Kombat II
├── test.py                  # Script for evaluating RL agents in Mortal Kombat II
└── tests/                   # Unit and integration tests for various modules and functionalities
```


