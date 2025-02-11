# Overview
The project is about two AIagents: **Abel** (Behaviour Tree) and **Kane** (Dueling Double DQN) competing against each other in Mortal Kombat II: Genesis using Gymnasium and stable-retro library.
# Example: Kane (Double DQN) vs Very Easy Opponent
The following example shows **Kane**, using Double DQN architecture playing with **Liu Kang** (Left character), competing against a very easy CPU opponent in **Mortal Kombat II (Genesis version)**.

![Kane vs Very Easy Opponent](data/replays/LiuKang-VeryEasy.gif) ![Kane vs Very Easy Opponent](data/replays/LiuKang-Jax-VeryEasy.gif)


# Packages 
This project is developed and tested using Python **3.10.12** on **WSL Ubuntu**. The project depends on the following packages:
- **Stable Baselines 3 (SB3):** `stable_baselines3==2.4.0`
- **Stable Retro (Retro):** `stable-retro==0.9.2`
- **Gymnasium:** `gymnasium==1.0.0`
- **Optuna:** `optuna==4.1.0` (For hyperparameters search)
- **PyTorch (torch):** `torch==2.5.1`
- **TensorBoard:** `tensorboard==2.18.0`

**Note for WSL Users:**  
When running GUI applications under WSL, an X server is required to display graphical interfaces. We recommend using **XLaunch**, a tool that helps configure and launch an X server (such as VcXsrv) on Windows, allowing you to run Linux GUI applications within your WSL environment.

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

# Usage 
This section describes how to train, test, and run the arena to evaluate different agents in the Mortal Kombat II environment.

## Training RL Agents

To train a reinforcement learning (RL) agent, run the following command:

```bash
python train.py
```

## Testing/Evaluating RL Agents

To evaluate against CPU opponents a pre-trained RL agent, use the test.py script. For example, to test a DQN-based agent:
```bash
python test.py \
    --model_path "models/kane/DoubleDQN_without_curriculum_4M_VeryEasyVsJax_Ln.zip" \
    --model_type DUELINGDDQN \
    --state "Level1.LiuKangVsJax"
```
## Running the Arena for Agent Comparison
The `arena.py` script allows you to evaluate different agents (current support: human input, DQNs and Behaviour Trees Agents) against each other in real-time. For example, to have a human player compete against a DQN agent, run:

```bash
python arena.py \
    --p1-type human \
    --p2-type double_dqn \
    --p2-model "models/kane/DoubleDQN_without_curriculum_4M_VeryEasyVsJax_Ln.zip"
```


