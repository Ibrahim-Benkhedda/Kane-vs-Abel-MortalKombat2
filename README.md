## Overview

This repository contains the code for training an agent to play Mortal Kombat II using reinforcement learning.

## Packages 
- Stable-Retro: Provides the environment (e.g., the game).
- Gymnasium: Acts as the interface for interacting with the environment (reset, step, render, etc.).
- Stable-Baselines3: Provides the RL algorithms to train an agent in the environment.

## Example: Kane (Double DQN) vs Very Easy Opponent
The following example shows **Kane**, using Double DQN architecture playing with **Liu Kang** (Left character), competing against a very easy CPU opponent in **Mortal Kombat II (Genesis version)**.
![Kane vs Very Easy Opponent](data/replays/LiuKang-VeryEasy.gif) ![Kane vs Very Easy Opponent](data/replays/LiuKang-Jax-VeryEasy.gif)
