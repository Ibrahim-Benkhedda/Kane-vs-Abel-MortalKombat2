This project is still on a working progress.

## Packages 
- Stable-Retro: Provides the environment (e.g., the game).
- Gymnasium: Acts as the interface for interacting with the environment (reset, step, render, etc.).
- Stable-Baselines3: Provides the RL algorithms to train an agent in the environment.


## Mortal Kombat II: Genesis Action Space 
 Since Mortal Kombat uses a 12-button controller, and each button can either be pressed (1) or not pressed (0), the total number of possible combinations of button states is 4096

## Example: Kane (Double DQN) vs Very Easy Opponent

The following example shows **Kane**, using Double DQN architecture playing with **Liu Kang** (Left character), competing against a very easy CPU opponent in **Mortal Kombat II (Genesis version)**.

![Kane vs Very Easy Opponent](replays/output.gif) ![Kane vs Very Easy Opponent](replays/LiuKang-Jax-VeryEasy.gif)



