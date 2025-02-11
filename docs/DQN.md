# DQN Training Loop Steps

## Initialize Environment and Replay Buffer:

The agent interacts with the environment to gather experience tuples 
(𝑠,𝑎,𝑟,𝑠′,𝑑) (s,a,r,s′,d), where 𝑑 is a done flag indicating the end of an episode. These are stored in a replay buffer.

## Initialize Q-Networks:

- A policy network (for Q-value estimation).
- A target network (a copy of the policy network, updated periodically for stability).

## Iterate Over Episodes:

### For each episode:
1. **Reset Environment**: Start with an initial state 𝑠.
2. **Choose Action**: Use an epsilon-greedy policy to balance exploration (random actions) and exploitation (actions with the highest predicted Q-value).
3. **Step in Environment**: Execute action 𝑎, observe next state 𝑠′, reward 𝑟, and done flag 𝑑.
4. **Store in Replay Buffer**: Add (𝑠,𝑎,𝑟,𝑠′,𝑑) (s,a,r,s′,d) to the buffer.
5. **Sample Mini-Batch**: Randomly sample experiences from the replay buffer for training.
6. **Compute Targets**: Use the target network to compute the target Q-value:  
   $ 𝑦 = 𝑟 + 𝛾 \max_{𝑎} Q_{\text{target}}(𝑠′,𝑎) $  
   for non-terminal states.
7. **Update Q-Network**: Minimize the loss using gradient descent.:  
   $ 𝐿 = \frac{1}{𝑁} \sum (𝑦 − Q_{\text{policy}}(𝑠,𝑎))^2 $  
8. **Update Target Network**: Periodically update the target network to the current policy network.
