# Deep Q-Learning in Kane vs Abel Mortal Kombat II

This document explains the different Deep Q-Learning variants implemented in the project, their key differences, and how they work.

## Basic DQN Algorithm

### Core Components:
- **Q-Network**: Neural network that approximates the Q-value function
- **Replay Buffer**: Stores experience tuples (state, action, reward, next_state, done)
- **Target Network**: Copy of Q-network for stable target calculation
- **Exploration Policy**: Typically epsilon-greedy for balancing exploration and exploitation

### Training Loop:

1. **Initialize Environment and Replay Buffer**:
   - Start with an empty replay buffer to store experiences

2. **Initialize Q-Networks**:
   - Policy network (for action selection)
   - Target network (periodically updated copy)

3. **For each episode**:
   - Reset environment to initial state
   - For each step until episode ends:
     - Select action using epsilon-greedy policy
     - Execute action, observe reward and next state
     - Store experience in replay buffer
     - Sample mini-batch from replay buffer
     - Compute target Q-values: $y = r + \gamma \max_{a'} Q_{target}(s', a')$
     - Update Q-network by minimizing loss: $L = \frac{1}{N} \sum (y - Q(s,a))^2$
     - Periodically update target network

## Implemented Variants

### 1. Double DQN

**Implementation**: `mk_ai.agents.DQN.double_dqn.py`

**Key Improvement**: Addresses overestimation bias in standard DQN

**How It Works**:
- Uses the online network to select actions
- Uses the target network to evaluate those actions
- Target calculation: $y = r + \gamma Q_{target}(s', \arg\max_{a'} Q_{online}(s', a'))$

**Code Highlight**:
```python
# Select action with online network
next_actions_online = th.argmax(next_q_values_online, dim=1)
# Use target network to evaluate action
next_q_values = th.gather(next_q_values, dim=1, 
                          index=next_actions_online.unsqueeze(1)).flatten()
```

### 2. Dueling DQN

**Implementation**: dueling_dqn.py

**Key Improvement**: Separates state value and action advantage estimation

**How It Works**:
- Splits the Q-network into two streams:
  - Value stream: estimates state value V(s)
  - Advantage stream: estimates action advantages A(s,a)
- Combines them: $Q(s,a) = V(s) + (A(s,a) - \frac{1}{|A|}\sum_a A(s,a))$

**Network Architecture**:
- Shared feature extractor (usually CNN for game images)
- Separate value and advantage heads
- Special aggregation layer to combine outputs

### 3. Dueling Double DQN

**Implementation**: dueling_ddqn.py

**Key Improvement**: Combines both Double DQN and Dueling architecture benefits

**How It Works**:
- Uses Dueling architecture (separate value and advantage streams)
- Applies Double DQN update rule (decoupling action selection from evaluation)
- Inherits from Double DQN but uses Dueling network architecture

**Network Structure**:
```
┌──────────────────┐
│  Feature Layers  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│   Value Stream  │     │ Advantage Stream │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌────────────────────────────────────────┐
│      Q(s,a) = V(s) + A(s,a) - mean     │
└────────────────────────────────────────┘
```

## Comparing Performance

In Mortal Kombat II gameplay, we've observed:

- **Standard DQN**: Tends to overestimate action values, leading to suboptimal policies
- **Double DQN**: More conservative value estimates, steadier learning
- **Dueling DQN**: Better at estimating state values independent of actions
- **Dueling Double DQN**: Best overall performance, combining the strengths of both approaches

## Using Different DQN Variants in the Arena

The Arena system allows selecting different DQN variants:

```bash
# Standard DQN
python arena.py --p1-type dqn --p1-model models/dqn_model

# Double DQN
python arena.py --p1-type double_dqn --p1-model models/double_dqn_model

# Dueling DQN (architecturally different but same update rule as DQN)
python arena.py --p1-type dqn --p1-model models/dueling_dqn_model

# Dueling Double DQN (both architectural and update rule changes)
python arena.py --p1-type dueling_ddqn --p1-model models/dueling_double_dqn_model
```

## Implementation References

- Double DQN: [Deep Reinforcement Learning with Double Q-Learning](https://arxiv.org/abs/1509.06461)
- Dueling DQN: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)