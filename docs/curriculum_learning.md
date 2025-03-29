# Curriculum Learning for Mortal Kombat II

Curriculum learning is an optional training strategy that gradually increases the difficulty of training scenarios as the agent improves. This approach can lead to more efficient learning and better final performance, especially for complex fighting game environments like Mortal Kombat II.

## What is Curriculum Learning?

Curriculum learning mimics human learning by starting with simpler tasks and gradually increasing complexity. In the context of Mortal Kombat II training:

1. The agent begins fighting against very easy opponents with limited move sets
2. As performance improves, the agent advances to more challenging opponents
3. Eventually, the agent faces opponents with full move sets and higher difficulty settings

## Benefits of Curriculum Learning

- **Faster initial learning**: Agents learn basic mechanics more quickly
- **Higher final performance**: Gradual progression helps avoid local optima
- **More robust behaviors**: Exposure to diverse scenarios builds generalization
- **Reduced training time**: More efficient exploration of the state space

## Implementing Curriculum Learning

Kane vs Abel framework implements curriculum learning through tiered state lists and a dedicated callback:

### 1. Define State Tiers

First, define tiers of game states with increasing difficulty:

```python
tier1_states = ["Level1.LiuKangVsJax", "VeryEasy.LiuKang-02", "VeryEasy.LiuKang-03"]

tier2_states = [
    "Level1.LiuKangVsJax", "VeryEasy.LiuKang-02", "VeryEasy.LiuKang-03",
    "VeryEasy.LiuKang-04", "VeryEasy.LiuKang-05"
]

tier3_states = [
    "Level1.LiuKangVsJax", "VeryEasy.LiuKang-02", "VeryEasy.LiuKang-03",
    "VeryEasy.LiuKang-04", "VeryEasy.LiuKang-05", "VeryEasy.LiuKang-06",
    "VeryEasy.LiuKang-07", "VeryEasy.LiuKang-08"
]

tiered_states = [tier1_states, tier2_states, tier3_states]
```

### 2. Create the CurriculumCallback

Enable the curriculum learning callback in your training script:

```python
curriculum_callback = CurriculumCallback(
    vec_env=venv,
    tiered_states=tiered_states,
    verbose=1,
    buffer_size=100
)
```

### 3. Add to Callback List

Include the curriculum callback in your model's training:

```python
callback_list = CallbackList([eval_callback, curriculum_callback])

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    reset_num_timesteps=True,
    callback=callback_list
)
```

## How the Curriculum Callback Works

The `CurriculumCallback` in `mk_ai.callbacks.curriculum` manages the progression through training tiers:

1. **Initialization**: Sets up with tier 1 states and creates a reward buffer
2. **Performance Tracking**: Monitors the agent's average reward over recent episodes
3. **Tier Advancement**: When average reward exceeds thresholds, advances to next tier:
   - Tier 1 → Tier 2: When average reward > 50
   - Tier 2 → Tier 3: When average reward > 150
   - Tier 3 → Tier 4 (if defined): When average reward > 250
4. **Environment Update**: When advancing tiers, updates all parallel environments with new state sets

### Key Parameters for CurriculumCallback

- `vec_env`: The vectorized environment to update
- `tiered_states`: List of state lists for each tier
- `buffer_size`: Number of episode rewards to average (default: 20)
- `verbose`: Logging verbosity level

## Customizing the Curriculum

### Custom Advancement Thresholds

If you want different thresholds for curriculum advancement, modify the `_on_step` method in `CurriculumCallback`:

```python
def _on_step(self) -> bool:
    # ...existing code...
    
    # Custom thresholds
    if self.current_tier_idx == 0 and avg_reward > 75:  # Changed from 50
        self.current_tier_idx = 1
        print(f"[Callback] Switching to Tier 2, avg_reward={avg_reward:.2f}")
        self._update_env_states()
    elif self.current_tier_idx == 1 and avg_reward > 200:  # Changed from 150
        self.current_tier_idx = 2
        print(f"[Callback] Switching to Tier 3, avg_reward={avg_reward:.2f}")
        self._update_env_states()
    
    # ...existing code...
```

### Custom State Progression

You can define your own progression strategy by creating custom tier lists:

```python
# Character-based progression (same character, increasing difficulty)
tier1 = ["VeryEasy.LiuKang-01", "VeryEasy.LiuKang-02"]
tier2 = ["Easy.LiuKang-01", "Easy.LiuKang-02"]
tier3 = ["Medium.LiuKang-01", "Medium.LiuKang-02"]

# Or opponent-based progression (increasing variety)
tier1 = ["VeryEasy.LiuKangVsJax", "VeryEasy.LiuKangVsBaraka"]
tier2 = ["VeryEasy.LiuKangVsJax", "VeryEasy.LiuKangVsBaraka", 
         "VeryEasy.LiuKangVsReptile", "VeryEasy.LiuKangVsKitana"]
tier3 = ["Easy.LiuKangVsJax", "Easy.LiuKangVsBaraka", 
         "Easy.LiuKangVsReptile", "Easy.LiuKangVsKitana"]
```

## Example: Full Training with Curriculum Learning

Here's a complete example of setting up curriculum learning in a training script:

```python
from mk_ai.callbacks import CurriculumCallback, CustomEvalCallback
from stable_baselines3.common.callbacks import CallbackList

# Define curriculum tiers
tier1_states = ["VeryEasy.LiuKang-01", "VeryEasy.LiuKang-02"]
tier2_states = ["Easy.LiuKang-01", "Easy.LiuKang-02", "Easy.LiuKang-03"]
tier3_states = ["Medium.LiuKang-01", "Medium.LiuKang-02"]
tiered_states = [tier1_states, tier2_states, tier3_states]

# Create vectorized environment (start with tier 1)
venv = SubprocVecEnv([make_env(tier1_states) for _ in range(8)])
stacked_env = VecFrameStack(venv, n_stack=4)

# Create model
model = DuelingDoubleDQN(
    env=stacked_env,
    verbose=1,
    device="cuda",
    # ...other parameters...
)

# Create curriculum callback
curriculum_callback = CurriculumCallback(
    vec_env=venv,
    tiered_states=tiered_states,
    verbose=1,
    buffer_size=100
)

# Create evaluation callback
eval_callback = CustomEvalCallback(
    # ...evaluation parameters...
)

# Create callback list
callbacks = CallbackList([eval_callback, curriculum_callback])

# Train with curriculum learning
model.learn(
    total_timesteps=16_000_000,
    callback=callbacks
)
```

## When to Use Curriculum Learning

Curriculum learning is most beneficial:
- When training from scratch in complex environments
- When the agent struggles to learn meaningful behaviors with random initialization
- When you have a clear progression of difficulty levels available

It may be less necessary:
- When fine-tuning pre-trained models
- In simple environments where learning is already efficient
- When computational resources are severely limited (adds overhead)

## Monitoring Curriculum Progress

During training, the curriculum callback will output log messages when it advances to a new tier:

```
[Callback] Switching to Tier 2, avg_reward=52.38
...
[Callback] Switching to Tier 3, avg_reward=157.92
```

You can monitor this progress along with other training metrics in TensorBoard.