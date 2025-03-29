# Fine-tuning RL Models for Mortal Kombat II

This guide explains how to fine-tune pre-trained reinforcement learning models to improve their performance in specific scenarios or against challenging opponents.

## What is Fine-tuning?

Fine-tuning is the process of continuing training on a pre-trained model, focusing on specific areas that need improvement. It's a form of transfer learning that leverages knowledge gained from prior training to efficiently adapt to new challenges.

## When to Fine-tune

Consider fine-tuning your models when:

- They perform well overall but struggle against specific opponents
- You want to improve performance in advanced scenarios without starting from scratch
- You need to adapt a general model to a specialized task
- You've reached a performance plateau with regular training
- You have limited computational resources for full retraining

## Using the Fine-tuning Script

The Kane vs Abel framework includes a dedicated fine-tuning script (`finetune.py`) that handles loading pre-trained models and continuing their training on challenging scenarios.

### Basic Usage

```bash
python finetune.py
```

By default, this will:
1. Load the model specified in the script
2. Create environments with challenging states
3. Continue training for a defined number of timesteps
4. Save the fine-tuned model

## Fine-tuning Process

### 1. Select a Pre-trained Model

Choose a well-performing base model to fine-tune:

```python
pretrained_model_path = os.path.join(MODEL_DIR, "kane", "DuellingDDQN_curriculum_16M_VeryEasy_3_Tiers")
model = DuelingDoubleDQN.load(pretrained_model_path, env=env, device="cuda")
```

### 2. Define Challenging States

Select specific challenging states where the model needs improvement:

```python
challenging_states = [
    "VeryEasy.LiuKang-04",
    "VeryEasy.LiuKang-05",
    "VeryEasy.LiuKang-06",
    "VeryEasy.LiuKang-07",
    "VeryEasy.LiuKang-08"
]
```

### 3. Adjust Learning Rate for Fine-tuning

Use a more conservative learning rate to avoid catastrophic forgetting:

```python
# Lower starting rate and gentler decay for fine-tuning
fine_tune_lr = Schedules.linear_decay(1e-4, 1e-5)  # Original might be 3e-4 to 1e-5
model.learning_rate = fine_tune_lr
```

### 4. Optionally Freeze Early Layers

To preserve learned features while allowing adaptation in higher layers:

```python
# Uncomment to freeze feature extractor
# for param in model.policy.features_extractor.parameters():
#     param.requires_grad = False
```

### 5. Configure Evaluation During Fine-tuning

Set up evaluation callbacks to monitor progress:

```python
eval_callback = CustomEvalCallback(
    eval_env=eval_env,
    best_model_save_path=os.path.join(LOG_DIR, "fine_tuned_best"),
    log_path=os.path.join(LOG_DIR, "fine_tune_eval"),
    eval_freq=62_500,  # How often to evaluate
    n_eval_episodes=10,
    deterministic=False,
    render=False,
    verbose=1
)
```

### 6. Continue Training

Train for additional timesteps, typically fewer than the original training:

```python
model.learn(
    total_timesteps=2_000_000,  # Less than original training
    reset_num_timesteps=False,  # Continue from previous steps
    callback=eval_callback,
)
```

## Customizing Fine-tuning

### Target Specific Weaknesses

Identify and focus on specific weaknesses through careful state selection:

```python
# Example: Focus on specific opponents
challenging_states = [
    "Medium.LiuKangVsBaraka",  # If struggling against Baraka
    "Hard.LiuKangVsReptile"     # If struggling against Reptile
]
```

### Learning Rate Schedule Strategies

Different schedules for different fine-tuning goals:

```python
# For minor adjustments (conservative)
conservative_lr = Schedules.linear_decay(5e-5, 1e-6)

# For significant adaptation (more aggressive)
adaptive_lr = Schedules.linear_decay(1e-4, 1e-5)

# For focused, short fine-tuning
cyclical_lr = Schedules.cyclical_lr(5e-5, 1e-4, 0.5)
```

### Selective Layer Freezing

For more control over what parts of the model adapt:

```python
# Option 1: Freeze just the convolutional base
for name, param in model.policy.features_extractor.cnn.named_parameters():
    param.requires_grad = False

# Option 2: Freeze specific layers
for name, param in model.policy.named_parameters():
    if "features_extractor" in name:
        param.requires_grad = False
    if "q_net.0" in name:  # First layer after feature extraction
        param.requires_grad = False
```

### Exploration Settings

Adjust exploration parameters for fine-tuning:

```python
# Reduce exploration for fine-tuning
model.exploration_schedule = Schedules.linear_decay(0.1, 0.01)  # Lower than original
```

## Advanced Fine-tuning Techniques

### Regularization for Fine-tuning

Add regularization to prevent overfitting to the new states:

```python
# Add L2 regularization to optimizer (example for Adam)
from torch import optim

# Get current params
optimizer_params = model.policy.optimizer.defaults
# Add weight decay (L2 regularization)
optimizer_params['weight_decay'] = 1e-4
# Recreate optimizer
model.policy.optimizer = optim.Adam(
    model.policy.parameters(), 
    **optimizer_params
)
```

### Elastic Weight Consolidation

For complex models, implement Elastic Weight Consolidation (EWC) to preserve important weights:

```python
# Pseudo-code for EWC implementation
original_params = {name: param.clone().detach() for name, param in model.named_parameters()}
fisher_information = estimate_fisher_information(model, old_env)

def ewc_loss(model, original_params, fisher_information, lambda_ewc=5000):
    loss = 0
    for name, param in model.named_parameters():
        loss += fisher_information[name] * (param - original_params[name]).pow(2).sum()
    return lambda_ewc * loss
```

### Rehearsal with Mixed Experiences

Fine-tune with a mix of old and new experiences to prevent forgetting:

```python
# Create environments with mixed states
mixed_states = original_training_states + challenging_states
mixed_env = SubprocVecEnv([make_env(mixed_states) for _ in range(NUM_ENVS)])
```

### Progressive Fine-tuning

Gradually introduce challenging scenarios:

```python
# Start with a mix favoring original states
phase1_states = original_states + [challenging_states[0]]
# Then introduce more challenging states
phase2_states = original_states + challenging_states[:3]
# Finally use all challenging states
phase3_states = challenging_states
```

## Evaluating Fine-tuning Success

After fine-tuning, evaluate comprehensively:

```bash
# Evaluate on original states
python test.py --model_path models/pre_finetuned.zip --model_type DUELINGDDQN --states "original_state1,original_state2" --individual_eval

python test.py --model_path models/post_finetuned.zip --model_type DUELINGDDQN --states "original_state1,original_state2" --individual_eval

# Evaluate on challenging states
python test.py --model_path models/pre_finetuned.zip --model_type DUELINGDDQN --states "challenging_state1,challenging_state2" --individual_eval

python test.py --model_path models/post_finetuned.zip --model_type DUELINGDDQN --states "challenging_state1,challenging_state2" --individual_eval
```

Look for:
1. Improvement on challenging states
2. Maintenance of performance on original states
3. Overall win rate changes
4. Changes in behavior patterns

## Example: Complete Fine-tuning Workflow

```python
import os
from mk_ai.agents import DuelingDoubleDQN
from mk_ai.utils import Schedules
from mk_ai.callbacks import CustomEvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

# 1. Identify where the model is struggling (through evaluation)
# python test.py --model_path models/base_model.zip --model_type DUELINGDDQN --states "all_states" --individual_eval

# 2. Select challenging states based on evaluation
challenging_states = [
    "VeryEasy.LiuKang-06",  # Low win rate identified here
    "VeryEasy.LiuKang-07",  # Low win rate identified here
    "VeryEasy.LiuKang-08"   # Low win rate identified here
]

# 3. Create environments for fine-tuning
env = SubprocVecEnv([make_env(challenging_states) for _ in range(8)])
env = VecFrameStack(env, n_stack=4)

# 4. Load pre-trained model
model = DuelingDoubleDQN.load("models/base_model.zip", env=env, device="cuda")

# 5. Configure for fine-tuning
model.learning_rate = Schedules.linear_decay(1e-4, 1e-5)

# 6. Setup evaluation
eval_env = DummyVecEnv([make_env(challenging_states)])
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_callback = CustomEvalCallback(
    eval_env=eval_env,
    best_model_save_path="./logs/fine_tuned_best",
    log_path="./logs/fine_tune_eval",
    eval_freq=60000,
    n_eval_episodes=10
)

# 7. Fine-tune the model
model.learn(
    total_timesteps=2_000_000,
    reset_num_timesteps=False,
    callback=eval_callback
)

# 8. Save fine-tuned model
model.save("models/fine_tuned_model")

# 9. Evaluate the fine-tuned model
# python test.py --model_path models/fine_tuned_model.zip --model_type DUELINGDDQN --states "all_states" --individual_eval
```

## Best Practices for Fine-tuning

1. **Always evaluate before and after** to quantify improvements
2. **Start with a lower learning rate** than the original training
3. **Be selective with states** - target specific weaknesses
4. **Monitor for catastrophic forgetting** on original scenarios
5. **Keep fine-tuning sessions shorter** than original training
6. **Save intermediate checkpoints** during fine-tuning
7. **Consider layer freezing** for specialized adaptations
8. **Use TensorBoard** to monitor the fine-tuning process
