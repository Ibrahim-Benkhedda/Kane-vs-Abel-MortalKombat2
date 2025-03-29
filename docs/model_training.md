# Training Custom RL Models for Mortal Kombat II

This guide provides an overview of training reinforcement learning models for the Mortal Kombat II environment. For more detailed information on specific topics, see the dedicated guides:

- [Curriculum Learning](curriculum_learning.md): Progressive training through increasing difficulty levels
- [Model Evaluation](model_evaluation.md): Testing and analyzing model performance
- [Fine-tuning Models](model_finetuning.md): Improving pre-trained models for specific scenarios

## Training a Model from Scratch

The `train.py` script provides a comprehensive framework for training RL models with various advanced features including optional curriculum learning and learning rate schedules.

### Basic Usage

```bash
python train.py
```

### Training Process Overview

1. **Environment Setup**: The script creates vectorized environments using `SubprocVecEnv` for parallel training
2. **Model Initialization**: Models are configured with hyperparameters optimized for fighting games
3. **Training Loop**: The model learns for a defined number of timesteps with periodic evaluation
4. **Model Saving**: The final model and best models during training are saved for later use

### Customizing Training

To customize your training, edit `train.py` and modify:

#### Model Type
Choose from:
- `DQN`: Standard Deep Q-Network
- `DoubleDQN`: Double DQN for more stable training
- `DuelingDoubleDQN`: Dueling architecture with Double DQN updates

```python
# Example: Change model type
model = DuelingDoubleDQN(  # or DoubleDQN or DQN
    env=stacked_env,
    verbose=1,
    device="cuda",
    # other parameters...
)
```

#### Learning Rate Schedules
Choose from multiple learning rate schedules in `mk_ai.utils.schedulers.Schedules`:

```python
# Linear decay
lr_schedule = Schedules.linear_decay(3.16e-4, 1e-5)

# Exponential decay
exp_decay_lr = Schedules.exponential_decay(3.16e-4, 0.295)

# Cyclical learning rates
cyclical_lr = Schedules.cyclical_lr(1e-4, 3e-4, 0.5)
```

#### Hyperparameters
Tune key hyperparameters for your specific training needs:

```python
model = DuelingDoubleDQN(
    env=stacked_env,
    buffer_size=200000,       # Replay buffer size
    batch_size=32,            # Batch size for updates
    gamma=0.95,               # Discount factor
    learning_rate=lr_schedule,
    exploration_fraction=0.3, # Fraction of training for exploration
    exploration_initial_eps=0.9,
    exploration_final_eps=0.07,
    tensorboard_log="./logs/my_custom_model/"
)
```

## Monitoring Training Progress

View training progress with TensorBoard:

```bash
tensorboard --logdir=./experiments_finals
```

This will show:
- Learning curves (reward, loss)
- Exploration rate decay
- Evaluation performance

## Best Practices

### Training
- Start with a simpler environment and gradually increase difficulty
- Use curriculum learning for complex environments (see [Curriculum Learning](curriculum_learning.md))
- Monitor training statistics via TensorBoard logs
- Save regular checkpoints during long training sessions
- Consider using parallel environments (SubprocVecEnv) to speed up training
- Experiment with different model architectures and learning rate schedules

## Troubleshooting

### Common Issues

- **GPU Out of Memory**: Reduce batch size or number of parallel environments
- **Slow Learning**: Adjust learning rate schedule or exploration parameters
- **Overfitting**: Increase environment variety or add regularization

### Debugging Tips

- Add verbose logging to track specific behaviors
- Use render_mode="human" for visual debugging of agent behaviors
- Implement custom callbacks for detailed monitoring
