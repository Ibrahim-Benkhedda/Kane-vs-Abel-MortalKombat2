# Evaluating RL Models for Mortal Kombat II

This guide explains how to thoroughly evaluate trained reinforcement learning models using the Kane vs Abel framework.

## Why Evaluate?

Proper evaluation helps:
- Determine if a model is ready for deployment
- Compare different training approaches
- Identify specific weaknesses or failure modes
- Track improvements during fine-tuning
- Understand generalization to new opponents/scenarios

## Using the Evaluation Script

The Kane vs Abel framework provides a comprehensive evaluation script (`test.py`) with multiple evaluation modes.

### Basic Usage

```bash
python test.py --model_path models/my_model.zip --model_type DUELINGDDQN --num_episodes 10
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--model_path` | Path to the saved model file |
| `--model_type` | Model type (DQN, DDQN, DUELINGDDQN, PPO) |
| `--game` | Name of the ROM/game (default: MortalKombatII-Genesis) |
| `--state` | Game state to load for evaluation |
| `--states` | Comma-separated list of states to evaluate on |
| `--individual_eval` | If set, evaluate each state individually |
| `--render_mode` | Render mode (human, rgb_array, none) |
| `--num_stack` | Number of frames to stack (default: 4) |
| `--num_skip` | Number of frames to skip (default: 4) |
| `--num_episodes` | Number of episodes for evaluation (default: 10) |

## Evaluation Modes

### Single State Evaluation

Evaluate on a single, specific game state:

```bash
python test.py --model_path models/kane/my_model.zip --model_type DUELINGDDQN \
    --state "Level1.LiuKangVsJax" --render_mode human
```

This mode is useful for:
- Visually inspecting model behavior against a specific opponent
- Focused testing of challenging scenarios
- Direct comparison between models on a standard benchmark

### Multiple States Evaluation (Combined)

Evaluate across multiple states with random sampling:

```bash
python test.py --model_path models/kane/my_model.zip --model_type DUELINGDDQN \
    --states "VeryEasy.LiuKang-04,VeryEasy.LiuKang-05,VeryEasy.LiuKang-06"
```

This mode is useful for:
- Testing overall model robustness
- Getting an aggregate performance metric
- Simulating varied opponent encounters

### Multiple States Evaluation (Individual)

Evaluate each state separately with individual results:

```bash
python test.py --model_path models/kane/my_model.zip --model_type DUELINGDDQN \
    --states "VeryEasy.LiuKang-04,VeryEasy.LiuKang-05" --individual_eval
```

This mode is useful for:
- Identifying specific strengths/weaknesses against different opponents
- Granular performance analysis
- Detecting overfitting to specific scenarios

## Understanding Evaluation Results

The evaluation script generates CSV files with detailed metrics:

### Output Format

```
episode,reward,won
1,253.0,True
2,187.5,True
3,-52.5,False
...
average,178.2,
std,97.5,
```

### Result Analysis

Key metrics to examine:

1. **Average Reward**: Higher is better, but context matters:
   - 200+: Excellent performance (usually wins consistently)
   - 100-200: Good performance (wins most matches)
   - 0-100: Mediocre performance (inconsistent results)
   - <0: Poor performance (usually loses)

2. **Win Rate**: Percentage of episodes won
   - Primary indicator of agent effectiveness
   - Consider alongside average reward (some wins might be barely scraped)

3. **Standard Deviation**: Indicates consistency:
   - Low std dev + high average: Consistent strong performance
   - High std dev: Inconsistent (sometimes great, sometimes terrible)

4. **Per-State Performance**: For individual evaluations
   - Identifies matchup-specific strengths/weaknesses
   - Helps target fine-tuning efforts

## Evaluation Strategies

### Visualization

Use the `--render_mode human` option to visually observe agent behavior:

```bash
python test.py --model_path models/my_model.zip --model_type DUELINGDDQN \
    --state "Level1.LiuKangVsJax" --render_mode human --num_episodes 3
```

This helps identify:
- Action patterns and strategies
- Positioning and spacing behavior
- Defensive reactions and counter-attacks
- Obvious mistakes or sub-optimal behaviors

### Cross-Model Comparison

To compare multiple models:

1. Evaluate each model on the same set of states:
   ```bash
   python test.py --model_path models/model_A.zip --model_type DUELINGDDQN --states "state1,state2,state3" --individual_eval
   python test.py --model_path models/model_B.zip --model_type DDQN --states "state1,state2,state3" --individual_eval
   ```

2. Compare results using the generated CSV files
3. Consider implementing an automated comparison script for large-scale evaluations

### Stress Testing

Test resilience by evaluating on particularly challenging scenarios:

```bash
python test.py --model_path models/my_model.zip --model_type DUELINGDDQN \
    --states "Hard.LiuKang-01,VeryHard.LiuKang-02" --individual_eval
```

### Ensemble Evaluation

For critical applications, use ensemble evaluation across many episodes (30+) for statistical significance:

```bash
python test.py --model_path models/my_model.zip --model_type DUELINGDDQN \
    --states "State1,State2,State3,State4,State5" --individual_eval --num_episodes 30
```

## Customizing the Evaluation Framework

### Custom Metrics

You can extend the `evaluate_agent` function in `test.py` to track additional metrics:

```python
def evaluate_agent(model, env, num_episodes=10):
    # ...existing code...
    
    additional_metrics = {
        'combos_performed': [],
        'avg_reaction_time': [],
        'defensive_actions': []
    }
    
    # ...track these during evaluation loop...
    
    return avg_reward, std_reward, episode_rewards, episode_wins, additional_metrics
```


## Best Practices for Evaluation

1. **Statistical Significance**: Always evaluate on enough episodes (10 minimum, 30+ preferred)
2. **Diverse Scenarios**: Test on states both seen and unseen during training
3. **Controlled Comparisons**: Use identical seeds when comparing models
4. **Regular Benchmarking**: Establish standard evaluation scenarios for ongoing development
5. **Version Control**: Track evaluation results alongside model versions
6. **Reproducibility**: Document exact evaluation parameters
7. **Reality Check**: Supplement metrics with visual inspection

## Troubleshooting Evaluation

Common issues and solutions:

- **Inconsistent Results**: Increase number of evaluation episodes
- **Model Loading Errors**: Verify model type matches what's specified
- **Environment Errors**: Check that specified game states exist in your ROM
- **Performance Gaps**: Compare against baseline models or human performance
- **Resource Usage**: For large evaluations, reduce render quality or use no rendering