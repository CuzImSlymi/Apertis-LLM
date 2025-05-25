# Absolute Zero Reasoner Configuration Guide

This document provides detailed information about configuring the Absolute Zero Reasoner (AZR) training method in Apertis-LLM.

## Configuration Parameters

The AZR training method can be configured through a JSON configuration file. Below are the key parameters:

### General Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_iterations` | int | 100 | Number of training iterations to run |
| `tasks_per_iteration` | int | 5 | Number of tasks to generate per iteration |
| `checkpoint_interval` | int | 10 | Save checkpoint every N iterations |
| `continue_from_checkpoint` | bool | false | Whether to continue from a previous checkpoint |
| `checkpoint_path` | string | "" | Path to checkpoint file when continuing training |
| `checkpoint_dir` | string | "checkpoints" | Directory to save checkpoints |
| `log_level` | string | "INFO" | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `log_file` | string | null | Path to log file (null for console logging) |

### Forced Acceptance Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_accept_tasks` | bool | true | Whether to force accept tasks regardless of validation |
| `force_accept_solutions` | bool | true | Whether to force accept solutions regardless of validation |
| `force_accept_threshold` | int | 10 | Iteration threshold after which to disable forced task acceptance |
| `min_valid_tasks_before_validation` | int | 20 | Minimum number of valid tasks before disabling forced solution acceptance |

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | object | {} | Model configuration parameters |
| `vocab_path` | string | null | Path to vocabulary file |
| `device` | string | "cuda" | Device to use for training ("cuda" or "cpu") |

### Task Generation and Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task_generator` | object | {} | Task generator configuration |
| `task_validator` | object | {} | Task validator configuration |

### Solution Generation and Validation

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `solution_generator` | object | {} | Solution generator configuration |
| `solution_validator` | object | {} | Solution validator configuration |

### Reward Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learnability_reward` | object | {} | Learnability reward configuration |
| `diversity_reward` | object | {} | Diversity reward configuration |
| `complexity_reward` | object | {} | Complexity reward configuration |

## Example Configuration

```json
{
  "num_iterations": 100,
  "tasks_per_iteration": 5,
  "checkpoint_interval": 10,
  "force_accept_tasks": true,
  "force_accept_threshold": 10,
  "min_valid_tasks_before_validation": 20,
  "model": {
    "vocab_size": 32000,
    "hidden_size": 512,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-06,
    "use_cache": true,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": false,
    "rope_theta": 10000.0
  },
  "device": "cuda",
  "task_generator": {
    "max_length": 512,
    "temperature": 0.7,
    "top_p": 0.9
  },
  "task_validator": {
    "min_length": 10,
    "complexity_threshold": 0.1,
    "clarity_threshold": 0.3
  },
  "solution_generator": {
    "max_length": 1024,
    "temperature": 0.8,
    "top_p": 0.95
  },
  "solution_validator": {
    "min_length": 20,
    "correctness_threshold": 0.5
  },
  "learnability_reward": {
    "weight": 1.0
  },
  "diversity_reward": {
    "weight": 0.5,
    "history_size": 10
  },
  "complexity_reward": {
    "weight": 0.3,
    "target_complexity": 0.7,
    "tolerance": 0.2
  }
}
```

## Training Dynamics

The AZR training method uses a self-play loop where the model generates reasoning tasks and solutions, with rewards guiding the learning process toward more effective reasoning.

### Forced Acceptance Mechanism

To facilitate early training when the model may not generate high-quality tasks or solutions, AZR implements a forced acceptance mechanism:

1. Initially, all generated tasks and solutions are accepted regardless of validation results
2. After `force_accept_threshold` iterations, task validation is enabled
3. After accumulating `min_valid_tasks_before_validation` valid tasks, solution validation is enabled

This gradual transition from forced acceptance to validation ensures that training can progress even with initially poor generation quality, while eventually enforcing quality standards as the model improves.

### Metrics and Monitoring

The AZR training process logs detailed metrics for monitoring training progress:

- Task and solution validation rates
- Reward distributions
- Task type distributions
- Generated content samples

These metrics can be used to assess whether the model is improving over time and to identify areas for configuration adjustment.

## Troubleshooting

If training is not progressing effectively:

1. Increase `force_accept_threshold` to allow more iterations with forced acceptance
2. Adjust reward weights to emphasize different aspects of task/solution quality
3. Modify validation thresholds to be more lenient initially
4. Check the logs for specific validation failures and adjust parameters accordingly
