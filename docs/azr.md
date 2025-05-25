# Absolute Zero Reasoner (AZR) Documentation

## Overview

The Absolute Zero Reasoner (AZR) is an advanced training method that enables language models to develop reasoning capabilities through self-play. Unlike traditional supervised fine-tuning, AZR allows models to generate their own reasoning tasks and solutions, creating a bootstrapped learning process that can improve reasoning abilities without human-labeled data.

## Key Features

- **Self-play Loop**: Models generate their own reasoning tasks and solutions
- **Multi-type Reasoning**: Supports abductive, deductive, and inductive reasoning
- **Reward-guided Learning**: Uses specialized rewards to guide the learning process
- **Zero-shot Improvement**: Enhances reasoning without human-labeled examples

## How It Works

AZR operates through a self-play loop with two main phases:

1. **Propose Phase**: The model generates reasoning tasks of varying types
2. **Solve Phase**: The model attempts to solve its own generated tasks

The quality of both tasks and solutions is evaluated using specialized reward functions:

- **Learnability**: How well the task can be learned from
- **Diversity**: How different the task is from previously generated tasks
- **Complexity**: How challenging the task is

## Training Configuration

AZR training can be configured through the dedicated UI tab or by editing the configuration file directly. Key parameters include:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `num_iterations` | Number of training iterations | 100 |
| `tasks_per_iteration` | Number of tasks to generate per iteration | 5 |
| `task_types` | Types of reasoning tasks to generate | ["abduction", "deduction", "induction"] |
| `task_distribution` | Probability distribution for task types | [0.3, 0.3, 0.4] |
| `force_accept_tasks` | Whether to force accept all tasks during early training | true |
| `force_accept_threshold` | Iteration threshold after which to disable force acceptance | 50 |

## Early Training Behavior

During early training iterations, AZR is configured to **force accept all generated tasks and solutions** regardless of their quality. This ensures that training can proceed even when the model is not yet capable of generating high-quality content.

This behavior is controlled by the `force_accept_tasks` parameter and is enabled by default. After the model has improved through several iterations (controlled by `force_accept_threshold`), you can disable this feature to enforce stricter validation.

## Usage Guidelines

1. **For New Models**: Keep `force_accept_tasks` enabled to ensure training progress
2. **For Pre-trained Models**: You may disable force acceptance if your model already generates reasonable content
3. **For Advanced Training**: After initial training, disable force acceptance to improve task quality

## Monitoring Training

AZR provides detailed logs during training, including:

- Task generation and validation metrics
- Solution generation and validation metrics
- Reward values for each task and solution
- Overall training progress

These logs can be used to monitor the model's improvement over time and to identify any issues with the training process.

## Customization

Advanced users can customize the AZR training process by:

1. Modifying the reward functions in `src/training/azr/rewards.py`
2. Adjusting the validation criteria in `src/training/azr/data_construction.py`
3. Changing the task generation prompts in `src/training/azr/data_construction.py`

## Troubleshooting

If you encounter issues during AZR training:

1. **Tasks Always Invalid**: Ensure `force_accept_tasks` is set to `true`
2. **Training Too Slow**: Reduce `tasks_per_iteration` or use a smaller model
3. **Out of Memory**: Reduce batch size or model size
4. **Poor Quality Output**: Increase training iterations or adjust reward weights

## References

The Absolute Zero Reasoner method is based on research in self-play and bootstrapped learning for language models. For more information, see the original paper and repository:

- [Absolute Zero Reasoner: Training Language Models with Zero Human Feedback](https://github.com/LeapLabTHU/Absolute-Zero-Reasoner)
