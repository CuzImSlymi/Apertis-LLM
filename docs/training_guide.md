# Training Guide for Apertis LLM

This guide provides detailed instructions for training your own models with Apertis LLM, using the YOLO-style training interface.

## Training Overview

Apertis makes training custom LLMs as simple as training YOLO models. The training process involves:

1. Preparing your training data
2. Configuring your model architecture
3. Setting training parameters
4. Monitoring training progress
5. Evaluating and using your trained model

## Preparing Training Data

Apertis uses JSONL (JSON Lines) format for training data, where each line is a valid JSON object:

```jsonl
{"text": "This is an example training sentence.", "label": "example"}
{"text": "Another training example with different text.", "label": "example"}
```

For multimodal training, include image paths:

```jsonl
{"text": "A description of the image", "image": "images/example1.jpg"}
{"text": "Another image description", "image": "images/example2.jpg"}
```

### Data Requirements

- Text data should be clean and representative of your target domain
- For multimodal training, images should be in JPG, PNG, or WebP format
- Recommended minimum dataset size: 1,000 examples
- Split your data into training (80%) and validation (20%) sets

## Using the Training Interface

### Web Interface (Recommended)

1. Launch the Apertis web interface:
   ```bash
   python src/apertis_cli.py chat --web
   ```

2. Navigate to the "Training" tab

3. Configure your model:
   - Select model size (small, base, large)
   - Enable/disable multimodal support
   - Enable/disable Adaptive Expert System

4. Upload your training data:
   - Training data (JSONL format)
   - Validation data (optional, JSONL format)
   - Vocabulary file (JSON format)
   - Image directory (for multimodal training)

5. Set training parameters:
   - Batch size
   - Learning rate
   - Number of epochs
   - Evaluation frequency
   - Output directory
   - Weights & Biases logging (optional)

6. Click "Start Training" to begin

### Command Line Training

For advanced users, you can train via the command line:

1. Create a training configuration file:
   ```bash
   python src/apertis_cli.py create-config --output my_config.json
   ```

2. Edit the configuration file to customize settings (including `eval_every_n_epochs`).

3. Start training:
   ```bash
   python src/apertis_cli.py train --config my_config.json
   ```

## Training Configuration Options

### Model Configuration

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| model_size | Size of the model | "small", "base", "large" |
| hidden_size | Dimension of hidden layers | 512 (small), 768 (base), 1024 (large) |
| num_hidden_layers | Number of transformer layers | 8 (small), 12 (base), 24 (large) |
| num_attention_heads | Number of attention heads | 8 (small), 12 (base), 16 (large) |
| attention_type | Type of attention mechanism | "selective_linear" (faster), "full" (more accurate) |
| use_expert_system | Whether to use mixture of experts | true/false |
| multimodal | Whether to process images | true/false |

### Training Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| batch_size | Number of examples per batch | 4-32 (depends on GPU memory) |
| learning_rate | Learning rate for optimizer | 1e-5 to 5e-5 |
| num_epochs | Number of training epochs | 3-10 |
| eval_every_n_epochs | Frequency of validation at epoch end | 1 (every epoch), N (every N epochs), 0 (disable) |
| warmup_steps | Steps for learning rate warmup | 10% of total steps |
| gradient_accumulation_steps | Steps to accumulate gradients | 1-8 |
| fp16 | Whether to use mixed precision | true (if GPU supports it) |

## Monitoring Training Progress

### Using Weights & Biases

For the best training visualization:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Install the wandb package: `pip install wandb`
3. Login: `wandb login`
4. Enable "Use Weights & Biases for Logging" in the training interface
5. View real-time training metrics in your browser

### Local Monitoring

If not using Weights & Biases:

- Training logs are saved to the output directory
- Progress is displayed in the terminal/web interface
- Metrics are saved in JSON format after training

## Evaluating Your Model

After training completes:

1. Navigate to the "Models" tab in the web interface
2. Load your trained model from the output directory
3. Test it in the "Chat" tab
4. Compare performance with baseline models

## Advanced Training Techniques

### Fine-tuning Existing Models

To fine-tune a pre-trained model:

1. In the training configuration, set "pretrained_model_path" to your base model
2. Use a smaller learning rate (1e-5 or lower)
3. Train for fewer epochs (2-3)

### Hyperparameter Optimization

For optimal performance:

1. Start with the recommended parameters
2. Try different learning rates (1e-5, 3e-5, 5e-5)
3. Experiment with batch sizes
4. Adjust the number of epochs based on validation loss

### Distributed Training

For large models or datasets:

1. Set up multiple GPUs
2. Modify the training configuration to enable distributed training
3. Use gradient accumulation for larger effective batch sizes

## Troubleshooting

### Out of Memory Errors

If you encounter GPU memory errors:

1. Reduce batch size
2. Enable gradient accumulation
3. Use a smaller model size
4. Enable mixed precision training (fp16)

### Slow Training

If training is too slow:

1. Use "selective_linear" attention type
2. Disable the expert system for smaller models
3. Reduce the model size
4. Use a GPU with more compute capability

### Poor Model Performance

If your trained model performs poorly:

1. Check your training data quality and quantity
2. Train for more epochs
3. Adjust learning rate
4. Try different model architectures
5. Add more diverse examples to your dataset