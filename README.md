# Apertis LLM

Apertis is a high-performance, multimodal Large Language Model architecture designed for superior reasoning, O(n) time complexity, and a streamlined user experience. It integrates a state-of-the-art selective state-space model, an adaptive expert system, and a professional-grade training and inference interface.

## Key Features

- **Innovative Hybrid Architecture**: Combines a Mamba-style Selective State-Space Model for linear-time processing with the robust structure of a traditional Transformer.
- **Adaptive Expert System**: Implements a Mixture-of-Experts (MoE) layer for efficient, conditional computation, activating only the necessary parameters per token.
- **Native Multimodal Support**: The Unified Multimodal Encoder seamlessly processes text and image inputs for genuine multimodal reasoning tasks.
- **Powerful Training Pipelines**:
  - **Standard Supervised Learning**: A comprehensive pipeline for pre-training and fine-tuning with your own data.
  - **Absolute Zero Reasoner (AZR)**: An advanced self-play mechanism for bootstrapping reasoning capabilities without human-labeled data, inspired by the paper "Absolute Zero Reasoner".
- **Professional Web Interface**: A Gradio-based UI inspired by Google AI Studio, providing integrated tools for chat, training, model management, and monitoring.
- **Cross-Platform & Reproducible**: One-click Windows launcher and Docker support ensure a consistent environment on Windows, macOS, and Linux.
- **Command-Line Interface**: A full-featured CLI for advanced users and automation.

## Quick Start

### Windows (Easiest Method)

1. Download and extract the Apertis zip file.
2. Ensure Python 3.8+ is installed.
3. Double-click `run_windows.py` to automatically install dependencies and launch the application.
4. A web browser will open at `http://localhost:7860`.

### Docker (Cross-Platform)

```bash
# Clone the repository
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Install the project in editable mode (recommended for development)
pip install -e .

# Run with Docker Compose
docker-compose up
```

Open your browser to [http://localhost:7860](http://localhost:7860).

### Manual Installation (All Platforms)

```bash
# Clone the repository
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Install dependencies
pip install -r requirements.txt

# Install the project in editable mode to make modules available
pip install -e .

# Launch the web interface
python src/apertis_cli.py chat --web
```

## Web Interface (Apertis AI Studio)

The interface provides a complete toolkit for working with Apertis models:

- **Chat**: Interact with loaded models. Supports text and image uploads for multimodal inference.
- **Pre-training**: Train a new model from scratch using a standard supervised learning pipeline.
- **Fine-tuning**: Adapt a pre-trained Apertis model to a new task.
- **Absolute Zero Reasoner**: Train a model using the advanced self-play reasoning pipeline.
- **Models**: Load existing models for chat or create new, randomly-initialized model architectures.

## Command-Line Interface (CLI)

Apertis provides a powerful CLI for programmatic access:

```bash
# Access help for all commands
apertis --help

# Launch interactive chat in the terminal
apertis chat --model-path path/to/your/model

# Train a model using a configuration file
apertis train --config my_training_config.json

# Create a new, randomly initialized model architecture
apertis create-model --size base --multimodal --output-dir models/my_new_model

# Generate a sample training configuration file
apertis create-config --output my_training_config.json
```

## Documentation

- **Architecture Design**: In-depth technical details of the Apertis architecture.
- **Training Guide**: How to use the standard supervised training pipeline.
- **Absolute Zero Reasoner (AZR)**: Guide to the self-play reasoning training method.
- **UI Guide**: A tour of the Apertis AI Studio web interface.
- **Docker Guide**: Instructions for using Apertis with Docker.
- **Windows Guide**: Detailed setup for Windows users.

## License

This project is licensed under the MIT License. See the LICENSE file for details.