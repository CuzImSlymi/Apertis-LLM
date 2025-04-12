# Apertis LLM

Apertis is a revolutionary new LLM architecture that combines state-of-the-art performance with exceptional ease of use. It features multimodal capabilities (text + image), innovative attention mechanisms, and a user-friendly interface inspired by Google AI Studio.

## Features

- **Innovative Architecture**: Combines the best aspects of Transformer, Mamba, RWKV, and Mixtral
- **Multimodal Support**: Process both text and images with the Unified Multimodal Encoder
- **Selective Linear Attention**: O(n) time complexity for efficient processing
- **Adaptive Expert System**: Efficient parameter usage through mixture of experts
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **User-Friendly Interface**: Google AI Studio-like UI for easy interaction
- **YOLO-Style Training**: Professional training pipeline with visualization
- **One-Click Setup**: Simple installation process for all skill levels

## Quick Start

### Windows Users (Easiest Method)

1. Download and extract the Apertis zip file
2. Double-click `run_windows.py` to launch
3. A web browser will open automatically with the Apertis interface

### Docker (Cross-Platform)

```bash
# Clone the repository
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Run with Docker
docker-compose up
```

Then open your browser to http://localhost:7860

### Manual Installation

```bash
# Clone the repository
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Install dependencies
pip install -r requirements.txt

# Launch the web interface
python src/apertis_cli.py chat --web
```

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get up and running in minutes
- [Windows Guide](docs/WINDOWS.md) - Detailed instructions for Windows users
- [Docker Guide](docs/DOCKER.md) - Using Apertis with Docker
- [Architecture Design](docs/architecture_design.md) - Technical details of the Apertis architecture
- [Training Guide](docs/training_guide.md) - How to train your own models

## Training Your Own Models

Apertis makes training custom models incredibly simple:

1. Launch the web interface
2. Go to the "Training" tab
3. Upload your training data
4. Configure your model settings
5. Click "Start Training"

For advanced training options, see the [Training Guide](docs/training_guide.md).

## Command Line Interface

Apertis includes a powerful CLI for advanced users:

```bash
# Chat with a model
python src/apertis_cli.py chat --model-path models/my_model

# Train a model
python src/apertis_cli.py train --config my_config.json

# Create a new model
python src/apertis_cli.py create-model --size base --multimodal

# Create a sample training configuration
python src/apertis_cli.py create-config --output my_config.json
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
