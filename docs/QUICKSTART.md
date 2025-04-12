# Quick Start Guide

This guide will help you get started with Apertis LLM in just a few minutes.

## Windows Users

### One-Click Method (Recommended)

1. Download and extract the Apertis zip file
2. Double-click `run_windows.py` to launch
3. A web browser will open automatically with the Apertis interface

### Troubleshooting Windows Installation

If you encounter any issues:

- Make sure Python 3.8+ is installed and in your PATH
- Try running as administrator if you get permission errors
- If dependencies fail to install, run `pip install -r requirements.txt` manually

## macOS and Linux Users

### Using Docker (Recommended)

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

## Using the Web Interface

The Apertis web interface has three main tabs:

### Chat Tab

- Type messages in the text box and click "Send"
- Upload images for multimodal chat
- Adjust generation settings in the sidebar

### Training Tab

1. Configure your model (size, multimodal options)
2. Upload your training data (JSONL format)
3. Set training parameters
4. Click "Start Training"

### Models Tab

- Load existing models
- Create new models with different configurations

## Next Steps

- Try uploading an image and asking about it
- Experiment with different model sizes
- Check out the [Training Guide](training_guide.md) to create your own models
- Explore the [Architecture Design](architecture_design.md) to understand how Apertis works
