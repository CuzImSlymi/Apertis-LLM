# Apertis LLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

> A high-performance, multimodal Large Language Model architecture designed for superior reasoning, O(n) time complexity, and a streamlined user experience.

Apertis integrates a state-of-the-art selective state-space model, an adaptive expert system, and a professional-grade training and inference interface to deliver cutting-edge AI capabilities.

---

## ✨ Key Features

### 🔧 Innovative Hybrid Architecture
- **Mamba-style Selective State-Space Model** for linear-time processing
- **Traditional Transformer structure** for robust performance
- **O(n) time complexity** for efficient scaling

### 🧠 Adaptive Expert System
- **Mixture-of-Experts (MoE)** layer implementation
- **Conditional computation** with selective parameter activation
- **Optimized resource utilization** per token

### 🎨 Native Multimodal Support
- **Unified Multimodal Encoder** for seamless text and image processing
- **Genuine multimodal reasoning** capabilities
- **Cross-modal understanding** and generation

### 🚀 Powerful Training & Data Pipelines
- **Standard Supervised Learning**: Comprehensive pre-training and fine-tuning pipeline
- **Absolute Zero Reasoner (AZR)**: Advanced self-play mechanism for bootstrapping reasoning without human-labeled data
- **Terabyte-Scale Data Processing**: State-of-the-art distributed pipeline using Spark for massive web-scale datasets

### 💻 Professional Web Interface
- **Gradio-based UI** inspired by Google AI Studio
- **Integrated tools** for chat, training, model management, and monitoring
- **Intuitive workflow** for both beginners and experts

### 🔄 Cross-Platform & Reproducible
- **One-click Windows launcher** for easy setup
- **Docker support** for consistent environments
- **Multi-platform compatibility** (Windows, macOS, Linux)

### ⚡ Command-Line Interface
- **Full-featured CLI** for advanced users
- **Automation-ready** commands
- **Scriptable workflows**

---

## 🚀 Quick Start

### Windows (Easiest Method)

1. **Download** and extract the Apertis zip file
2. **Ensure** Python 3.8+ and Java are installed
3. **Double-click** `run_windows.py` to auto-install dependencies and launch
4. **Open** your browser at `http://localhost:7860`

### Docker (Cross-Platform)

```bash
# Clone the repository
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Install the project in editable mode (recommended for development)
pip install -e .

# Run with Docker Compose
docker-compose up


Open your browser to http://localhost:7860

Manual Installation (All Platforms)
Generated bash
# Clone the repository
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Install dependencies (ensure you have Java installed for the data pipeline)
pip install -r requirements.txt

# Install the project in editable mode to make modules available
pip install -e .

# Launch the web interface
python src/apertis_cli.py chat --web
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
🔄 Data Processing Pipeline

Apertis includes a powerful, configuration-driven data processing pipeline for preparing web-scale pre-training data. It uses PySpark for distributed processing and can run on your local machine for testing or scale to a massive cluster.

Prerequisites

Java: PySpark requires a Java installation on your system

Workflow
1. Create a Configuration File

Generate a default configuration file to control every stage of the pipeline:

Generated bash
apertis create-pipeline-config --output my_pipeline_config.yaml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
2. Edit the Configuration

Open my_pipeline_config.yaml and adjust the settings:

For local testing, keep the default spark.master: "local[*]"

Set the number of Common Crawl files to download (e.g., num_warc_files: 10 for a small test)

Download the FastText language identification model (lid.176.bin) and update the path in clean.fasttext_model_path

3. Run the Pipeline
Generated bash
apertis data-pipeline --config my_pipeline_config.yaml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The pipeline will execute the configured stages (download, clean, deduplicate, tokenize), saving the output of each stage to disk.

🌐 Web Interface (Apertis AI Studio)

The interface provides a complete toolkit for working with Apertis models:

Feature	Description
Chat	Interact with loaded models. Supports text and image uploads for multimodal inference
Pre-training	Train a new model from scratch using a standard supervised learning pipeline
Fine-tuning	Adapt a pre-trained Apertis model to a new task
Absolute Zero Reasoner	Train a model using the advanced self-play reasoning pipeline
Models	Load existing models for chat or create new, randomly-initialized model architectures
💻 Command-Line Interface (CLI)

Apertis provides a powerful CLI for programmatic access:

Generated bash
# Access help for all commands
apertis --help

# Launch interactive chat in the terminal
apertis chat --model-path path/to/your/model

# Train a model using a configuration file
apertis train --config my_training_config.json

# Run the data processing pipeline
apertis data-pipeline --config my_pipeline_config.yaml

# Create a new, randomly initialized model architecture
apertis create-model --target-params 1.5B --multimodal --output-dir models/my_new_model

# Generate a sample training configuration file
apertis create-config --output my_training_config.json

# Generate a sample data pipeline configuration file
apertis create-pipeline-config --output my_pipeline_config.yaml
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
📖 Documentation
Resource	Description
Architecture Design	In-depth technical details of the Apertis architecture
Training Guide	How to use the standard supervised training pipeline
Data Pipeline Guide	Detailed instructions for the distributed data processing pipeline
Absolute Zero Reasoner (AZR)	Guide to the self-play reasoning training method
UI Guide	A tour of the Apertis AI Studio web interface
Docker Guide	Instructions for using Apertis with Docker
Windows Guide	Detailed setup for Windows users
📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

<div align="center">
<p>Made with ❤️ by Slymi</p>
<p>
<a href="https://github.com/CuzImSlymi/Apertis-LLM">⭐ Star it on GitHub</a> •
<a href="https://github.com/CuzImSlymi/Apertis-LLM/issues">🐛 Report Issues</a> •
<a href="https://github.com/CuzImSlymi/Apertis-LLM/discussions">💬 Discussions</a>
</p>
</div>
