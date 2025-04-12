# Windows Installation Guide

This guide provides detailed instructions for installing and running Apertis LLM on Windows.

## System Requirements

- Windows 10 or 11
- Python 3.8 or higher
- 8GB RAM minimum (16GB recommended)
- 2GB free disk space

## Installation Methods

### Method 1: One-Click Launcher (Recommended)

This is the simplest method for Windows users:

1. Download and extract the Apertis zip file
2. Double-click `run_windows.py` to launch
3. A web browser will open automatically with the Apertis interface

The launcher will:
- Check your Python installation
- Install required dependencies
- Set up the environment
- Launch the web interface

### Method 2: Manual Installation

If you prefer to install manually:

1. Download and extract the Apertis zip file
2. Open Command Prompt or PowerShell
3. Navigate to the extracted directory:
   ```
   cd path\to\apertis
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Launch the web interface:
   ```
   python src\apertis_cli.py chat --web
   ```

### Method 3: Using Docker on Windows

1. Install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
2. Download and extract the Apertis zip file
3. Open Command Prompt or PowerShell
4. Navigate to the extracted directory:
   ```
   cd path\to\apertis
   ```
5. Run with Docker Compose:
   ```
   docker-compose up
   ```
6. Open your browser to http://localhost:7860

## Troubleshooting

### Python Not Found

If you get an error that Python is not found:

1. Make sure Python is installed. Download from [python.org](https://www.python.org/downloads/)
2. During installation, check "Add Python to PATH"
3. Restart your computer
4. Try running the launcher again

### Dependency Installation Errors

If dependencies fail to install:

1. Try running Command Prompt or PowerShell as Administrator
2. Install dependencies manually:
   ```
   pip install torch numpy tqdm pillow gradio wandb torchvision
   ```
3. If torch fails to install, try:
   ```
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Docker Issues

If Docker doesn't work:

1. Make sure Docker Desktop is running
2. Check that virtualization is enabled in your BIOS
3. Try restarting Docker Desktop
4. Run the following commands to reset Docker:
   ```
   docker system prune -a
   docker-compose up
   ```

## Using Apertis on Windows

After installation, you can:

1. Use the web interface by opening http://localhost:7860 in your browser
2. Use the command line interface:
   ```
   python src\apertis_cli.py chat
   ```
3. Train models using the web interface or command line:
   ```
   python src\apertis_cli.py train --config config.json
   ```

## Getting Help

If you encounter any issues not covered in this guide:

1. Check the [GitHub repository](https://github.com/CuzImSlymi/Apertis-LLM) for updates
2. Open an issue on GitHub with details about your problem
3. Try the manual installation method if the one-click launcher doesn't work
