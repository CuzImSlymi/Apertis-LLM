# Docker Guide for Apertis LLM

This guide provides detailed instructions for running Apertis LLM using Docker, which ensures consistent behavior across all platforms.

## Prerequisites

- Docker installed on your system
  - [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
  - [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop)
  - [Docker for Linux](https://docs.docker.com/engine/install/)
- Docker Compose installed (included with Docker Desktop)

## Quick Start

The simplest way to run Apertis with Docker:

```bash
# Clone the repository (or extract the zip file)
git clone https://github.com/CuzImSlymi/Apertis-LLM.git
cd Apertis-LLM

# Start Apertis using Docker Compose
docker-compose up
```

Then open your browser to http://localhost:7860

## Manual Docker Commands

If you prefer not to use Docker Compose, you can use these commands:

```bash
# Build the Docker image
docker build -t apertis .

# Run the container
docker run -p 7860:7860 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models apertis
```

## Understanding the Docker Setup

The Apertis Docker setup includes:

- A base image with Python and all dependencies pre-installed
- Volume mounts for data and models to persist your work
- Port mapping to access the web interface
- Automatic startup of the Apertis interface

## Customizing the Docker Setup

### Changing the Port

If port 7860 is already in use, you can change it in the `docker-compose.yml` file:

```yaml
services:
  apertis:
    # ... other settings ...
    ports:
      - "8080:7860"  # Change 8080 to any available port
```

### Adding GPU Support

To use GPU acceleration with Docker:

1. Install [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) (for NVIDIA GPUs)
2. Modify the `docker-compose.yml` file:

```yaml
services:
  apertis:
    # ... other settings ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Troubleshooting Docker Issues

### Container Fails to Start

If the container fails to start:

```bash
# Check the logs
docker-compose logs

# Try rebuilding the image
docker-compose build --no-cache
docker-compose up
```

### Module Import Errors

If you see "ModuleNotFoundError: No module named 'apertis'":

```bash
# Enter the container
docker-compose exec apertis bash

# Check the Python path
python -c "import sys; print(sys.path)"

# Try installing the package in development mode
pip install -e .
```

### Permission Issues with Mounted Volumes

If you encounter permission issues with mounted volumes:

```bash
# Fix permissions (Linux/Mac)
sudo chown -R $(id -u):$(id -g) ./data ./models

# For Windows, ensure you have full control of these folders
```

## Advanced Docker Usage

### Running in Detached Mode

To run Docker in the background:

```bash
docker-compose up -d
```

### Stopping the Container

To stop the running container:

```bash
docker-compose down
```

### Viewing Logs

To view the logs of a running container:

```bash
docker-compose logs -f
```

## Building Custom Docker Images

You can customize the Dockerfile to create your own version of Apertis:

```Dockerfile
FROM python:3.10-slim

# Add your custom dependencies
RUN pip install your-custom-package

# Copy the Apertis code
COPY . /app
WORKDIR /app

# Install Apertis
RUN pip install -e .

# Your custom commands here
RUN mkdir -p /app/custom_data

# Start Apertis
CMD ["python", "src/apertis_cli.py", "chat", "--web", "--port", "7860"]
```

Build your custom image:

```bash
docker build -t custom-apertis -f CustomDockerfile .
```
