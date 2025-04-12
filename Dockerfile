# Use a specific Python version for consistency
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire application at once
# This ensures all files are available during installation
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode
# Using pip directly instead of python -m pip for better compatibility
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data models

# Create a sample vocabulary file if it doesn't exist
RUN if [ ! -f "data/vocab.json" ]; then \
    echo '{"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "the": 4, "a": 5, "an": 6, "is": 7, "was": 8, "are": 9, "were": 10}' > data/vocab.json; \
    fi

# Verify Python path and package installation
RUN python -c "import sys; print(sys.path)"

# Create a small test model with proper import path
RUN PYTHONPATH=/app python -c "import os; import torch; import sys; sys.path.insert(0, '/app'); from src.multimodal.module import UnifiedMultimodalEncoder; from src.model.core import create_apertis_model; os.makedirs('models', exist_ok=True); model = create_apertis_model(model_size='small', multimodal=True); torch.save(model.state_dict(), 'models/test_model.pt')"

# Expose the port for the web interface
EXPOSE 7860

# Set the entrypoint to run the web interface with correct Python path
CMD ["sh", "-c", "PYTHONPATH=/app python -m src.inference.interface --model-path models/test_model.pt --vocab-file data/vocab.json --web --multimodal --server-name 0.0.0.0"]
