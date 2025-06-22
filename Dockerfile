# Use a PyTorch base image with CUDA support
# This image includes Python 3.10, PyTorch 2.0.1, CUDA 11.7
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

WORKDIR /app

# System dependencies from the original Dockerfile (git, build-essential) are often included
# or compatible with the pytorch base image. If specific versions are needed, they can be added.
# RUN apt-get update && apt-get install -y git build-essential

# Copy the entire application
COPY . .

# Install dependencies from requirements.txt
# flash-attn will be compiled here, requires CUDA toolkit from the base image.
# Adding --no-cache-dir to reduce image size.
# Using pip directly as in the original file.
RUN pip install --no-cache-dir -r requirements.txt

# Install the package in development mode
RUN pip install -e .

# Create necessary directories
RUN mkdir -p data models

# Create a sample vocabulary file if it doesn't exist
RUN if [ ! -f "data/vocab.json" ]; then \
    echo '{"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "the": 4, "a": 5, "an": 6, "is": 7, "was": 8, "are": 9, "were": 10}' > data/vocab.json; \
    fi

# Verify Python path and package installation (includes checking torch and flash_attn)
RUN python -c "import sys; print(sys.path); import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); import flash_attn; print('flash_attn imported successfully')"

# Create a small test model with proper import path
# Ensure PYTHONPATH is correctly set for this command if run in a new shell context by RUN
# The WORKDIR /app should make modules in /app importable, but being explicit with PYTHONPATH is safer.
RUN PYTHONPATH=/app:$PYTHONPATH python -c "import os; import torch; import sys; sys.path.insert(0, '/app'); from src.multimodal.module import UnifiedMultimodalEncoder; from src.model.core import create_apertis_model; os.makedirs('models', exist_ok=True); model = create_apertis_model(model_size='small', multimodal=True, use_flash_attention=False); torch.save(model.state_dict(), 'models/test_model.pt')"

# Expose the port for the web interface
EXPOSE 7860

# Set the entrypoint
# Ensure PYTHONPATH is correctly set for the CMD execution context.
CMD ["sh", "-c", "PYTHONPATH=/app:$PYTHONPATH python -m src.inference.interface --model-path models/test_model.pt --vocab-file data/vocab.json --web --multimodal"]
