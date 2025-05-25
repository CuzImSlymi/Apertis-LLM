# Apertis-LLM Web Interface Documentation

This document provides an overview of the Apertis-LLM web interface and its features.

## Overview

The Apertis AI Studio web interface provides a user-friendly way to interact with the Apertis-LLM system. It offers several tabs for different functionalities:

1. **Chat**: Interact with your models in a chat interface
2. **Training**: Train models using standard supervised learning
3. **Absolute Zero Reasoner**: Train models using the AZR self-play method
4. **Models**: Load existing models or create new ones

## Chat Tab

The Chat tab allows you to interact with your loaded model in a conversational format.

Features:
- Chat history with user and assistant messages
- Optional image upload for multimodal models
- Adjustable generation parameters (temperature, top-k, top-p)
- Clear chat button to reset the conversation

## Training Tab

The Training tab provides a standard supervised learning interface for training models.

Features:
- Model configuration (size, attention type, multimodal options)
- Data upload (training data, validation data, vocabulary)
- Training parameters (batch size, learning rate, epochs)
- Checkpoint configuration
- GPU selection and memory management
- Weights & Biases integration for tracking

## Absolute Zero Reasoner Tab

The Absolute Zero Reasoner (AZR) tab enables training models through self-play without external data.

Features:
- Model configuration (size, attention type, multimodal options)
- Vocabulary file upload
- Optional seed tasks for jumpstarting training
- AZR-specific training parameters:
  - Number of iterations and tasks per iteration
  - Task generation settings (types, distribution, parameters)
  - Reward configuration (learnability, accuracy, diversity, complexity)
  - Python executor settings for code validation
  - GPU selection and memory management
  - Output directory and W&B logging options

For detailed information about AZR, see the [AZR documentation](azr.md).

## Models Tab

The Models tab allows you to load existing models or create new ones.

Features:
- Load models from local directories or files
- Create new models with customizable parameters
- View model information and configuration

## Using the Interface

### Loading a Model

1. Go to the Models tab
2. Enter the path to your model and vocabulary file
3. Click "Load Model"

### Training a Model

#### Standard Training:
1. Go to the Training tab
2. Configure your model and training parameters
3. Upload your training data and vocabulary
4. Click "Start Training"

#### AZR Training:
1. Go to the Absolute Zero Reasoner tab
2. Configure your model and AZR parameters
3. Upload your vocabulary file (and optional seed tasks)
4. Click "Start AZR Training"

### Chatting with a Model

1. Load a model using the Models tab
2. Go to the Chat tab
3. Type your message and click "Send"
4. For multimodal models, you can upload an image

## Advanced Features

### GPU Management

Both training tabs allow you to select which GPUs to use and how much memory to allocate. This is useful for systems with multiple GPUs or when you want to limit resource usage.

### Weights & Biases Integration

Enable W&B logging to track your training progress in real-time. This provides visualizations and metrics to help you understand how your model is performing.

### Model Creation

The Models tab allows you to create new models from scratch with various configurations. This is useful when you want to start with a fresh model before training.

## Troubleshooting

- If the interface doesn't load, check that the server is running correctly
- If training fails, check the training status output for error messages
- For GPU-related issues, try reducing the memory fraction or using fewer GPUs
- If model loading fails, verify that the model and vocabulary paths are correct
