import os
import sys
import json
from typing import Dict, Any, Optional

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.training.pipeline import train_from_config as standard_train
from src.training.azr_pipeline import train_from_config as azr_train

def train_from_config(config_path: str):
    """
    Train a model using the specified configuration file.
    Delegates to the appropriate training pipeline based on the method specified in the config.
    """
    # Load the configuration
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    # Determine which training method to use
    training_method = config.get("training", {}).get("method", "standard")
    
    if training_method.lower() == "azr":
        # Use the Absolute Zero Reasoner pipeline
        return azr_train(config_path)
    else:
        # Use the standard training pipeline
        return standard_train(config_path)
