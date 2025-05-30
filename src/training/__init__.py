import os
import sys
import json
from typing import Dict, Any, Optional
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.training.pipeline import train_from_config as standard_train
from src.training.azr_pipeline import train_from_config as azr_train

def train_from_config(config_path: str, stop_event: Optional[threading.Event] = None):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {e}")
    
    training_method = config.get("training", {}).get("method", "standard")
    
    actual_stop_event = stop_event if stop_event is not None else threading.Event()

    if training_method.lower() == "azr":
        return azr_train(config_path, actual_stop_event)
    else:
        return standard_train(config_path, actual_stop_event)