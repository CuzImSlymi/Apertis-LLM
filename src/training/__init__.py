import os
import sys
import json
from typing import Dict, Any, Optional
import threading

# The UI/CLI layer will be responsible for choosing the correct training function.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .pipeline import train_from_config
from .azr_pipeline import train_from_config as azr_train