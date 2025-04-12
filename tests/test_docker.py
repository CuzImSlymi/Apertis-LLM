import os
import sys
import torch
import unittest

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.core import ApertisConfig, ApertisModel, ApertisForCausalLM, create_apertis_model
from src.multimodal.module import UnifiedMultimodalEncoder, MultimodalDataProcessor

class TestDockerSetup(unittest.TestCase):
    """Test cases for Docker setup and module imports."""
    
    def test_module_imports(self):
        """Test that all modules can be imported correctly."""
        # This test will fail if any of the imports above fail
        self.assertTrue(True)
    
    def test_python_path(self):
        """Test that the Python path is set up correctly."""
        # Check that the src directory is in the Python path
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
        self.assertIn(src_path, sys.path)
        
        # Try importing modules using both direct and package imports
        try:
            # Direct import from src
            from src.model.core import ApertisConfig
            self.assertTrue(True)
        except ImportError:
            self.fail("Failed to import using direct path")


if __name__ == '__main__':
    unittest.main()
