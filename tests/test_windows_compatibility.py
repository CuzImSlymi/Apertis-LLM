import os
import sys
import unittest

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestWindowsCompatibility(unittest.TestCase):
    """Test cases for Windows compatibility."""
    
    def test_module_imports(self):
        """Test that all modules can be imported correctly."""
        # Import core modules
        from src.model.core import ApertisConfig, ApertisModel, ApertisForCausalLM, create_apertis_model
        self.assertTrue(True, "Successfully imported core modules")
        
        # Import multimodal modules
        from src.multimodal.module import UnifiedMultimodalEncoder, MultimodalDataProcessor
        self.assertTrue(True, "Successfully imported multimodal modules")
        
        # Import inference modules
        from src.inference.interface import ApertisInterface
        self.assertTrue(True, "Successfully imported inference modules")
        
        # Import training modules
        from src.training.pipeline import YoloStyleTrainingPipeline, ApertisDataset
        self.assertTrue(True, "Successfully imported training modules")
    
    def test_path_handling(self):
        """Test that path handling is correct for Windows."""
        # Check that the src directory is in the Python path
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
        self.assertIn(src_path, [os.path.abspath(p) for p in sys.path if p])
        
        # Check that path handling works with Windows-style paths
        test_path = os.path.join('path', 'to', 'file')
        self.assertTrue(os.path.normpath(test_path))

if __name__ == '__main__':
    unittest.main()
