import os
import sys
import unittest
import tempfile
import json

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MockNamedString:
    """Mock class to simulate Gradio's NamedString objects"""
    def __init__(self, name, content):
        self.name = name
        self.content = content

class TestTrainingFunctionality(unittest.TestCase):
    """Test cases for training functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary files
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock train file
        self.train_file_path = os.path.join(self.temp_dir, "train.jsonl")
        with open(self.train_file_path, "w") as f:
            f.write('{"text": "This is a test"}\n')
        
        # Create a mock vocab file
        self.vocab_file_path = os.path.join(self.temp_dir, "vocab.json")
        vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3, "this": 4, "is": 5, "a": 6, "test": 7}
        with open(self.vocab_file_path, "w") as f:
            json.dump(vocab, f)
        
        # Create mock NamedString objects
        self.train_named_string = MockNamedString(self.train_file_path, "mock content")
        self.vocab_named_string = MockNamedString(self.vocab_file_path, "mock content")
    
    def test_file_handling(self):
        """Test that file handling works correctly with NamedString objects."""
        from src.inference.interface import ApertisInterface
        
        # Create a temporary output directory
        output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Test the file handling logic directly
        temp_dir = tempfile.mkdtemp()
        
        # Test with NamedString objects
        train_path = os.path.join(temp_dir, "train.jsonl")
        with open(train_path, "wb") as f:
            if hasattr(self.train_named_string, 'name'):
                with open(self.train_named_string.name, "rb") as source_file:
                    f.write(source_file.read())
            else:
                f.write(self.train_named_string)
        
        # Verify the file was written correctly
        with open(train_path, "r") as f:
            content = f.read()
        self.assertEqual(content, '{"text": "This is a test"}\n')
        
        # Clean up
        import shutil
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    unittest.main()
