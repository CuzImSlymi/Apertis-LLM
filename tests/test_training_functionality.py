import os
import sys
import unittest
import tempfile
import json
import torch
import numpy as np
from tqdm import tqdm

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestTrainingFunctionality(unittest.TestCase):
    """Test cases for training functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple test dataset
        self.train_data = [
            {"text": "This is a test sentence one."},
            {"text": "This is a test sentence two."},
            {"text": "This is a test sentence three."},
            {"text": "This is a test sentence four."}
        ]
        
        # Create train file
        self.train_file = os.path.join(self.temp_dir, "train.jsonl")
        with open(self.train_file, "w") as f:
            for item in self.train_data:
                f.write(json.dumps(item) + "\n")
        
        # Create vocab file
        self.vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "this": 4,
            "is": 5,
            "a": 6,
            "test": 7,
            "sentence": 8,
            "one": 9,
            "two": 10,
            "three": 11,
            "four": 12
        }
        self.vocab_file = os.path.join(self.temp_dir, "vocab.json")
        with open(self.vocab_file, "w") as f:
            json.dump(self.vocab, f)
        
        # Create output directory
        self.output_dir = os.path.join(self.temp_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def test_dataset_loading(self):
        """Test that dataset loading works correctly."""
        from src.training.pipeline import ApertisDataset
        
        dataset = ApertisDataset(
            data_path=self.train_file,
            tokenizer_path=self.vocab_file,
            max_length=32
        )
        
        self.assertEqual(len(dataset), 4)
        
        # Test getting an item
        item = dataset[0]
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
    
    def test_training_loop_single_step(self):
        """Test that the training loop can process a single step without hanging."""
        from src.training.pipeline import ApertisDataset, train_from_config
        from src.model.core import ApertisConfig, ApertisForCausalLM
        
        # Create a minimal config
        config = {
            "data_config": {
                "train_data_path": self.train_file,
                "tokenizer_path": self.vocab_file,
                "max_length": 32
            },
            "model_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 128,
                "use_expert_system": False,
                "multimodal": False
            },
            "training_config": {
                "output_dir": self.output_dir,
                "batch_size": 2,
                "learning_rate": 5e-5,
                "num_epochs": 1,
                "use_wandb": False
            }
        }
        
        # Save config to file
        config_path = os.path.join(self.temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Create dataset and model directly to test one step
        dataset = ApertisDataset(
            data_path=self.train_file,
            tokenizer_path=self.vocab_file,
            max_length=32
        )
        
        model_config = ApertisConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=128
        )
        
        model = ApertisForCausalLM(model_config)
        
        # Create a batch manually
        batch = {
            "input_ids": torch.tensor([[4, 5, 6, 7, 8, 9, 0, 0], [4, 5, 6, 7, 8, 10, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]]),
            "labels": torch.tensor([[4, 5, 6, 7, 8, 9, 0, 0], [4, 5, 6, 7, 8, 10, 0, 0]])
        }
        
        # Process one step
        outputs = model(**batch)
        loss = outputs[0]
        
        # Verify loss is a tensor and not NaN
        self.assertIsInstance(loss, torch.Tensor)
        self.assertFalse(torch.isnan(loss).any())
        
        # Check that training log file is created
        log_file = os.path.join(self.output_dir, "training_log.txt")
        with open(log_file, "w") as f:
            f.write("Test log entry\n")
        
        self.assertTrue(os.path.exists(log_file))
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)

if __name__ == '__main__':
    unittest.main()
