import os
import sys
import torch
import unittest

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.core import ApertisConfig, ApertisModel, ApertisForCausalLM, create_apertis_model
from src.multimodal.module import UnifiedMultimodalEncoder, MultimodalDataProcessor

class TestApertisModel(unittest.TestCase):
    """Test cases for the Apertis model architecture."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a small config for testing
        self.config = ApertisConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            multimodal=True,
            image_size=112,
            vision_embed_dim=128,
            vision_patch_size=16,
            vision_layers=2,
            vision_heads=4,
        )
        
        # Create temporary directory for test artifacts
        os.makedirs('test_outputs', exist_ok=True)
    
    def test_model_creation(self):
        """Test that models can be created without errors."""
        # Create base model
        model = ApertisModel(self.config)
        self.assertIsInstance(model, ApertisModel)
        
        # Create causal LM model
        causal_model = ApertisForCausalLM(self.config)
        self.assertIsInstance(causal_model, ApertisForCausalLM)
        
        # Test factory function
        factory_model = create_apertis_model(model_size='small', multimodal=True)
        self.assertIsInstance(factory_model, ApertisForCausalLM)
    
    def test_load_state_dict(self):
        """Test that load_state_dict works correctly."""
        # Create model and get state dict
        model = ApertisModel(self.config)
        state_dict = model.state_dict()
        
        # Create a new model and load the state dict
        new_model = ApertisModel(self.config)
        missing_keys, unexpected_keys = new_model.load_state_dict(state_dict)
        
        # Check that there are no missing or unexpected keys
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)
        
        # Test with causal LM model
        causal_model = ApertisForCausalLM(self.config)
        causal_state_dict = causal_model.state_dict()
        
        new_causal_model = ApertisForCausalLM(self.config)
        missing_keys, unexpected_keys = new_causal_model.load_state_dict(causal_state_dict)
        
        self.assertEqual(len(missing_keys), 0)
        self.assertEqual(len(unexpected_keys), 0)
    
    def test_save_and_load(self):
        """Test saving and loading model weights."""
        # Create model
        model = create_apertis_model(model_size='small', multimodal=True)
        
        # Save model weights
        save_path = os.path.join('test_outputs', 'test_model.pt')
        torch.save(model.state_dict(), save_path)
        
        # Load model weights
        new_model = create_apertis_model(model_size='small', multimodal=True)
        new_model.load_state_dict(torch.load(save_path))
        
        # Check that the model loaded correctly
        self.assertTrue(os.path.exists(save_path))
    
    def test_forward_pass(self):
        """Test forward pass through the model."""
        # Create model
        model = ApertisForCausalLM(self.config)
        
        # Create dummy input
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        
        # Check output shapes
        self.assertEqual(outputs[1].shape, (batch_size, seq_length, self.config.vocab_size))
    
    def test_multimodal_forward(self):
        """Test forward pass with multimodal inputs."""
        # Create model
        model = ApertisForCausalLM(self.config)
        
        # Create dummy inputs
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.rand(batch_size, 3, self.config.image_size, self.config.image_size)
        
        # Run forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
        
        # Check output shapes
        self.assertEqual(outputs[1].shape, (batch_size, seq_length, self.config.vocab_size))
    
    def test_multimodal_processor(self):
        """Test the multimodal data processor."""
        # Create processor
        processor = MultimodalDataProcessor(
            image_size=self.config.image_size,
            vision_embed_dim=self.config.vision_embed_dim,
            vision_patch_size=self.config.vision_patch_size,
        )
        
        # Create dummy inputs
        batch_size = 2
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, 10))
        attention_mask = torch.ones_like(input_ids)
        pixel_values = torch.rand(batch_size, 3, self.config.image_size, self.config.image_size)
        
        # Run forward pass
        with torch.no_grad():
            outputs = processor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
            )
        
        # Check output shapes
        self.assertIn('vision_features', outputs)
        self.assertIn('combined_features', outputs)
    
    def tearDown(self):
        """Clean up test fixtures."""
        # Remove test artifacts
        if os.path.exists('test_outputs/test_model.pt'):
            os.remove('test_outputs/test_model.pt')
        
        if os.path.exists('test_outputs'):
            os.rmdir('test_outputs')


if __name__ == '__main__':
    unittest.main()
