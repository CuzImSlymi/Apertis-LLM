import os
import sys
import time
import torch
import unittest

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model

class TestEfficiency(unittest.TestCase):
    """Test cases for model efficiency optimizations."""
    
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
    
    def test_inference_speed(self):
        """Test inference speed with and without caching."""
        # Create model
        model = ApertisForCausalLM(self.config)
        model.eval()  # Set to evaluation mode
        
        # Create dummy input
        batch_size = 1
        seq_length = 20
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Warm-up run
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Test without caching
        start_time = time.time()
        with torch.no_grad():
            for _ in range(5):
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        no_cache_time = time.time() - start_time
        
        # Test with caching
        start_time = time.time()
        with torch.no_grad():
            past_key_values = None
            for i in range(5):
                outputs = model(
                    input_ids=input_ids if i == 0 else input_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                past_key_values = outputs[2]
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)],
                    dim=-1
                )
        cache_time = time.time() - start_time
        
        # Check that caching is faster
        print(f"No cache time: {no_cache_time:.4f}s, Cache time: {cache_time:.4f}s")
        self.assertLess(cache_time, no_cache_time)
    
    def test_chunked_attention(self):
        """Test that chunked attention works correctly."""
        # Create models with different chunk sizes
        config_no_chunk = self.config
        config_no_chunk.sliding_window = None
        
        config_with_chunk = self.config
        config_with_chunk.sliding_window = 10
        
        model_no_chunk = ApertisForCausalLM(config_no_chunk)
        model_with_chunk = ApertisForCausalLM(config_with_chunk)
        
        # Create dummy input
        batch_size = 1
        seq_length = 20
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))
        attention_mask = torch.ones_like(input_ids)
        
        # Run forward pass
        with torch.no_grad():
            outputs_no_chunk = model_no_chunk(input_ids=input_ids, attention_mask=attention_mask)
            outputs_with_chunk = model_with_chunk(input_ids=input_ids, attention_mask=attention_mask)
        
        # Check that outputs have the same shape
        self.assertEqual(outputs_no_chunk[1].shape, outputs_with_chunk[1].shape)


if __name__ == '__main__':
    unittest.main()
