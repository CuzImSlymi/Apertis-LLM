import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Optional, List, Tuple, Dict, Any, Union

import torch.nn as nn

class UnifiedMultimodalEncoder(nn.Module):
    """Unified Multimodal Encoder (UME) for text+image processing."""
    
    def __init__(self, config):
        """
        Initialize the multimodal encoder.
        
        Args:
            config: Configuration object with vision parameters
        """
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.vision_embed_dim = config.vision_embed_dim
        self.vision_patch_size = config.vision_patch_size
        
        # Image transformation pipeline
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Vision encoder components
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.vision_embed_dim,
            kernel_size=self.vision_patch_size,
            stride=self.vision_patch_size,
        )
        
        # Calculate number of patches
        self.num_patches = (self.image_size // self.vision_patch_size) ** 2
        
        # Position embeddings for vision
        self.vision_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.vision_embed_dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vision_embed_dim))
        
        # Vision transformer layers
        vision_heads = getattr(config, 'vision_heads', 12)
        vision_layers = getattr(config, 'vision_layers', 12)
        
        self.vision_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.vision_embed_dim,
                nhead=vision_heads,
                dim_feedforward=self.vision_embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # Pre-normalization for better stability
            )
            for _ in range(vision_layers)
        ])
        
        # Vision layer norm
        self.vision_ln = nn.LayerNorm(self.vision_embed_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the vision encoder."""
        # Initialize position embeddings
        nn.init.normal_(self.vision_pos_embed, std=0.02)
        
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize patch embedding
        nn.init.normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Process image tensors through the vision encoder.
        
        Args:
            pixel_values: Image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Encoded image features
        """
        batch_size = pixel_values.shape[0]
        
        # Create patch embeddings
        patch_embeds = self.patch_embed(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, num_patches, vision_embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.vision_pos_embed
        
        # Apply vision transformer layers
        for layer in self.vision_layers:
            embeddings = layer(embeddings)
        
        # Apply layer norm
        embeddings = self.vision_ln(embeddings)
        
        return embeddings
    
    def process_image(self, image_path: str) -> torch.Tensor:
        """
        Process an image file and return tensor representation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Tensor representation of the image
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Apply transformations
            image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return a blank image as fallback
            return torch.zeros(1, 3, self.image_size, self.image_size)
    
    def process_batch(self, image_paths: List[str]) -> torch.Tensor:
        """
        Process a batch of images and return tensor representations.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Batch of tensor representations
        """
        tensors = []
        for path in image_paths:
            # Remove batch dimension for individual processing
            tensor = self.process_image(path).squeeze(0)
            tensors.append(tensor)
        
        # Stack tensors into a batch
        return torch.stack(tensors)


class MultimodalDataProcessor(nn.Module):
    """Data processor for multimodal inputs (text + images)."""
    
    def __init__(
        self,
        image_size: int = 224,
        max_text_length: int = 512,
        vision_embed_dim: int = 768,
        vision_patch_size: int = 16,
        vision_heads: int = 12,
        vision_layers: int = 12,
        use_cache: bool = True,
    ):
        """
        Initialize the multimodal data processor.
        
        Args:
            image_size: Size of images (height and width)
            max_text_length: Maximum text sequence length
            vision_embed_dim: Dimension of vision embeddings
            vision_patch_size: Size of vision patches
            vision_heads: Number of attention heads in vision transformer
            vision_layers: Number of layers in vision transformer
            use_cache: Whether to cache processed images
        """
        super().__init__()
        self.image_size = image_size
        self.max_text_length = max_text_length
        self.vision_embed_dim = vision_embed_dim
        self.vision_patch_size = vision_patch_size
        self.use_cache = use_cache
        
        # Image cache for efficiency
        self.image_cache = {} if use_cache else None
        
        # Create a config object for the encoder
        class Config:
            pass
        
        config = Config()
        config.image_size = image_size
        config.vision_embed_dim = vision_embed_dim
        config.vision_patch_size = vision_patch_size
        config.vision_heads = vision_heads
        config.vision_layers = vision_layers
        
        # Initialize the multimodal encoder
        self.encoder = UnifiedMultimodalEncoder(config)
        
        # Cross-modal fusion layer
        self.cross_modal_layer = nn.TransformerEncoderLayer(
            d_model=vision_embed_dim,
            nhead=8,
            dim_feedforward=vision_embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        # Output projection
        self.output_projection = nn.Linear(vision_embed_dim, vision_embed_dim)
        self.output_norm = nn.LayerNorm(vision_embed_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for end-to-end processing.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            pixel_values: Image tensor [batch_size, channels, height, width]
            
        Returns:
            Dictionary with processed outputs
        """
        # Process images through vision encoder
        vision_features = self.encoder(pixel_values)
        
        # For now, just return the processed features
        # In a full implementation, this would combine with text features
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "vision_features": vision_features,
            "combined_features": self.output_norm(self.output_projection(vision_features)),
        }
    
    def process_sample(
        self,
        text: str,
        image_path: Optional[str] = None,
        tokenizer: Any = None,
        raw_image: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a single sample with text and optional image.
        
        Args:
            text: Input text
            image_path: Path to image file (optional)
            tokenizer: Tokenizer for text processing
            raw_image: Pre-loaded image tensor (optional)
            
        Returns:
            Dictionary with processed inputs
        """
        # Process text
        if tokenizer is not None:
            # Use provided tokenizer
            tokens = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
        else:
            # Simple fallback tokenization (just for demonstration)
            # In a real implementation, you would use a proper tokenizer
            input_ids = torch.tensor([[0] * self.max_text_length])
            attention_mask = torch.tensor([[1] * self.max_text_length])
        
        # Prepare output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Process image if provided
        if raw_image is not None:
            output["pixel_values"] = raw_image
        elif image_path is not None:
            # Check cache first if enabled
            if self.use_cache and image_path in self.image_cache:
                output["pixel_values"] = self.image_cache[image_path]
            else:
                pixel_values = self.encoder.process_image(image_path)
                output["pixel_values"] = pixel_values
                
                # Cache the result if enabled
                if self.use_cache:
                    self.image_cache[image_path] = pixel_values
        
        return output
    
    def process_batch(
        self,
        texts: List[str],
        image_paths: Optional[List[str]] = None,
        tokenizer: Any = None,
        raw_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of samples with text and optional images.
        
        Args:
            texts: List of input texts
            image_paths: List of paths to image files (optional)
            tokenizer: Tokenizer for text processing
            raw_images: Pre-loaded image tensors (optional)
            
        Returns:
            Dictionary with processed batch inputs
        """
        # Process text
        if tokenizer is not None:
            # Use provided tokenizer
            tokens = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            input_ids = tokens["input_ids"]
            attention_mask = tokens["attention_mask"]
        else:
            # Simple fallback tokenization (just for demonstration)
            batch_size = len(texts)
            input_ids = torch.tensor([[0] * self.max_text_length] * batch_size)
            attention_mask = torch.tensor([[1] * self.max_text_length] * batch_size)
        
        # Prepare output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Process images if provided
        if raw_images is not None:
            output["pixel_values"] = raw_images
        elif image_paths is not None:
            # Check cache first for each image if enabled
            if self.use_cache:
                cached_images = []
                uncached_paths = []
                uncached_indices = []
                
                for i, path in enumerate(image_paths):
                    if path in self.image_cache:
                        cached_images.append((i, self.image_cache[path]))
                    else:
                        uncached_paths.append(path)
                        uncached_indices.append(i)
                
                # Process uncached images
                if uncached_paths:
                    uncached_tensors = self.encoder.process_batch(uncached_paths)
                    
                    # Update cache
                    for i, path in enumerate(uncached_paths):
                        self.image_cache[path] = uncached_tensors[i].unsqueeze(0)
                
                # Combine cached and newly processed images
                batch_size = len(image_paths)
                pixel_values = torch.zeros(
                    batch_size, 3, self.image_size, self.image_size,
                    device=input_ids.device
                )
                
                # Add cached images
                for idx, tensor in cached_images:
                    pixel_values[idx] = tensor.squeeze(0)
                
                # Add newly processed images
                for i, orig_idx in enumerate(uncached_indices):
                    pixel_values[orig_idx] = uncached_tensors[i]
            else:
                # Process all images without caching
                pixel_values = self.encoder.process_batch(image_paths)
            
            output["pixel_values"] = pixel_values
        
        return output
        
    def clear_cache(self):
        """Clear the image cache to free memory."""
        if self.use_cache:
            self.image_cache = {}


def create_sample_image(output_path: str, size: int = 224) -> None:
    """
    Create a sample image for testing multimodal capabilities.
    
    Args:
        output_path: Path to save the image
        size: Size of the image (height and width)
    """
    try:
        # Create a simple gradient image
        array = np.zeros((size, size, 3), dtype=np.uint8)
        for i in range(size):
            for j in range(size):
                array[i, j, 0] = int(255 * i / size)  # Red channel
                array[i, j, 1] = int(255 * j / size)  # Green channel
                array[i, j, 2] = int(255 * (i + j) / (2 * size))  # Blue channel
        
        # Convert to PIL Image and save
        image = Image.fromarray(array)
        image.save(output_path)
        
        print(f"Created sample image at {output_path}")
    except Exception as e:
        print(f"Error creating sample image: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apertis Multimodal Module")
    parser.add_argument("--create-sample", type=str, help="Create a sample image at the specified path")
    parser.add_argument("--process-image", type=str, help="Process an image and print tensor shape")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_image(args.create_sample)
    
    if args.process_image:
        # Create a simple processor and process the image
        processor = MultimodalDataProcessor()
        result = processor.process_sample("Sample text", args.process_image)
        
        # Print tensor shapes
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
