import os
import sys
import json
import torch
import gradio as gr
import numpy as np
from PIL import Image
import tempfile
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time
from pathlib import Path

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import Apertis modules
from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model
from src.training.pipeline import YoloStyleTrainingPipeline, ApertisDataset, get_available_gpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApertisInterface:
    """Google AI Studio-like interface for Apertis."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        multimodal: bool = False,
        device: Optional[str] = None,
        web: bool = False,
        port: int = 7860,
        share: bool = False,
    ):
        """
        Initialize the interface.
        
        Args:
            model_path: Path to the model file
            vocab_file: Path to the vocabulary file
            multimodal: Whether to use multimodal capabilities
            device: Device to use for inference
            web: Whether to launch the web interface
            port: Port for the web interface
            share: Whether to create a public link
        """
        self.model_path = model_path
        self.vocab_file = vocab_file
        self.multimodal = multimodal
        self.web = web
        self.port = port
        self.share = share
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and vocabulary if provided
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.token_mapping = None  # For handling vocabulary mismatches
        
        if model_path is not None and vocab_file is not None:
            self.load_model(model_path)
            self.load_vocabulary(vocab_file)
        
        # Initialize chat history
        self.chat_history = []
        
        # Launch web interface if requested
        if web:
            self.launch_web_interface()
    
    def load_model(self, model_path: str) -> None:
        """Load model from file."""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Check if path is a directory (saved model) or a file (state dict)
            if os.path.isdir(model_path):
                # Load configuration
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    config = ApertisConfig.from_dict(config_dict)
                    self.model = ApertisForCausalLM(config)
                    
                    # Load state dict
                    state_dict_path = os.path.join(model_path, "model.pt")
                    if os.path.exists(state_dict_path):
                        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device, weights_only=True))
                    else:
                        logger.warning(f"Model state dict not found at {state_dict_path}")
                else:
                    logger.warning(f"Model configuration not found at {config_path}")
                    # Create a default model as fallback
                    self.model = create_apertis_model(model_size="small", multimodal=self.multimodal)
            else:
                # Assume it's a state dict file
                # First try to determine the model size, architecture, and vocabulary size from the state dict
                try:
                    # Load state dict to examine its structure
                    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                    
                    # Check if model uses expert system
                    use_expert_system = False
                    # Look for expert system specific parameters
                    for key in state_dict.keys():
                        if "feed_forward.feed_forward.experts" in key or "feed_forward.feed_forward.router" in key:
                            use_expert_system = True
                            break
                    
                    # Extract vocabulary size from embeddings
                    vocab_size = None
                    if "model.token_embeddings.weight" in state_dict:
                        vocab_size = state_dict["model.token_embeddings.weight"].size(0)
                        logger.info(f"Detected vocabulary size: {vocab_size}")
                    
                    # Determine model size from hidden dimensions
                    hidden_size = None
                    layer_count = 0
                    if "model.token_embeddings.weight" in state_dict:
                        hidden_size = state_dict["model.token_embeddings.weight"].size(1)
                        
                        # Count number of layers
                        i = 0
                        # Check for different possible layer parameter names
                        while any(f"model.layers.{i}.{suffix}" in state_dict for suffix in 
                                ["input_layernorm.weight", "attn_norm.weight", 
                                 "attention.layer_norm.weight", "feed_forward.layer_norm.weight"]):
                            layer_count += 1
                            i += 1
                        
                        # Determine model size based on hidden_size and layer_count
                        if hidden_size == 512 and layer_count <= 8:
                            model_size = "small"
                        elif hidden_size == 768 and layer_count <= 12:
                            model_size = "base"
                        elif hidden_size == 1024 or layer_count > 12:
                            model_size = "large"
                        else:
                            # Default to base if dimensions don't match standard sizes
                            model_size = "base"
                            
                        logger.info(f"Detected model size: {model_size} (hidden_size={hidden_size}, layers={layer_count})")
                        logger.info(f"Expert system: {use_expert_system}")
                    else:
                        # If can't determine, default to base
                        model_size = "base"
                        logger.info(f"Could not determine model size, defaulting to {model_size}")
                    
                    # Create a custom configuration with the detected parameters
                    config_kwargs = {
                        "hidden_size": hidden_size if hidden_size else {"small": 512, "base": 768, "large": 1024}[model_size],
                        "num_hidden_layers": layer_count if layer_count > 0 else {"small": 8, "base": 12, "large": 24}[model_size],
                        "use_expert_system": use_expert_system,
                        "multimodal": self.multimodal,
                    }
                    
                    # Add vocabulary size if detected
                    if vocab_size:
                        config_kwargs["vocab_size"] = vocab_size
                    
                    # Create model with the custom configuration
                    config = ApertisConfig(**config_kwargs)
                    self.model = ApertisForCausalLM(config)
                    
                    # Load the state dict
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Successfully loaded model with custom configuration")
                except Exception as inner_e:
                    logger.warning(f"Error determining model parameters: {inner_e}")
                    logger.warning("Trying to load with exact vocabulary size and layer count...")
                    
                    # Try loading with exact vocabulary size and layer count
                    try:
                        # Load state dict again
                        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                        
                        # Extract vocabulary size
                        vocab_size = state_dict["model.token_embeddings.weight"].size(0)
                        hidden_size = state_dict["model.token_embeddings.weight"].size(1)
                        
                        # Count layers
                        layer_count = 0
                        i = 0
                        while any(f"model.layers.{i}.{suffix}" in state_dict for suffix in 
                                ["input_layernorm.weight", "attn_norm.weight", 
                                 "attention.layer_norm.weight", "feed_forward.layer_norm.weight"]):
                            layer_count += 1
                            i += 1
                        
                        # Check for expert system
                        use_expert_system = any("feed_forward.feed_forward.experts" in key for key in state_dict.keys())
                        
                        # Create custom configuration
                        config = ApertisConfig(
                            vocab_size=vocab_size,
                            hidden_size=hidden_size,
                            num_hidden_layers=layer_count,
                            num_attention_heads=hidden_size // 64,  # Common ratio
                            intermediate_size=hidden_size * 4,      # Common ratio
                            use_expert_system=use_expert_system,
                            multimodal=self.multimodal
                        )
                        
                        # Create model with exact parameters
                        self.model = ApertisForCausalLM(config)
                        
                        # Load state dict
                        self.model.load_state_dict(state_dict, strict=False)
                        logger.info(f"Successfully loaded model with exact parameters (vocab={vocab_size}, layers={layer_count})")
                    except Exception as exact_e:
                        logger.warning(f"Error loading with exact parameters: {exact_e}")
                        logger.warning("Falling back to loading with relaxed constraints...")
                        
                        # Final fallback: try loading with strict=False
                        try:
                            # Load state dict again
                            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                            
                            # Extract vocabulary size and hidden size
                            if "model.token_embeddings.weight" in state_dict:
                                vocab_size = state_dict["model.token_embeddings.weight"].size(0)
                                hidden_size = state_dict["model.token_embeddings.weight"].size(1)
                            else:
                                # Default values if can't determine
                                vocab_size = 32000
                                hidden_size = 768
                            
                            # Create model with detected parameters but default layer count
                            config = ApertisConfig(
                                vocab_size=vocab_size,
                                hidden_size=hidden_size,
                                use_expert_system=True,  # Try with expert system as last resort
                                multimodal=self.multimodal
                            )
                            
                            self.model = ApertisForCausalLM(config)
                            
                            # Load with strict=False to allow parameter mismatches
                            self.model.load_state_dict(state_dict, strict=False)
                            logger.info(f"Loaded model with relaxed constraints (strict=False)")
                        except Exception as fallback_e:
                            logger.error(f"All loading attempts failed: {fallback_e}")
                            # Create a minimal model as absolute fallback
                            self.model = create_apertis_model(
                                model_size="small", 
                                multimodal=self.multimodal,
                                use_expert_system=False
                            )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a default model as fallback
            logger.info("Creating default model as fallback")
            self.model = create_apertis_model(model_size="base", multimodal=self.multimodal)
            self.model.to(self.device)
            self.model.eval()
    
    def load_vocabulary(self, vocab_file: str) -> None:
        """Load vocabulary from file."""
        try:
            logger.info(f"Loading vocabulary from {vocab_file}")
            
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)
            
            # Create reverse vocabulary for decoding
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            logger.info(f"Vocabulary loaded with {len(self.vocab)} tokens")
            
            # Check if model vocabulary size matches loaded vocabulary
            if self.model and hasattr(self.model.config, "vocab_size"):
                model_vocab_size = self.model.config.vocab_size
                vocab_size = len(self.vocab)
                
                if model_vocab_size != vocab_size:
                    logger.warning(f"Model vocabulary size ({model_vocab_size}) doesn't match loaded vocabulary size ({vocab_size})")
                    
                    # Create a token mapping to handle the mismatch
                    self.create_token_mapping(model_vocab_size, vocab_size)
                    
                    # Inform user about the automatic adaptation
                    logger.info("Created token mapping to handle vocabulary size mismatch")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            # Create a minimal vocabulary as fallback
            logger.info("Creating minimal vocabulary as fallback")
            self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def create_token_mapping(self, model_vocab_size: int, loaded_vocab_size: int) -> None:
        """Create a mapping between model tokens and loaded vocabulary tokens."""
        try:
            # Initialize token mapping dictionary
            self.token_mapping = {}
            
            # Special tokens should map directly if they exist in both vocabularies
            special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
            for token in special_tokens:
                if token in self.vocab:
                    model_idx = min(self.vocab[token], model_vocab_size - 1)
                    self.token_mapping[self.vocab[token]] = model_idx
            
            # For regular tokens, create a mapping strategy
            if model_vocab_size > loaded_vocab_size:
                # Model has more tokens than loaded vocabulary
                # Map loaded vocab tokens to same indices in model vocab
                for token_id in self.reverse_vocab:
                    if token_id not in self.token_mapping:
                        if token_id < model_vocab_size:
                            self.token_mapping[token_id] = token_id
                        else:
                            # For tokens beyond model vocab size, map to <unk>
                            self.token_mapping[token_id] = self.vocab.get("<unk>", 3)
            else:
                # Model has fewer tokens than loaded vocabulary
                # Map loaded vocab tokens to model vocab, with overflow going to <unk>
                for token_id in self.reverse_vocab:
                    if token_id not in self.token_mapping:
                        if token_id < model_vocab_size:
                            self.token_mapping[token_id] = token_id
                        else:
                            # For tokens beyond model vocab size, map to <unk>
                            self.token_mapping[token_id] = self.vocab.get("<unk>", 3)
            
            logger.info(f"Created token mapping between model vocabulary ({model_vocab_size} tokens) and loaded vocabulary ({loaded_vocab_size} tokens)")
        except Exception as e:
            logger.error(f"Error creating token mapping: {e}")
            self.token_mapping = None
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded vocabulary."""
        if self.vocab is None:
            logger.warning("Vocabulary not loaded, using minimal tokenization")
            return [3] * len(text.split())  # All tokens as <unk>
        
        # Simple tokenization by splitting on spaces
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab.get("<unk>", 3))
        
        # Apply token mapping if it exists and there's a vocabulary mismatch
        if self.token_mapping is not None:
            mapped_tokens = []
            for token in tokens:
                if token in self.token_mapping:
                    mapped_tokens.append(self.token_mapping[token])
                else:
                    # If token not in mapping, use <unk>
                    mapped_tokens.append(self.vocab.get("<unk>", 3))
            return mapped_tokens
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        if self.reverse_vocab is None:
            logger.warning("Vocabulary not loaded, cannot detokenize")
            return "[Unable to detokenize without vocabulary]"
        
        # Apply reverse token mapping if it exists
        if self.token_mapping is not None:
            # Create reverse mapping (model token -> vocab token)
            reverse_mapping = {}
            for vocab_token, model_token in self.token_mapping.items():
                if model_token not in reverse_mapping:
                    reverse_mapping[model_token] = vocab_token
            
            # Apply reverse mapping
            mapped_token_ids = []
            for token_id in token_ids:
                if token_id in reverse_mapping:
                    mapped_token_ids.append(reverse_mapping[token_id])
                else:
                    # If no mapping exists, keep original token
                    mapped_token_ids.append(token_id)
            
            token_ids = mapped_token_ids
        
        # Convert token IDs to words
        words = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                word = self.reverse_vocab[token_id]
                # Skip special tokens
                if word not in ["<pad>", "<bos>", "<eos>", "<unk>"]:
                    words.append(word)
            else:
                words.append("<unk>")
        
        return " ".join(words)
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        try:
            from torchvision import transforms
            
            # Define image transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return a blank image as fallback
            return torch.zeros(1, 3, 224, 224)
    
    def generate_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response to the given prompt."""
        if self.model is None:
            return "Model not loaded. Please load a model first."
        
        try:
            # Tokenize prompt
            input_ids = self.tokenize(prompt)
            
            # Add <bos> token if not present
            if len(input_ids) == 0 or input_ids[0] != self.vocab.get("<bos>", 1):
                input_ids = [self.vocab.get("<bos>", 1)] + input_ids
            
            # Convert to tensor
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Create attention mask (all 1s for input tokens)
            attention_mask = torch.ones_like(input_ids)
            
            # Prepare image if provided and model is multimodal
            pixel_values = None
            if image_path is not None and self.multimodal:
                pixel_values = self.preprocess_image(image_path).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_length=max_length,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            
            # Detokenize response (excluding input tokens)
            response_ids = output_ids[0, len(input_ids[0]):].tolist()
            response = self.detokenize(response_ids)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat(
        self,
        message: str,
        image_path: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Chat with the model and maintain conversation history."""
        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        
        # Create prompt from chat history
        prompt = ""
        for entry in self.chat_history:
            if entry["role"] == "user":
                prompt += f"User: {entry['content']}\n"
            else:
                prompt += f"Assistant: {entry['content']}\n"
        
        prompt += "Assistant: "
        
        # Generate response
        response = self.generate_response(
            prompt=prompt,
            image_path=image_path,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset_chat(self) -> None:
        """Reset chat history."""
        self.chat_history = []
    
    def launch_web_interface(self) -> None:
        """Launch a Google AI Studio-like web interface."""
        logger.info("Launching web interface")
        
        # Define interface components
        with gr.Blocks(title="Apertis AI Studio", theme=gr.themes.Soft()) as interface:
            # Header
            with gr.Row():
                gr.Markdown("# Apertis AI Studio")
            
            # Tabs for different functionalities
            with gr.Tabs():
                # Chat tab
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            chatbot = gr.Chatbot(height=500)
                            with gr.Row():
                                message = gr.Textbox(
                                    placeholder="Type your message here...",
                                    show_label=False,
                                )
                                submit_btn = gr.Button("Send")
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")
                        
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="filepath",
                                label="Upload Image (optional)",
                            )
                            with gr.Accordion("Generation Settings", open=False):
                                max_length = gr.Slider(
                                    minimum=10,
                                    maximum=500,
                                    value=100,
                                    step=10,
                                    label="Max Length",
                                )
                                temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.5,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature",
                                )
                                top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="Top K",
                                )
                                top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.1,
                                    label="Top P",
                                )
                
                # Training tab
                with gr.TabItem("Training"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Configuration")
                            model_size = gr.Dropdown(
                                choices=["small", "base", "large"],
                                value="base",
                                label="Model Size",
                            )
                            multimodal_checkbox = gr.Checkbox(
                                value=self.multimodal,
                                label="Multimodal (Text + Image)",
                            )
                            use_expert_system = gr.Checkbox(
                                value=False,
                                label="Use Adaptive Expert System",
                            )
                            
                            gr.Markdown("## Training Data")
                            train_data = gr.File(
                                label="Training Data (JSONL)",
                                file_types=[".jsonl"],
                            )
                            val_data = gr.File(
                                label="Validation Data (JSONL, optional)",
                                file_types=[".jsonl"],
                            )
                            vocab_data = gr.File(
                                label="Vocabulary File (JSON)",
                                file_types=[".json"],
                            )
                            image_dir = gr.Textbox(
                                label="Image Directory (for multimodal training)",
                                placeholder="/path/to/images",
                                visible=self.multimodal,
                            )
                            
                            # Update image_dir visibility based on multimodal checkbox
                            multimodal_checkbox.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[multimodal_checkbox],
                                outputs=[image_dir],
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Training Parameters")
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=64,
                                value=4,
                                step=1,
                                label="Batch Size",
                            )
                            learning_rate = gr.Slider(
                                minimum=1e-6,
                                maximum=1e-3,
                                value=5e-5,
                                step=1e-6,
                                label="Learning Rate",
                            )
                            num_epochs = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=3,
                                step=1,
                                label="Number of Epochs",
                            )
                            
                            # Add checkpoint frequency controls
                            with gr.Accordion("Checkpoint Settings", open=True):
                                checkpoint_steps = gr.Slider(
                                    minimum=0,
                                    maximum=5000,
                                    value=1000,
                                    step=100,
                                    label="Global Step Checkpoint Frequency (0 to disable)",
                                )
                                iteration_checkpoint_steps = gr.Slider(
                                    minimum=0,
                                    maximum=1000,
                                    value=100,
                                    step=10,
                                    label="Iteration Checkpoint Frequency (0 to disable)",
                                )
                                gr.Markdown("""
                                * **Global Step Checkpoint**: Save checkpoint every N global steps (across epochs)
                                * **Iteration Checkpoint**: Save checkpoint every N iterations within each epoch
                                """)
                            
                            # Add GPU management controls
                            with gr.Accordion("GPU Settings", open=True):
                                # Get available GPUs
                                available_gpus = get_available_gpus()
                                gpu_info_text = ""
                                gpu_choices = []
                                
                                if available_gpus:
                                    gpu_info_text = "### Available GPUs:\n"
                                    for gpu in available_gpus:
                                        gpu_info_text += f"- GPU {gpu['id']}: {gpu['name']} ({gpu['total_memory']:.1f} GB)\n"
                                        gpu_choices.append(str(gpu['id']))
                                else:
                                    gpu_info_text = "No CUDA-capable GPUs detected."
                                
                                gr.Markdown(gpu_info_text)
                                
                                # GPU selection
                                if gpu_choices:
                                    gpu_selection = gr.CheckboxGroup(
                                        choices=gpu_choices,
                                        value=[gpu_choices[0]],  # Default to first GPU
                                        label="Select GPUs for Training",
                                    )
                                    
                                    distributed_training = gr.Checkbox(
                                        value=False,
                                        label="Use Distributed Training (recommended for multiple GPUs)",
                                    )
                                    
                                    # Only show distributed training option if multiple GPUs are selected
                                    def update_distributed_visibility(selected_gpus):
                                        return gr.update(visible=len(selected_gpus) > 1)
                                    
                                    gpu_selection.change(
                                        fn=update_distributed_visibility,
                                        inputs=[gpu_selection],
                                        outputs=[distributed_training],
                                    )
                                else:
                                    # If no GPUs available, show a message
                                    gr.Markdown("Training will use CPU as no GPUs are available.")
                                    # Create hidden elements to avoid errors
                                    gpu_selection = gr.CheckboxGroup(
                                        choices=["0"],
                                        value=[],
                                        label="Select GPUs for Training",
                                        visible=False,
                                    )
                                    distributed_training = gr.Checkbox(
                                        value=False,
                                        label="Use Distributed Training",
                                        visible=False,
                                    )
                                
                                gpu_memory_fraction = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.7,
                                    step=0.05,
                                    label="GPU Memory Fraction (per GPU)",
                                )
                            
                            output_dir = gr.Textbox(
                                label="Output Directory",
                                value="output",
                            )
                            use_wandb = gr.Checkbox(
                                value=False,
                                label="Use Weights & Biases for Logging",
                            )
                            wandb_project = gr.Textbox(
                                label="Weights & Biases Project",
                                value="apertis",
                                visible=False,
                            )
                            
                            # Update wandb_project visibility based on use_wandb checkbox
                            use_wandb.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[use_wandb],
                                outputs=[wandb_project],
                            )
                            
                            train_btn = gr.Button("Start Training")
                            training_output = gr.Textbox(
                                label="Training Output",
                                interactive=False,
                                lines=10,
                            )
                
                # Model tab
                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model")
                            model_path_input = gr.Textbox(
                                label="Model Path",
                                value=self.model_path or "",
                                placeholder="/path/to/model.pt or /path/to/model_dir",
                            )
                            vocab_path_input = gr.Textbox(
                                label="Vocabulary Path",
                                value=self.vocab_file or "",
                                placeholder="/path/to/vocab.json",
                            )
                            
                            # Add advanced model loading options
                            with gr.Accordion("Advanced Loading Options", open=True):
                                expert_system_checkbox = gr.Checkbox(
                                    value=False,
                                    label="Model uses Expert System",
                                )
                                custom_vocab_size = gr.Number(
                                    value=0,
                                    label="Custom Vocabulary Size (0 to auto-detect)",
                                    precision=0
                                )
                                custom_layer_count = gr.Number(
                                    value=0,
                                    label="Custom Layer Count (0 to auto-detect)",
                                    precision=0
                                )
                                strict_loading = gr.Checkbox(
                                    value=True,
                                    label="Strict Loading (uncheck to allow parameter mismatches)",
                                )
                                adapt_vocab_checkbox = gr.Checkbox(
                                    value=True,
                                    label="Auto-adapt to vocabulary mismatches",
                                )
                            
                            load_model_btn = gr.Button("Load Model")
                            model_info = gr.Textbox(
                                label="Model Information",
                                interactive=False,
                                lines=10,
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Create New Model")
                            new_model_size = gr.Dropdown(
                                choices=["small", "base", "large"],
                                value="base",
                                label="Model Size",
                            )
                            new_model_multimodal = gr.Checkbox(
                                value=False,
                                label="Multimodal (Text + Image)",
                            )
                            new_model_expert = gr.Checkbox(
                                value=False,
                                label="Use Expert System",
                            )
                            new_model_vocab_size = gr.Number(
                                value=32000,
                                label="Vocabulary Size",
                                precision=0
                            )
                            new_model_output = gr.Textbox(
                                label="Output Path",
                                value="models/new_model",
                                placeholder="/path/to/save/model",
                            )
                            create_model_btn = gr.Button("Create Model")
                            create_model_info = gr.Textbox(
                                label="Creation Status",
                                interactive=False,
                                lines=3,
                            )
            
            # Define event handlers
            
            # Chat functionality
            def chat_response(message, image, max_len, temp, k, p, history):
                if not message.strip():
                    return history, ""
                
                # Add user message to history
                history = history + [(message, None)]
                
                # Generate response
                response = self.chat(
                    message=message,
                    image_path=image,
                    max_length=max_len,
                    temperature=temp,
                    top_k=k,
                    top_p=p,
                )
                
                # Update history
                history[-1] = (history[-1][0], response)
                
                return history, ""
            
            submit_btn.click(
                chat_response,
                inputs=[message, image_input, max_length, temperature, top_k, top_p, chatbot],
                outputs=[chatbot, message],
            )
            
            message.submit(
                chat_response,
                inputs=[message, image_input, max_length, temperature, top_k, top_p, chatbot],
                outputs=[chatbot, message],
            )
            
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, message],
            )
            
            # Model loading
            def load_model_handler(model_path, vocab_path, use_expert_system, custom_vocab_size, custom_layer_count, strict_loading, adapt_vocab):
                try:
                    # Save current model settings
                    old_multimodal = self.multimodal
                    self.multimodal = self.model.config.multimodal if self.model else old_multimodal
                    
                    # If custom parameters are provided, create a custom configuration
                    if custom_vocab_size > 0 or custom_layer_count > 0 or use_expert_system:
                        # Load state dict to examine structure
                        if os.path.isdir(model_path):
                            state_dict_path = os.path.join(model_path, "model.pt")
                            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
                        else:
                            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                        
                        # Determine parameters
                        if "model.token_embeddings.weight" in state_dict:
                            # Use provided vocab size or detect from state dict
                            vocab_size = custom_vocab_size if custom_vocab_size > 0 else state_dict["model.token_embeddings.weight"].size(0)
                            hidden_size = state_dict["model.token_embeddings.weight"].size(1)
                            
                            # Use provided layer count or detect from state dict
                            if custom_layer_count > 0:
                                layer_count = custom_layer_count
                            else:
                                # Count layers
                                layer_count = 0
                                i = 0
                                while any(f"model.layers.{i}.{suffix}" in state_dict for suffix in 
                                        ["input_layernorm.weight", "attn_norm.weight", 
                                         "attention.layer_norm.weight", "feed_forward.layer_norm.weight"]):
                                    layer_count += 1
                                    i += 1
                            
                            # Create custom configuration
                            config = ApertisConfig(
                                vocab_size=vocab_size,
                                hidden_size=hidden_size,
                                num_hidden_layers=layer_count,
                                num_attention_heads=hidden_size // 64,  # Common ratio
                                intermediate_size=hidden_size * 4,      # Common ratio
                                use_expert_system=use_expert_system,
                                multimodal=self.multimodal
                            )
                            
                            # Create model with custom configuration
                            self.model = ApertisForCausalLM(config)
                            
                            # Load state dict with specified strictness
                            self.model.load_state_dict(state_dict, strict=strict_loading)
                            
                            # Load vocabulary
                            self.load_vocabulary(vocab_path)
                            
                            # Get model information
                            config = self.model.config
                            info = f"Model loaded successfully with custom configuration!\n"
                            info += f"Model type: {config.model_type}\n"
                            info += f"Hidden size: {config.hidden_size}\n"
                            info += f"Layers: {config.num_hidden_layers}\n"
                            info += f"Attention heads: {config.num_attention_heads}\n"
                            info += f"Expert system: {config.use_expert_system}\n"
                            info += f"Model vocabulary size: {config.vocab_size}\n"
                            info += f"Loaded vocabulary size: {len(self.vocab) if self.vocab else 'Unknown'}\n"
                            
                            # Add information about vocabulary adaptation
                            if self.token_mapping is not None and adapt_vocab:
                                info += f"Vocabulary adaptation: Enabled (mapping between model and loaded vocabulary)\n"
                            else:
                                info += f"Vocabulary adaptation: Disabled\n"
                            
                            # Move model to device
                            self.model.to(self.device)
                            self.model.eval()
                            
                            return info
                    
                    # Standard loading if no custom parameters
                    self.model = None  # Reset model to ensure clean loading
                    self.token_mapping = None  # Reset token mapping
                    
                    # Load model
                    self.load_model(model_path)
                    
                    # Load vocabulary with adaptation if enabled
                    if not adapt_vocab:
                        # Temporarily disable token mapping creation
                        original_create_token_mapping = self.create_token_mapping
                        self.create_token_mapping = lambda *args, **kwargs: None
                        self.load_vocabulary(vocab_path)
                        self.create_token_mapping = original_create_token_mapping
                    else:
                        self.load_vocabulary(vocab_path)
                    
                    # Restore multimodal setting
                    self.multimodal = old_multimodal
                    
                    # Get model information
                    if self.model is not None:
                        config = self.model.config
                        info = f"Model loaded successfully!\n"
                        info += f"Model type: {config.model_type}\n"
                        info += f"Hidden size: {config.hidden_size}\n"
                        info += f"Layers: {config.num_hidden_layers}\n"
                        info += f"Attention heads: {config.num_attention_heads}\n"
                        info += f"Expert system: {config.use_expert_system}\n"
                        info += f"Model vocabulary size: {config.vocab_size}\n"
                        info += f"Loaded vocabulary size: {len(self.vocab) if self.vocab else 'Unknown'}\n"
                        
                        # Add information about vocabulary adaptation
                        if self.token_mapping is not None and adapt_vocab:
                            info += f"Vocabulary adaptation: Enabled (mapping between model and loaded vocabulary)\n"
                        else:
                            info += f"Vocabulary adaptation: Disabled\n"
                        
                        return info
                    else:
                        return "Failed to load model."
                except Exception as e:
                    return f"Error loading model: {str(e)}"
            
            load_model_btn.click(
                load_model_handler,
                inputs=[model_path_input, vocab_path_input, expert_system_checkbox, 
                        custom_vocab_size, custom_layer_count, strict_loading, adapt_vocab_checkbox],
                outputs=[model_info],
            )
            
            # Model creation
            def create_model_handler(model_size, multimodal, use_expert_system, vocab_size, output_path):
                try:
                    # Create custom configuration
                    config = ApertisConfig(
                        vocab_size=int(vocab_size),
                        hidden_size={"small": 512, "base": 768, "large": 1024}[model_size],
                        num_hidden_layers={"small": 8, "base": 12, "large": 24}[model_size],
                        num_attention_heads={"small": 8, "base": 12, "large": 16}[model_size],
                        intermediate_size={"small": 2048, "base": 3072, "large": 4096}[model_size],
                        use_expert_system=use_expert_system,
                        multimodal=multimodal
                    )
                    
                    # Create model
                    model = ApertisForCausalLM(config)
                    
                    # Create output directory
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Save model
                    torch.save(model.state_dict(), os.path.join(output_path, "model.pt"))
                    
                    # Save configuration
                    with open(os.path.join(output_path, "config.json"), "w") as f:
                        json.dump(model.config.to_dict(), f, indent=2)
                    
                    # Create a minimal vocabulary if it doesn't exist
                    vocab_path = os.path.join(output_path, "vocab.json")
                    if not os.path.exists(vocab_path):
                        minimal_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
                        with open(vocab_path, "w") as f:
                            json.dump(minimal_vocab, f, indent=2)
                    
                    return f"Model created successfully at {output_path}"
                except Exception as e:
                    return f"Error creating model: {str(e)}"
            
            create_model_btn.click(
                create_model_handler,
                inputs=[new_model_size, new_model_multimodal, new_model_expert, new_model_vocab_size, new_model_output],
                outputs=[create_model_info],
            )
            
            # Training functionality
            def start_training(
                model_size, multimodal, expert_system,
                train_file, val_file, vocab_file, img_dir,
                batch, lr, epochs, checkpoint_freq, iter_checkpoint_freq, 
                gpu_ids, distributed, gpu_memory_fraction,
                out_dir, use_wb, wb_project,
            ):
                try:
                    # Create temporary directory for training files
                    temp_dir = tempfile.mkdtemp()
                    
                    # Save uploaded files
                    train_path = os.path.join(temp_dir, "train.jsonl")
                    with open(train_path, "wb") as f:
                        if hasattr(train_file, 'name'):  # Handle NamedString objects from Gradio
                            with open(train_file.name, "rb") as source_file:
                                f.write(source_file.read())
                        else:
                            f.write(train_file)
                    
                    vocab_path = os.path.join(temp_dir, "vocab.json")
                    with open(vocab_path, "wb") as f:
                        if hasattr(vocab_file, 'name'):  # Handle NamedString objects from Gradio
                            with open(vocab_file.name, "rb") as source_file:
                                f.write(source_file.read())
                        else:
                            f.write(vocab_file)
                    
                    val_path = None
                    if val_file is not None:
                        val_path = os.path.join(temp_dir, "val.jsonl")
                        with open(val_path, "wb") as f:
                            if hasattr(val_file, 'name'):  # Handle NamedString objects from Gradio
                                with open(val_file.name, "rb") as source_file:
                                    f.write(source_file.read())
                            else:
                                f.write(val_file)
                    
                    # Process GPU selection
                    selected_gpu_ids = [int(gpu_id) for gpu_id in gpu_ids] if gpu_ids else None
                    
                    # Create configuration
                    config = {
                        "data_config": {
                            "train_data_path": train_path,
                            "tokenizer_path": vocab_path,
                            "max_length": 512,
                            "multimodal": multimodal,
                        },
                        "model_config": {
                            "hidden_size": {"small": 512, "base": 768, "large": 1024}[model_size],
                            "num_hidden_layers": {"small": 8, "base": 12, "large": 24}[model_size],
                            "num_attention_heads": {"small": 8, "base": 12, "large": 16}[model_size],
                            "intermediate_size": {"small": 2048, "base": 3072, "large": 4096}[model_size],
                            "use_expert_system": expert_system,
                            "multimodal": multimodal,
                        },
                        "training_config": {
                            "output_dir": out_dir,
                            "batch_size": batch,
                            "learning_rate": lr,
                            "num_epochs": epochs,
                            "use_wandb": use_wb,
                            "wandb_project": wb_project,
                            "gradient_accumulation_steps": 4,
                            "fp16": True,
                            "gpu_memory_fraction": gpu_memory_fraction,
                            "use_gradient_checkpointing": True,
                            "dynamic_batch_sizing": True,
                            "checkpoint_steps": checkpoint_freq,
                            "iteration_checkpoint_steps": iter_checkpoint_freq,
                            "gpu_ids": selected_gpu_ids,
                            "distributed_training": distributed,
                        },
                    }
                    
                    # Add validation data if provided
                    if val_path is not None:
                        config["data_config"]["val_data_path"] = val_path
                    
                    # Add image directory if provided
                    if multimodal and img_dir:
                        config["data_config"]["image_dir"] = img_dir
                    
                    # Save configuration
                    config_path = os.path.join(temp_dir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)
                    
                    # Start training in a separate thread
                    import threading
                    
                    def train_thread():
                        try:
                            from src.training.pipeline import train_from_config
                            train_from_config(config_path)
                        except Exception as e:
                            logger.error(f"Error in training thread: {str(e)}")
                        finally:
                            # Clean up temporary directory
                            shutil.rmtree(temp_dir)
                    
                    thread = threading.Thread(target=train_thread)
                    thread.daemon = True
                    thread.start()
                    
                    # Prepare GPU information for output
                    gpu_info = ""
                    if selected_gpu_ids:
                        gpu_info = f"- GPUs: {selected_gpu_ids}\n"
                        if distributed and len(selected_gpu_ids) > 1:
                            gpu_info += f"- Distributed training: Enabled\n"
                        elif len(selected_gpu_ids) > 1:
                            gpu_info += f"- Using DataParallel across GPUs\n"
                        gpu_info += f"- GPU memory fraction: {gpu_memory_fraction}\n"
                    else:
                        gpu_info = "- Using CPU for training\n"
                    
                    checkpoint_info = ""
                    if checkpoint_freq > 0:
                        checkpoint_info += f"- Global step checkpoints: Every {checkpoint_freq} steps\n"
                    else:
                        checkpoint_info += "- Global step checkpoints: Disabled\n"
                        
                    if iter_checkpoint_freq > 0:
                        checkpoint_info += f"- Iteration checkpoints: Every {iter_checkpoint_freq} iterations within each epoch\n"
                    else:
                        checkpoint_info += "- Iteration checkpoints: Disabled\n"
                    
                    return f"Training started with configuration:\n" \
                           f"- Model: {model_size} {'(multimodal)' if multimodal else ''} {'(expert system)' if expert_system else ''}\n" \
                           f"- Batch size: {batch}\n" \
                           f"- Learning rate: {lr}\n" \
                           f"- Epochs: {epochs}\n" \
                           f"{gpu_info}" \
                           f"{checkpoint_info}" \
                           f"- Output directory: {out_dir}\n\n" \
                           f"Training is running in the background. Check the console for progress."
                except Exception as e:
                    return f"Error starting training: {str(e)}"
            
            train_btn.click(
                start_training,
                inputs=[
                    model_size, multimodal_checkbox, use_expert_system,
                    train_data, val_data, vocab_data, image_dir,
                    batch_size, learning_rate, num_epochs, 
                    checkpoint_steps, iteration_checkpoint_steps,
                    gpu_selection, distributed_training, gpu_memory_fraction,
                    output_dir, use_wandb, wandb_project,
                ],
                outputs=[training_output],
            )
        
        # Launch interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=self.port,
            share=self.share,
        )
