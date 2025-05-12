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
                    with open(config_path, "r", encoding="utf-8") as f: # Added encoding
                        config_dict = json.load(f)
                    
                    config = ApertisConfig.from_dict(config_dict)
                    self.model = ApertisForCausalLM(config)
                    
                    # Load state dict
                    state_dict_path = os.path.join(model_path, "model.pt")
                    if os.path.exists(state_dict_path):
                        # When loading a model with its config, we can be strict.
                        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device, weights_only=True), strict=True)
                    else:
                        logger.warning(f"Model state dict not found at {state_dict_path}")
                else:
                    logger.warning(f"Model configuration not found at {config_path}. Attempting to infer parameters from .pt file if this is a raw state_dict path.")
                    # Fall through to .pt loading logic if config.json is missing but path is a dir
                    if os.path.exists(os.path.join(model_path, "model.pt")):
                        self._load_from_pt_file(os.path.join(model_path, "model.pt"))
                    else:
                        logger.error(f"Neither config.json nor model.pt found in directory {model_path}.")
                        self.model = create_apertis_model(model_size="small", multimodal=self.multimodal) # Fallback

            elif os.path.isfile(model_path) and model_path.endswith(".pt"):
                 self._load_from_pt_file(model_path)
            else:
                logger.error(f"Invalid model_path: {model_path}. Not a directory or .pt file.")
                self.model = create_apertis_model(model_size="small", multimodal=self.multimodal) # Fallback
            
            if self.model: # Ensure model was loaded or created
                self.model.to(self.device)
                self.model.eval()
            
            logger.info("Model loading process completed.")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            logger.info("Creating default model as fallback")
            self.model = create_apertis_model(model_size="base", multimodal=self.multimodal)
            if self.model:
                self.model.to(self.device)
                self.model.eval()

    def _load_from_pt_file(self, pt_path:str):
        """Helper to load model purely from a .pt state_dict file."""
        logger.info(f"Attempting to load model from raw .pt file: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            
            use_expert_system = any("feed_forward.feed_forward.experts" in k or "feed_forward.feed_forward.router" in k for k in state_dict.keys())
            
            vocab_size = None
            if "model.token_embeddings.weight" in state_dict:
                vocab_size = state_dict["model.token_embeddings.weight"].size(0)
                logger.info(f"Detected vocabulary size from state_dict: {vocab_size}")
            
            hidden_size = None
            layer_count = 0
            model_size_name = "base" 

            if "model.token_embeddings.weight" in state_dict:
                hidden_size = state_dict["model.token_embeddings.weight"].size(1)
                i = 0
                while any(f"model.layers.{i}.{suffix}" in state_dict for suffix in 
                        ["attention.layer_norm.weight", "feed_forward.layer_norm.weight", # More specific keys
                         "attention.attention.q_proj.weight", "feed_forward.feed_forward.0.weight"]): # Even more specific
                    layer_count += 1
                    i += 1
                
                if hidden_size == 512 and layer_count > 0 and layer_count <= 8 : model_size_name = "small"
                elif hidden_size == 768 and layer_count > 0 and layer_count <= 12: model_size_name = "base"
                elif hidden_size == 1024 and layer_count > 0: model_size_name = "large"
                elif layer_count == 0 : logger.warning("Could not detect layer_count from state_dict, using preset default.")
                
                logger.info(f"Detected model size: {model_size_name} (hidden_size={hidden_size}, layers={layer_count if layer_count > 0 else 'preset'})")
                logger.info(f"Expert system: {use_expert_system}")
            else:
                logger.warning(f"Could not determine hidden_size/layers from token_embeddings. Using '{model_size_name}' preset.")

            _model_presets = {
                "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
            }
            
            preset_config = _model_presets.get(model_size_name, _model_presets["base"])
            config_kwargs = {**preset_config}

            if vocab_size: config_kwargs["vocab_size"] = vocab_size
            if hidden_size: config_kwargs["hidden_size"] = hidden_size # Override preset if detected
            if layer_count > 0: config_kwargs["num_hidden_layers"] = layer_count # Override preset if detected
            
            # If hidden_size was detected, recalculate heads and intermediate if not preset for that exact hidden_size
            if hidden_size and hidden_size not in [_model_presets[ms]["hidden_size"] for ms in _model_presets]:
                 config_kwargs["num_attention_heads"] = hidden_size // 64 # Heuristic
                 config_kwargs["intermediate_size"] = hidden_size * 4    # Heuristic

            config_kwargs["use_expert_system"] = use_expert_system
            config_kwargs["multimodal"] = self.multimodal 
            
            config = ApertisConfig(**config_kwargs)
            self.model = ApertisForCausalLM(config)
            
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys or unexpected_keys:
                logger.warning(f"Loading from .pt completed with issues:"
                               f" Missing keys: {missing_keys if missing_keys else 'None'}."
                               f" Unexpected keys: {unexpected_keys if unexpected_keys else 'None'}.")
            else:
                logger.info(f"Successfully loaded model from .pt using inferred '{model_size_name}' base configuration.")

        except Exception as e:
            logger.error(f"Error during _load_from_pt_file for {pt_path}: {e}", exc_info=True)
            logger.info("Creating default model as fallback after .pt load failure.")
            self.model = create_apertis_model(model_size="small", multimodal=self.multimodal)


    def load_vocabulary(self, vocab_file: str) -> None:
        """Load vocabulary from file."""
        try:
            logger.info(f"Loading vocabulary from {vocab_file}")
            
            with open(vocab_file, "r", encoding="utf-8") as f: # Explicitly use UTF-8
                vocab_data = json.load(f)
            
            # Handle different vocabulary formats
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    token_list = vocab_data["tokens"]
                    self.vocab = {token: idx for idx, token in enumerate(token_list)}
                    logger.info(f"Converted list-based vocabulary to dictionary format with {len(self.vocab)} tokens")
                else:
                    self.vocab = vocab_data
            else:
                raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")
            
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            logger.info(f"Vocabulary loaded with {len(self.vocab)} tokens")
            
            if self.model and hasattr(self.model.config, "vocab_size"):
                model_vocab_size = self.model.config.vocab_size
                loaded_vocab_actual_size = len(self.vocab) # Number of entries
                
                # This warning is key: it means the model's expected vocab dimension
                # doesn't match the number of unique tokens in the vocab file.
                if model_vocab_size != loaded_vocab_actual_size:
                    logger.warning(
                        f"Model's configured vocabulary size ({model_vocab_size}) "
                        f"doesn't match the number of unique tokens in the loaded vocabulary file ({loaded_vocab_actual_size}). "
                        f"This can lead to issues if not handled by token mapping or a correctly sized model."
                    )
                    # Token mapping will attempt to bridge this if `adapt_vocab` is true in UI,
                    # or by default if called directly.
                    self.create_token_mapping(model_vocab_size, loaded_vocab_actual_size)
                    logger.info("Attempted to create token mapping to handle vocabulary size mismatch.")
                else:
                    # If sizes match, no mapping is needed.
                    self.token_mapping = None 
                    logger.info("Model vocabulary size matches loaded vocabulary size. No token mapping needed.")


        except FileNotFoundError:
            logger.error(f"Vocabulary file not found: {vocab_file}")
            self._create_fallback_vocab()
        except Exception as e:
            logger.error(f"Error loading vocabulary from {vocab_file}: {e}", exc_info=True)
            self._create_fallback_vocab()

    def _create_fallback_vocab(self):
        logger.info("Creating minimal vocabulary as fallback.")
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_mapping = None


    def create_token_mapping(self, model_vocab_size: int, loaded_vocab_num_entries: int) -> None:
        """Create a mapping between model tokens and loaded vocabulary tokens."""
        # This mapping is generally needed if model_vocab_size (from model config)
        # differs from len(self.vocab) (actual entries in the loaded vocab file).
        # The goal is to map token IDs from self.vocab (which might be sparse or different)
        # to valid indices within the model's embedding table (0 to model_vocab_size-1).

        if self.vocab is None:
            logger.error("Cannot create token mapping: vocabulary not loaded.")
            return

        self.token_mapping = {}
        special_tokens_map = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3} # Ideal low IDs

        # Ensure special tokens from the vocab file are mapped, preferably to standard low IDs if possible
        for token_str, ideal_id in special_tokens_map.items():
            if token_str in self.vocab:
                vocab_file_id = self.vocab[token_str]
                # Map to ideal_id if it's within model's range, otherwise map to its own ID if valid, else unk
                if ideal_id < model_vocab_size:
                    self.token_mapping[vocab_file_id] = ideal_id
                elif vocab_file_id < model_vocab_size: # If ideal_id is OOB, but its original ID is fine
                    self.token_mapping[vocab_file_id] = vocab_file_id
                else: # Both ideal and original are OOB, map to model's <unk>
                    self.token_mapping[vocab_file_id] = min(special_tokens_map["<unk>"], model_vocab_size -1)


        # Map remaining tokens from self.vocab
        # model_unk_id should be determined from special_tokens_map or default to a safe value
        model_unk_id = min(special_tokens_map["<unk>"], model_vocab_size -1) if model_vocab_size > 0 else 0

        for vocab_file_token_id in self.vocab.values(): # Iterate over IDs from the loaded vocab file
            if vocab_file_token_id not in self.token_mapping: # If not already mapped (e.g. as a special token)
                if vocab_file_token_id < model_vocab_size:
                    # If the ID from vocab file is a valid index for the model, map it directly
                    self.token_mapping[vocab_file_token_id] = vocab_file_token_id
                else:
                    # If the ID from vocab file is out of bounds for the model, map to model's UNK
                    self.token_mapping[vocab_file_token_id] = model_unk_id
        
        logger.info(f"Created token mapping. Model expects up to {model_vocab_size} tokens. Loaded vocab file has {loaded_vocab_num_entries} unique entries.")
        # Example: if model_vocab_size=100, loaded_vocab_num_entries=50, but vocab file contains token "xyz" : 200
        # self.token_mapping[200] would be mapped to model_unk_id.
        # If vocab file has "abc": 5, it maps to 5.

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded vocabulary and apply mapping."""
        if self.vocab is None:
            logger.warning("Vocabulary not loaded, using minimal tokenization (all UNK).")
            # Fallback to a known UNK or a default if model not fully loaded.
            return [self.model.config.pad_token_id if self.model else 0] * len(text.split())


        raw_token_ids = []
        # Determine a safe UNK ID from the loaded vocab, or a default.
        loaded_vocab_unk_id = self.vocab.get("<unk>")
        if loaded_vocab_unk_id is None: # If "<unk>" is not in the vocab file
            # Try to use a conventional ID if the model is small, otherwise 0.
            # This is a fallback, ideally <unk> is always in vocab.json.
            loaded_vocab_unk_id = 3 if 3 < len(self.vocab) else (len(self.vocab) -1 if len(self.vocab) > 0 else 0)


        for word in text.split():
            raw_token_ids.append(self.vocab.get(word, loaded_vocab_unk_id))

        if self.token_mapping:
            # Apply mapping: raw_token_id (from vocab file) -> model_token_id (valid for model)
            # The model_unk_id for mapping should be the one the model expects (e.g., from config or standard low ID)
            model_unk_for_mapping = min(getattr(self.model.config, 'unk_token_id', 3), self.model.config.vocab_size -1) if self.model else 0
            
            mapped_tokens = [self.token_mapping.get(tid, model_unk_for_mapping) for tid in raw_token_ids]
            return mapped_tokens
        else:
            # No mapping needed (or failed to create), directly use raw_token_ids
            # but ensure they are within model's actual vocab_size
            model_vocab_size = self.model.config.vocab_size if self.model else len(self.vocab)
            model_unk_direct = min(getattr(self.model.config, 'unk_token_id', 3), model_vocab_size -1) if self.model else 0

            return [tid if tid < model_vocab_size else model_unk_direct for tid in raw_token_ids]

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs (from model output) back to text."""
        if self.reverse_vocab is None:
            logger.warning("Reverse vocabulary not loaded, cannot detokenize properly.")
            return f"[DetokenizationError: Reverse vocab missing. IDs: {token_ids[:10]}...]"

        words = []
        # If a token_mapping exists, it means self.vocab's IDs (vocab_file_ids) map to model_ids.
        # For detokenization, we have model_ids and need to find the corresponding vocab_file_id
        # to look up in self.reverse_vocab.
        
        # Create a reverse of the token_mapping: model_id -> vocab_file_id
        # This is only needed if token_mapping was actively used to bridge a discrepancy.
        # If model_vocab_size == len(self.vocab) and IDs are dense, token_mapping might be identity or None.
        
        reverse_id_map = None
        if self.token_mapping:
            reverse_id_map = {model_id: vocab_file_id for vocab_file_id, model_id in self.token_mapping.items()}

        for model_token_id in token_ids:
            id_to_lookup_in_reverse_vocab = model_token_id

            if reverse_id_map:
                # Find the original vocab_file_id that corresponds to this model_token_id
                original_vocab_file_id = reverse_id_map.get(model_token_id)
                if original_vocab_file_id is not None:
                    id_to_lookup_in_reverse_vocab = original_vocab_file_id
                # else: if model_token_id not in reverse_id_map, it might be an ID that
                # wasn't in the original vocab file but was part of the model's larger vocab.
                # In this case, looking it up directly in reverse_vocab might fail or give wrong token.
                # For now, we'll proceed with the model_token_id if not found in reverse_id_map.

            word = self.reverse_vocab.get(id_to_lookup_in_reverse_vocab)
            
            if word is not None and word not in ["<pad>", "<bos>", "<eos>"]: # Don't print <unk> either if it was the result
                if word == "<unk>" and id_to_lookup_in_reverse_vocab != self.vocab.get("<unk>"): # If it resolved to unk but wasn't the unk_id
                    words.append(f"[UNK:{id_to_lookup_in_reverse_vocab}]") # show the ID that became UNK
                else:
                    words.append(word)
            elif word is None: # ID not in reverse_vocab
                 words.append(f"[ID:{id_to_lookup_in_reverse_vocab}]")


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
        if self.vocab is None:
            return "Vocabulary not loaded. Cannot process prompt."

        try:
            # Tokenize prompt
            input_ids_list = self.tokenize(prompt)
            
            # Add <bos> token if not present and vocab is loaded
            bos_token_id = self.vocab.get("<bos>")
            if bos_token_id is None: # Fallback if <bos> not in vocab
                bos_token_id = min(1, self.model.config.vocab_size -1) if self.model.config.vocab_size > 0 else 0
            
            # Apply mapping to BOS if needed (though special tokens are usually handled in create_token_mapping)
            if self.token_mapping and bos_token_id in self.token_mapping:
                 bos_token_id = self.token_mapping[bos_token_id]


            if not input_ids_list or input_ids_list[0] != bos_token_id:
                input_ids_list = [bos_token_id] + input_ids_list
            
            # Convert to tensor
            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)
            
            # Create attention mask (all 1s for input tokens)
            attention_mask = torch.ones_like(input_ids)
            
            # Prepare image if provided and model is multimodal
            pixel_values = None
            if image_path is not None and self.multimodal: # self.multimodal should reflect loaded model
                if hasattr(self.model.config, 'multimodal') and self.model.config.multimodal:
                    pixel_values = self.preprocess_image(image_path).to(self.device)
                else:
                    logger.warning("Image provided, but loaded model is not configured as multimodal.")

            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_length=max_length + input_ids.shape[1], # max_length should be for new tokens
                    do_sample=temperature > 0.001, # Allow very low temp for near-greedy
                    temperature=temperature if temperature > 0.001 else 1.0, # Temp 0 can cause issues
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=self.token_mapping.get(self.vocab.get("<eos>"), self.model.config.eos_token_id) if self.token_mapping and self.vocab.get("<eos>") is not None else self.model.config.eos_token_id,
                    pad_token_id=self.token_mapping.get(self.vocab.get("<pad>"), self.model.config.pad_token_id) if self.token_mapping and self.vocab.get("<pad>") is not None else self.model.config.pad_token_id,

                )
            
            # Detokenize response (excluding input tokens)
            response_ids = output_ids[0, input_ids.shape[1]:].tolist()
            response = self.detokenize(response_ids)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
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
                                max_length_slider = gr.Slider( # Renamed to avoid conflict with function arg
                                    minimum=10,
                                    maximum=500,
                                    value=100,
                                    step=10,
                                    label="Max New Tokens",
                                )
                                temperature_slider = gr.Slider( # Renamed
                                    minimum=0.0, # Allow 0 for greedy
                                    maximum=1.5,
                                    value=0.7,
                                    step=0.05,
                                    label="Temperature",
                                )
                                top_k_slider = gr.Slider( # Renamed
                                    minimum=0, # Allow 0 to disable top_k
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="Top K",
                                )
                                top_p_slider = gr.Slider( # Renamed
                                    minimum=0.0, # Allow 0 to disable top_p
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.05,
                                    label="Top P",
                                )
                
                # Training tab
                with gr.TabItem("Training"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Configuration")
                            model_size_dropdown = gr.Dropdown( # Renamed
                                choices=["small", "base", "large"],
                                value="base",
                                label="Model Size",
                            )
                            multimodal_training_checkbox = gr.Checkbox( # Renamed
                                value=self.multimodal, # Default to interface's current multimodal state
                                label="Multimodal (Text + Image)",
                            )
                            use_expert_system_training_checkbox = gr.Checkbox( # Renamed
                                value=False,
                                label="Use Adaptive Expert System",
                            )
                            
                            gr.Markdown("## Training Data")
                            train_data_file = gr.File( # Renamed
                                label="Training Data (JSONL)",
                                file_types=[".jsonl"],
                            )
                            val_data_file = gr.File( # Renamed
                                label="Validation Data (JSONL, optional)",
                                file_types=[".jsonl"],
                            )
                            vocab_data_file = gr.File( # Renamed
                                label="Vocabulary File (JSON)",
                                file_types=[".json"],
                            )
                            image_dir_textbox = gr.Textbox( # Renamed
                                label="Image Directory (for multimodal training)",
                                placeholder="/path/to/images",
                                visible=self.multimodal, # Default visibility
                            )
                            
                            multimodal_training_checkbox.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[multimodal_training_checkbox],
                                outputs=[image_dir_textbox],
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Training Parameters")
                            batch_size_slider = gr.Slider( # Renamed
                                minimum=1, maximum=64, value=4, step=1, label="Batch Size",
                            )
                            learning_rate_slider = gr.Slider( # Renamed
                                minimum=1e-6, maximum=1e-3, value=5e-5, step=1e-6, label="Learning Rate", format="%.1e"
                            )
                            num_epochs_slider = gr.Slider( # Renamed
                                minimum=1, maximum=100, value=3, step=1, label="Number of Epochs",
                            )
                            
                            eval_every_n_epochs_slider = gr.Slider( # Renamed
                                minimum=0, maximum=10, value=1, step=1,
                                label="Evaluate Every N Epochs (0 to disable epoch-end validation)",
                            )

                            with gr.Accordion("Checkpoint Settings", open=True):
                                checkpoint_steps_slider = gr.Slider( # Renamed
                                    minimum=0, maximum=5000, value=1000, step=100,
                                    label="Global Step Checkpoint Freq (0 to disable)",
                                )
                                iteration_checkpoint_steps_slider = gr.Slider( # Renamed
                                    minimum=0, maximum=1000, value=100, step=10,
                                    label="Iteration Checkpoint Freq (0 to disable)",
                                )
                                gr.Markdown("*Global: Save every N global steps. Iteration: Save every N iterations within epoch.*")
                            
                            with gr.Accordion("GPU Settings", open=True):
                                available_gpus = get_available_gpus()
                                gpu_info_md = "### Available GPUs:\n" + ("\n".join([f"- GPU {g['id']}: {g['name']} ({g['total_memory']:.1f} GB)" for g in available_gpus]) if available_gpus else "No CUDA-capable GPUs detected.")
                                gr.Markdown(gpu_info_md)
                                
                                gpu_choices_list = [str(g['id']) for g in available_gpus]
                                gpu_selection_checkboxgroup = gr.CheckboxGroup( # Renamed
                                    choices=gpu_choices_list,
                                    value=[gpu_choices_list[0]] if gpu_choices_list else [],
                                    label="Select GPUs for Training", visible=bool(gpu_choices_list)
                                )
                                distributed_training_checkbox = gr.Checkbox( # Renamed
                                    value=False, label="Use Distributed Training (multi-GPU)", visible=len(gpu_choices_list) > 1
                                )
                                if not gpu_choices_list: gr.Markdown("Training will use CPU.")

                                def update_dist_train_visibility(selected_gpus): return gr.update(visible=len(selected_gpus) > 1)
                                gpu_selection_checkboxgroup.change(update_dist_train_visibility, inputs=[gpu_selection_checkboxgroup], outputs=[distributed_training_checkbox])
                                
                                gpu_memory_fraction_slider = gr.Slider( # Renamed
                                    minimum=0.1, maximum=1.0, value=0.7, step=0.05, label="GPU Memory Fraction (per GPU)",
                                )
                            
                            output_dir_textbox = gr.Textbox(label="Output Directory", value="output") # Renamed
                            use_wandb_checkbox = gr.Checkbox(value=False, label="Use Weights & Biases") # Renamed
                            wandb_project_textbox = gr.Textbox(label="W&B Project", value="apertis", visible=False) # Renamed
                            use_wandb_checkbox.change(lambda x: gr.update(visible=x), inputs=[use_wandb_checkbox], outputs=[wandb_project_textbox])
                            
                            train_btn = gr.Button("Start Training")
                            training_output_textbox = gr.Textbox(label="Training Status", interactive=False, lines=10) # Renamed
                
                # Model tab
                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model")
                            model_path_load_textbox = gr.Textbox(label="Model Path", value=self.model_path or "", placeholder="/path/to/model.pt or /path/to/model_dir") # Renamed
                            vocab_path_load_textbox = gr.Textbox(label="Vocabulary Path", value=self.vocab_file or "", placeholder="/path/to/vocab.json") # Renamed
                            
                            with gr.Accordion("Advanced Loading Options", open=False): # Default closed
                                expert_system_load_checkbox = gr.Checkbox(value=False, label="Model uses Expert System") # Renamed
                                custom_vocab_size_load_number = gr.Number(value=0, label="Custom Vocab Size (0=auto)", precision=0) # Renamed
                                custom_layer_count_load_number = gr.Number(value=0, label="Custom Layer Count (0=auto)", precision=0) # Renamed
                                strict_loading_checkbox = gr.Checkbox(value=False, label="Strict Loading (Exact Match)") # Default False for .pt files
                                adapt_vocab_load_checkbox = gr.Checkbox(value=True, label="Auto-adapt Vocab Mismatches") # Renamed
                            
                            load_model_btn = gr.Button("Load Model")
                            model_info_textbox = gr.Textbox(label="Model Information", interactive=False, lines=10) # Renamed
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Create New Model")
                            new_model_size_dropdown = gr.Dropdown(choices=["small", "base", "large"], value="base", label="Model Size") # Renamed
                            new_model_multimodal_checkbox = gr.Checkbox(value=False, label="Multimodal (Text + Image)") # Renamed
                            new_model_expert_checkbox = gr.Checkbox(value=False, label="Use Expert System") # Renamed
                            new_model_vocab_size_number = gr.Number(value=32000, label="Vocabulary Size", precision=0) # Renamed
                            new_model_output_textbox = gr.Textbox(label="Output Path", value="models/new_model", placeholder="/path/to/save/model") # Renamed
                            create_model_btn = gr.Button("Create Model")
                            create_model_info_textbox = gr.Textbox(label="Creation Status", interactive=False, lines=3) # Renamed
            
            # Define event handlers
            
            def chat_response_handler(msg_text, img_path, max_len, temp, k, p, history): # Unique names for handler args
                if not msg_text.strip() and not img_path: # Require text or image
                    return history, ""
                
                history = history + [(f"{msg_text}{' (Image Attached)' if img_path else ''}", None)]
                
                response_text = self.chat(
                    message=msg_text, image_path=img_path, max_length=max_len,
                    temperature=temp, top_k=k, top_p=p,
                )
                history[-1] = (history[-1][0], response_text)
                return history, "" # Clear input textbox
            
            submit_btn.click(
                chat_response_handler,
                inputs=[message, image_input, max_length_slider, temperature_slider, top_k_slider, top_p_slider, chatbot],
                outputs=[chatbot, message],
            )
            message.submit( # Allow Enter to submit
                chat_response_handler,
                inputs=[message, image_input, max_length_slider, temperature_slider, top_k_slider, top_p_slider, chatbot],
                outputs=[chatbot, message],
            )
            clear_btn.click(lambda: ([], "", None), outputs=[chatbot, message, image_input]) # Clear image_input too
            
            def load_model_ui_handler(m_path, v_path, expert_sys, cust_v_size, cust_l_count, strict_load, adapt_v):
                # This handler now directly calls the robust self.load_model and self.load_vocabulary
                # The complex logic is inside those methods.
                # We just need to manage the self.multimodal state if it's derived from UI perhaps.
                # For loading, self.multimodal is best determined by the loaded model's config.
                
                # Reset internal model state before loading
                self.model = None
                self.vocab = None
                self.reverse_vocab = None
                self.token_mapping = None
                self.model_path = m_path # Update interface state
                self.vocab_file = v_path # Update interface state

                # The advanced options like expert_sys, cust_v_size, cust_l_count are more for guiding
                # the _load_from_pt_file if a config.json is missing. Strict loading is also used there.
                # The `adapt_vocab` will influence `create_token_mapping` if there's a mismatch.
                
                # Set a temporary flag for how strict_loading from UI should influence .pt loading
                # This is a bit of a hack; ideally, the config object creation would take these.
                # For now, the _load_from_pt_file uses strict=False by default.
                # If strict_loading from UI is True, we'd want the initial config guess to be very precise.
                # However, loading a raw .pt without its config is inherently non-strict.
                
                # The primary action is to call load_model then load_vocabulary
                self.load_model(m_path) # This now has better inference for .pt files
                
                # If `adapt_v` is false, we want to avoid token mapping during the subsequent load_vocabulary
                if not adapt_v and hasattr(self, 'create_token_mapping'):
                    original_create_mapping_func = self.create_token_mapping
                    self.create_token_mapping = lambda mv_size, lv_size: None # Temp disable
                    self.load_vocabulary(v_path)
                    self.create_token_mapping = original_create_mapping_func # Restore
                else:
                    self.load_vocabulary(v_path)

                if self.model and hasattr(self.model.config, 'to_dict'):
                    # Update self.multimodal based on loaded model config
                    self.multimodal = self.model.config.multimodal

                    info = "Model Loaded:\n" + json.dumps(self.model.config.to_dict(), indent=2)
                    info += f"\n\nVocabulary: {len(self.vocab) if self.vocab else 'Not loaded'} tokens."
                    if self.token_mapping:
                        info += "\nToken mapping is active due to vocab/model size mismatch."
                    return info
                return "Failed to load model or model has no config."

            load_model_btn.click(
                load_model_ui_handler,
                inputs=[model_path_load_textbox, vocab_path_load_textbox, expert_system_load_checkbox, 
                        custom_vocab_size_load_number, custom_layer_count_load_number, strict_loading_checkbox, adapt_vocab_load_checkbox],
                outputs=[model_info_textbox],
            )
            
            def create_model_ui_handler(new_m_size, new_m_multi, new_m_expert, new_m_v_size, new_m_out_path):
                try:
                    _presets_create = {
                        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
                    }
                    preset = _presets_create[new_m_size]
                    config = ApertisConfig(
                        vocab_size=int(new_m_v_size), **preset,
                        use_expert_system=new_m_expert, multimodal=new_m_multi
                    )
                    model = ApertisForCausalLM(config)
                    os.makedirs(new_m_out_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(new_m_out_path, "model.pt"))
                    config.save_pretrained(new_m_out_path) # Saves config.json

                    # Create a dummy vocab.json for the new model
                    dummy_vocab = {f"<token_{i}>": i for i in range(int(new_m_v_size))}
                    # Ensure special tokens are there if vocab is large enough
                    if int(new_m_v_size) >= 4:
                         dummy_vocab["<pad>"] = 0
                         dummy_vocab["<bos>"] = 1
                         dummy_vocab["<eos>"] = 2
                         dummy_vocab["<unk>"] = 3
                    with open(os.path.join(new_m_out_path, "vocab.json"), "w", encoding="utf-8") as f:
                        json.dump(dummy_vocab, f, indent=2)
                    
                    return f"Model & config.json created at {new_m_out_path}. Dummy vocab.json also created."
                except Exception as e:
                    logger.error(f"Create model error: {e}", exc_info=True)
                    return f"Error: {str(e)}"

            create_model_btn.click(
                create_model_ui_handler,
                inputs=[new_model_size_dropdown, new_model_multimodal_checkbox, new_model_expert_checkbox, 
                        new_model_vocab_size_number, new_model_output_textbox],
                outputs=[create_model_info_textbox],
            )
            
            def start_training_ui_handler(
                m_size_ui, m_multi_ui, exp_sys_ui, train_f, val_f, vocab_f_ui, img_dir_ui,
                batch_s_ui, lr_s_ui, epochs_s_ui, eval_n_epochs_ui,
                chkpt_steps_ui, iter_chkpt_steps_ui, gpu_sel_ui, dist_train_ui, gpu_mem_frac_ui,
                out_dir_s_ui, use_wb_s_ui, wb_proj_s_ui,
            ):
                # This function mostly prepares the config and calls the backend training.
                # The actual file handling for temp files is good.
                # Ensure all UI component names are correctly passed.
                
                # Basic validation for required files
                if not train_f: return "Error: Training data file is required."
                if not vocab_f_ui: return "Error: Vocabulary file is required."

                temp_train_dir = tempfile.mkdtemp() # Unique temp dir for this training run
                
                train_p = os.path.join(temp_train_dir, "train.jsonl")
                with open(train_p, "wb") as f_out, open(train_f.name, "rb") as f_in: shutil.copyfileobj(f_in, f_out)
                
                vocab_p = os.path.join(temp_train_dir, "vocab.json")
                with open(vocab_p, "wb") as f_out, open(vocab_f_ui.name, "rb") as f_in: shutil.copyfileobj(f_in, f_out)
                
                val_p = None
                if val_f:
                    val_p = os.path.join(temp_train_dir, "val.jsonl")
                    with open(val_p, "wb") as f_out, open(val_f.name, "rb") as f_in: shutil.copyfileobj(f_in, f_out)

                selected_gpus = [int(gid) for gid in gpu_sel_ui] if gpu_sel_ui else None
                
                _train_model_presets = {
                    "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
                }
                model_p = _train_model_presets[m_size_ui]

                train_run_config = {
                    "data_config": {
                        "train_data_path": train_p, "tokenizer_path": vocab_p, "val_data_path": val_p,
                        "max_length": 512, "multimodal": m_multi_ui, "image_dir": img_dir_ui if m_multi_ui else None,
                        "image_size": 224, # Consider making this configurable in UI
                    },
                    "model_config": {
                        **model_p, # vocab_size will be auto-determined by pipeline
                        "use_expert_system": exp_sys_ui, "multimodal": m_multi_ui,
                    },
                    "training_config": {
                        "output_dir": out_dir_s_ui, "batch_size": batch_s_ui, "learning_rate": lr_s_ui,
                        "num_epochs": epochs_s_ui, "eval_every_n_epochs": eval_n_epochs_ui,
                        "use_wandb": use_wb_s_ui, "wandb_project": wb_proj_s_ui if use_wb_s_ui else None,
                        "gradient_accumulation_steps": 4, "fp16": True, 
                        "gpu_memory_fraction": gpu_mem_frac_ui, "use_gradient_checkpointing": True,
                        "dynamic_batch_sizing": True, "checkpoint_steps": chkpt_steps_ui,
                        "iteration_checkpoint_steps": iter_chkpt_steps_ui,
                        "gpu_ids": selected_gpus, "distributed_training": dist_train_ui if selected_gpus and len(selected_gpus) > 1 else False,
                    }
                }
                
                config_file_path = os.path.join(temp_train_dir, "training_run_config.json")
                with open(config_file_path, "w", encoding="utf-8") as f: json.dump(train_run_config, f, indent=2)
                
                import threading
                def training_thread_fn(cfg_path, tmp_dir):
                    try:
                        from src.training.pipeline import train_from_config
                        train_from_config(cfg_path)
                        logger.info(f"Training thread completed for {cfg_path}.")
                    except Exception as e_thr:
                        logger.error(f"Error in training thread for {cfg_path}: {e_thr}", exc_info=True)
                    finally:
                        try: shutil.rmtree(tmp_dir)
                        except Exception as e_rm_tmp: logger.error(f"Error removing temp dir {tmp_dir}: {e_rm_tmp}")
                
                thread = threading.Thread(target=training_thread_fn, args=(config_file_path, temp_train_dir))
                thread.daemon = True
                thread.start()
                
                return f"Training initiated in background. Config: {config_file_path}. Check console/logs in '{out_dir_s_ui}'."

            train_btn.click(
                start_training_ui_handler,
                inputs=[
                    model_size_dropdown, multimodal_training_checkbox, use_expert_system_training_checkbox,
                    train_data_file, val_data_file, vocab_data_file, image_dir_textbox,
                    batch_size_slider, learning_rate_slider, num_epochs_slider, eval_every_n_epochs_slider,
                    checkpoint_steps_slider, iteration_checkpoint_steps_slider,
                    gpu_selection_checkboxgroup, distributed_training_checkbox, gpu_memory_fraction_slider,
                    output_dir_textbox, use_wandb_checkbox, wandb_project_textbox,
                ],
                outputs=[training_output_textbox],
            )
        
        interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share)