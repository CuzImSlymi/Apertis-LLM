# /home/ubuntu/ApertisAI_Project/Apertis AI_/src/inference/interface.py
import gradio as gr
import torch
import logging
import json
import os
import threading
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image

# Add project root to sys.path to allow imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.core import ApertisConfig, ApertisForCausalLM
from src.inference.tokenizer import HFTokenizer # Use the dedicated tokenizer class
from src.training.pipeline import YoloStyleTrainingPipeline, get_available_gpus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ApertisInterface:
    """Gradio interface for interacting with Apertis models."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        multimodal: bool = False,
        device: Optional[str] = None,
        web: bool = True, # Launch web UI by default
        port: int = 7860,
        share: bool = False,
    ):
        self.model_path = model_path
        self.vocab_file = vocab_file
        self.multimodal = multimodal
        self.model: Optional[ApertisForCausalLM] = None
        self.tokenizer: Optional[HFTokenizer] = None
        self.vocab: Optional[Dict[str, int]] = None
        self.reverse_vocab: Optional[Dict[int, str]] = None
        self.training_pipeline: Optional[YoloStyleTrainingPipeline] = None
        self.training_thread: Optional[threading.Thread] = None
        self.stop_training_flag = threading.Event() # Flag to signal training stop
        self.port = port
        self.share = share

        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Pre-load model and vocab if paths are provided
        if self.vocab_file:
            try:
                self.load_vocabulary(self.vocab_file)
            except Exception as e:
                logger.error(f"Failed to preload vocabulary: {e}")
        if self.model_path:
            try:
                self.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Failed to preload model: {e}")

        # Launch web interface if requested
        if web:
            logger.info("Launching Apertis web interface...")
            self.launch_web_interface()

    def load_model(self, model_path: str):
        """Load model weights and configuration.

        Instantiates the model using the configuration saved within the checkpoint directory.
        """
        logger.info(f"Loading model from {model_path}")
        self.model_path = model_path # Store path

        try:
            # Determine if path is a directory (checkpoint) or a single file
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
                weights_path = os.path.join(model_path, "model.pt")
            elif os.path.isfile(model_path) and model_path.endswith(".pt"):
                # Assume config.json is in the same directory as the .pt file
                weights_path = model_path
                config_path = os.path.join(os.path.dirname(model_path), "config.json")
            else:
                raise FileNotFoundError(f"Invalid model path: {model_path}. Expecting directory or .pt file.")

            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found at {config_path}")
            if not os.path.exists(weights_path):
                raise FileNotFoundError(f"Model weights file not found at {weights_path}")

            # 1. Load config directly from the checkpoint directory
            logger.info(f"Loading configuration from {config_path}")
            # Use the from_pretrained classmethod which handles loading from file
            checkpoint_config = ApertisConfig.from_pretrained(config_path)
            logger.info(f"Loaded checkpoint config: {checkpoint_config.to_dict()}")

            # 2. Instantiate the model using the *checkpoint's* configuration
            logger.info("Instantiating model architecture based on checkpoint config...")
            self.model = ApertisForCausalLM(checkpoint_config)
            self.model.eval() # Set to evaluation mode
            self.model.to(self.device)
            logger.info(f"Instantiated model with config: {self.model.config.to_dict()}")

            # 3. Load state dict from the checkpoint file
            logger.info(f"Loading state dictionary from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)

            # 4. Load state dict with strict=False to handle potential minor mismatches
            #    (e.g., newly added buffers like RoPE cache, minor refactoring)
            #    The major architecture dimensions should now match due to using checkpoint_config.
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

            # Refine logging for missing/unexpected keys
            actual_missing = []
            if missing_keys:
                benign_missing = [k for k in missing_keys if "rotary_emb.cos_cached" in k or "rotary_emb.sin_cached" in k]
                actual_missing = [k for k in missing_keys if k not in benign_missing]
                if benign_missing:
                     logger.info(f"Ignoring missing keys related to RoPE cache: {benign_missing}")
                if actual_missing:
                     logger.warning(f"Potentially problematic missing keys when loading state_dict: {actual_missing}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")

            if not actual_missing and not unexpected_keys:
                 logger.info("State dictionary loaded successfully.")
            else:
                 logger.warning("State dictionary loaded with some mismatches (strict=False). Review warnings carefully.")

            # Update multimodal flag based on loaded model's config
            self.multimodal = self.model.config.multimodal

            # Resize embeddings if tokenizer is already loaded and sizes differ
            self._resize_embeddings_if_needed()

            logger.info("Model loaded successfully.")

        except FileNotFoundError as e:
            logger.error(f"Error loading model: {e}")
            self.model = None # Ensure model is None on failure
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading model: {e}", exc_info=True)
            self.model = None # Ensure model is None on failure
            # Reraise as RuntimeError for clarity in UI
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def load_vocabulary(self, vocab_file: str):
        """Load vocabulary and initialize tokenizer."""
        try:
            if not os.path.isfile(vocab_file):
                raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")

            logger.info(f"Loading vocabulary from {vocab_file}")
            self.vocab_file = vocab_file # Store the path

            # Load the vocab dictionary from the file for internal use (e.g., fallback BOS ID)
            with open(vocab_file, "r", encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Handle different vocabulary formats to get a {token: id} dict
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    self.vocab = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                    logger.info(f"Converted list-based vocabulary to dictionary format with {len(self.vocab)} tokens")
                elif all(isinstance(k, str) and isinstance(v, int) for k, v in vocab_data.items()):
                    self.vocab = vocab_data
                elif "model" in vocab_data and "vocab" in vocab_data["model"] and isinstance(vocab_data["model"]["vocab"], dict):
                     logger.info("Extracting vocab dict from 'model.vocab' structure in JSON")
                     self.vocab = {k: int(v) for k, v in vocab_data["model"]["vocab"].items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
                else:
                    # Attempt to handle simple {token: id} dict even if values aren't strictly int
                    try:
                        self.vocab = {str(k): int(v) for k, v in vocab_data.items()}
                        logger.info(f"Loaded vocabulary dictionary with {len(self.vocab)} tokens (converted values to int)." )
                    except (ValueError, TypeError):
                         raise ValueError("Unsupported vocabulary dictionary format in JSON file.")
            else:
                raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")

            # Create reverse vocabulary for decoding (if needed, though tokenizer handles this)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

            logger.info(f"Vocabulary dictionary loaded with {len(self.vocab)} tokens")

            # Initialize the tokenizer using the vocab file path
            # The HFTokenizer class now handles loading logic internally
            self.tokenizer = HFTokenizer(vocab_file=self.vocab_file)
            logger.info("HF Tokenizer initialized.")

            # Attempt to resize embeddings if model is already loaded
            self._resize_embeddings_if_needed()

        except Exception as e:
            logger.error(f"Error loading vocabulary or initializing tokenizer: {e}")
            self.tokenizer = None # Ensure tokenizer is None on failure
            self.vocab = None
            self.reverse_vocab = None
            # Raise the exception instead of creating a minimal fallback here
            raise RuntimeError(f"Failed to load vocabulary from {vocab_file}: {e}")

    def _resize_embeddings_if_needed(self):
        """Internal helper to resize model embeddings if model and tokenizer are loaded and sizes mismatch."""
        if not self.model or not self.tokenizer:
            # Cannot resize if either model or tokenizer isn't loaded
            return

        try:
            loaded_vocab_size = self.tokenizer.get_vocab_size()
            if loaded_vocab_size <= 1: # Treat size 0 or 1 (fallback) as failure
                logger.error(
                    f"Tokenizer reported minimal size ({loaded_vocab_size}). "
                    f"Skipping model embedding resizing. Model vocab size remains {self.model.config.vocab_size}. "
                    f"Generation will likely fail."
                )
                return

            # Use the vocab size from the *model's current config* for comparison
            current_model_config_vocab_size = self.model.config.vocab_size
            if hasattr(self.model, "resize_token_embeddings") and current_model_config_vocab_size != loaded_vocab_size:
                logger.warning(
                    f"Model config vocabulary size ({current_model_config_vocab_size}) doesn't match loaded "
                    f"tokenizer vocabulary size ({loaded_vocab_size}). Resizing model embeddings."
                )
                try:
                    self.model.resize_token_embeddings(loaded_vocab_size)
                    # IMPORTANT: The resize_token_embeddings method should update the config internally
                    # self.model.config.vocab_size = loaded_vocab_size # No longer needed if method updates config
                    logger.info(f"Resized model token embeddings to {loaded_vocab_size}. Updated model config.")
                except Exception as resize_err:
                    logger.error(f"Error resizing token embeddings: {resize_err}. Model vocab size remains {current_model_config_vocab_size}")
            elif current_model_config_vocab_size == loaded_vocab_size:
                 logger.info(f"Model config and tokenizer vocabulary sizes match ({loaded_vocab_size}). No resizing needed.")

        except Exception as e:
            logger.error(f"Error during embedding resize check: {e}")

    # Removed create_token_mapping as resizing is preferred

    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded HF tokenizer."""
        if not self.tokenizer:
            logger.error("Tokenizer not initialized. Cannot tokenize.")
            return []

        # Use the HFTokenizer's encode method
        # Set add_special_tokens=False because BOS/EOS are often handled during generation prep
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text using the loaded HF tokenizer."""
        if not self.tokenizer:
            logger.error("Tokenizer not initialized. Cannot detokenize.")
            return "[Detokenization Error]"

        # Use the HFTokenizer's decode method
        # skip_special_tokens=True is common for displaying generated text
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        try:
            from torchvision import transforms

            # Define image transformations based on model config if available
            image_size = getattr(self.model.config, 'image_size', 224)
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            raise ValueError(f"Failed to preprocess image: {e}") # Raise error instead of returning zeros

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
        if self.tokenizer is None:
             return "Tokenizer not loaded. Please load a vocabulary first."

        try:
            # Tokenize prompt
            input_ids_list = self.tokenize(prompt)

            # Get BOS token ID from the tokenizer
            bos_token_id = None
            if self.tokenizer and self.tokenizer.tokenizer:
                # Try standard <bos> first, then <start>
                bos_token_id = self.tokenizer.tokenizer.token_to_id("<bos>")
                if bos_token_id is None:
                    bos_token_id = self.tokenizer.tokenizer.token_to_id("<start>")

            # Fallback if BOS token not found or tokenizer failed
            if bos_token_id is None:
                logger.warning("BOS token (<bos> or <start>) not found in tokenizer, using default ID 1.")
                # Try getting ID from internal vocab dict as another fallback
                bos_token_id = self.vocab.get("<bos>") or self.vocab.get("<start>") or 1

            # Add <bos> token if not present
            if not input_ids_list or input_ids_list[0] != bos_token_id:
                input_ids_list = [bos_token_id] + input_ids_list

            # Convert to tensor
            input_ids = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)

            # --- Token Index Validation ---
            # Use the *actual* vocab size of the potentially resized model
            current_model_vocab_size = self.model.get_input_embeddings().weight.size(0)
            invalid_indices = input_ids >= current_model_vocab_size
            if torch.any(invalid_indices):
                num_invalid = torch.sum(invalid_indices).item()
                logger.error(
                    f"Input contains {num_invalid} token ID(s) >= model vocab size ({current_model_vocab_size}). "
                    f"Max ID found: {torch.max(input_ids)}. "
                    f"This likely means the tokenizer/vocab doesn't match the original model training or resizing failed. "
                    f"Clamping invalid IDs to {current_model_vocab_size - 1}. Generation quality WILL be affected."
                )
                # Clamp invalid indices to the maximum valid index
                input_ids = torch.clamp(input_ids, max=current_model_vocab_size - 1)
            # --- End Validation ---

            # Create attention mask (all 1s for input tokens)
            attention_mask = torch.ones_like(input_ids)
            # Prepare image if provided and model is multimodal
            pixel_values = None
            if image_path is not None and self.multimodal:
                pixel_values = self.preprocess_image(image_path).to(self.device)

            # Get EOS token ID for stopping generation
            eos_token_id = None
            if self.tokenizer and self.tokenizer.tokenizer:
                 eos_token_id = self.tokenizer.tokenizer.token_to_id("<eos>")
                 if eos_token_id is None:
                     eos_token_id = self.tokenizer.tokenizer.token_to_id("<end>")
            # Fallback for EOS ID
            if eos_token_id is None:
                 logger.warning("EOS token (<eos> or <end>) not found in tokenizer. Generation might not stop correctly.")
                 # Try getting ID from internal vocab dict
                 eos_token_id = self.vocab.get("<eos>") or self.vocab.get("<end>")
                 # If still None, generation might run to max_length

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_length=max_length + len(input_ids_list), # Adjust max_length based on input
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    eos_token_id=eos_token_id, # Pass EOS token ID
                    pad_token_id=self.tokenizer.tokenizer.token_to_id("<pad>") if self.tokenizer and self.tokenizer.tokenizer else 0 # Pass PAD ID
                )

            # Detokenize response (excluding input tokens)
            response_ids = output_ids[0, len(input_ids_list):].tolist()
            response = self.detokenize(response_ids)

            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True) # Log traceback
            return f"Error generating response: {str(e)}"

    def chat(
        self,
        message: str,
        history: List[Tuple[str, str]], # Gradio chat history format
        image_path: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """Chat with the model, taking Gradio history format."""
        # Create prompt from chat history
        prompt = ""
        for user_msg, assistant_msg in history:
            prompt += f"User: {user_msg}\n"
            if assistant_msg is not None:
                prompt += f"Assistant: {assistant_msg}\n"

        prompt += f"User: {message}\nAssistant: "

        # Generate response
        response = self.generate_response(
            prompt=prompt,
            image_path=image_path,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        # Add assistant response to history for Gradio
        history.append((message, response))

        return "", history # Return empty string for textbox, updated history for chatbot

    def reset_chat(self) -> Tuple[List, str]:
        """Reset chat history and clear message box."""
        self.chat_history = [] # Keep internal history if needed, but clear Gradio's
        return [], "" # Return empty list for chatbot, empty string for textbox

    # --- Training Methods --- 
    def _run_training_with_progress(self, pipeline: YoloStyleTrainingPipeline, progress: gr.Progress):
        """Wraps the pipeline's train method to handle Gradio progress and stop event."""
        try:
            # Pass the stop event and progress tracker to the pipeline's train method
            pipeline.train(stop_event=self.stop_training_flag, progress_callback=progress)
            if self.stop_training_flag.is_set():
                 logger.info("Training stopped by user.")
                 progress(1, desc="Training stopped by user.")
            else:
                 logger.info("Training finished successfully.")
                 progress(1, desc="Training finished successfully.")
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            # Update progress bar on error
            try:
                progress(1, desc=f"Training failed: {e}")
            except Exception as pe:
                 logger.error(f"Failed to update progress on error: {pe}")
        finally:
            # Ensure flag is cleared and thread reference removed after training stops or fails
            self.stop_training_flag.clear()
            self.training_thread = None
            self.training_pipeline = None # Release pipeline object
            logger.info("Training thread finished.")
            # TODO: Add mechanism to re-enable UI elements if needed

    def start_training(
        self,
        model_size: str, # Changed from config to size string
        multimodal: bool,
        use_expert_system: bool,
        train_data_file: Optional[tempfile._TemporaryFileWrapper],
        val_data_file: Optional[tempfile._TemporaryFileWrapper],
        vocab_data_file: Optional[tempfile._TemporaryFileWrapper],
        image_dir: Optional[str],
        batch_size: int,
        learning_rate: float,
        num_epochs: int,
        checkpoint_steps: int,
        iteration_checkpoint_steps: int,
        gpu_selection: List[str],
        distributed_training: bool,
        gpu_memory_fraction: float,
        output_dir: str,
        use_wandb: bool,
        wandb_project: str,
        progress=gr.Progress() # Add Gradio progress tracker
    ) -> str:
        """Starts the training process in a separate thread."""
        if self.training_thread is not None and self.training_thread.is_alive():
            return "Training is already in progress."

        # --- Input Validation ---
        if not train_data_file:
            return "Error: Training data file is required."
        if not vocab_data_file:
            return "Error: Vocabulary file is required."
        if multimodal and (not image_dir or not os.path.isdir(image_dir)):
            # Check if image_dir is empty or not a directory
            return "Error: A valid Image directory is required for multimodal training."
        if not output_dir:
             return "Error: Output directory is required."

        # Ensure output directory exists
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return f"Error creating output directory: {e}"

        # Get absolute paths for temp files (Gradio provides temp file wrappers)
        train_data_path = train_data_file.name
        vocab_path = vocab_data_file.name
        val_data_path = val_data_file.name if val_data_file else None

        # --- Prepare Training Pipeline ---
        try:
            progress(0, desc="Initializing training...")
            # Determine vocab size from the provided vocab file
            try:
                with open(vocab_path, "r", encoding='utf-8') as f:
                    # Simple way to count tokens assuming {token: id} format or {"tokens": [...]} format
                    vocab_json = json.load(f)
                    if isinstance(vocab_json, dict):
                         if "tokens" in vocab_json and isinstance(vocab_json["tokens"], list):
                             actual_vocab_size = len(vocab_json["tokens"])
                         else:
                             actual_vocab_size = len(vocab_json)
                    else:
                         raise ValueError("Cannot determine vocab size from file format.")
                    # Add buffer for potential special tokens added by tokenizer/pipeline
                    # A better approach is to let the tokenizer handle this, but for now:
                    actual_vocab_size += 10 # Add a small buffer
            except Exception as ve:
                 logger.error(f"Could not determine vocab size from {vocab_path}: {ve}. Using default 32000.")
                 actual_vocab_size = 32000 # Fallback

            # Create model configuration based on UI selections
            # Map model_size string to actual parameters
            size_map = {
                "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
            }
            if model_size not in size_map:
                 return f"Error: Invalid model size \t'{model_size}\t'. Choose from {list(size_map.keys())}."
            model_params = size_map[model_size]

            # Explicitly create config with dimensions from size_map
            config = ApertisConfig(
                vocab_size=actual_vocab_size, # Use determined vocab size
                hidden_size=model_params["hidden_size"],
                num_hidden_layers=model_params["num_hidden_layers"],
                num_attention_heads=model_params["num_attention_heads"],
                intermediate_size=model_params["intermediate_size"],
                multimodal=multimodal,
                use_expert_system=use_expert_system,
                # Add other relevant config options if needed from defaults or UI
                # e.g., max_position_embeddings=2048, etc.
                # Let's keep defaults for others unless specified in UI
                model_size=model_size # Store the size string too
            )
            logger.info(f"Created training config: {config.to_dict()}")

            # GPU IDs as integers
            gpu_ids = [int(gid) for gid in gpu_selection] if gpu_selection else None
            use_distributed = distributed_training and gpu_ids and len(gpu_ids) > 1

            # Initialize the pipeline
            self.training_pipeline = YoloStyleTrainingPipeline(
                config=config,
                train_data_path=train_data_path,
                vocab_path=vocab_path,
                output_dir=output_dir,
                val_data_path=val_data_path,
                image_dir=image_dir,
                batch_size=batch_size,
                learning_rate=learning_rate,
                num_epochs=num_epochs,
                checkpoint_steps=checkpoint_steps,
                iteration_checkpoint_steps=iteration_checkpoint_steps,
                gpu_ids=gpu_ids,
                use_distributed=use_distributed,
                # local_rank will be handled by torchrun/distributed launcher if use_distributed is True
                gpu_memory_fraction=gpu_memory_fraction,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                # Pass other relevant params like fp16, warmup_steps etc. if needed
            )

            # Clear previous stop flag
            self.stop_training_flag.clear()

            # Start training in a separate thread, passing the progress tracker
            self.training_thread = threading.Thread(
                target=self._run_training_with_progress, # Use the wrapper method
                args=(self.training_pipeline, progress),
                daemon=True # Allows main thread to exit even if training thread is running
            )
            self.training_thread.start()

            return "Training started in the background. Monitor progress below."

        except Exception as e:
            logger.error(f"Failed to initialize training pipeline: {e}", exc_info=True)
            self.training_pipeline = None # Ensure pipeline is None on failure
            return f"Error initializing training: {e}"

    def stop_training(self) -> str:
        """Signals the training thread to stop."""
        if self.training_thread is None or not self.training_thread.is_alive():
            return "No active training process to stop."

        if self.stop_training_flag.is_set():
             return "Stop signal already sent."

        logger.info("Sending stop signal to training thread...")
        self.stop_training_flag.set()
        return "Stop signal sent. Training will stop shortly."

    # --- Model Management Methods ---
    def handle_load_model(
        self,
        model_path: str,
        vocab_path: str,
        # Add advanced options if needed later
    ) -> str:
        """Handles loading model and vocab from the UI."""
        model_info_text = ""
        try:
            if not model_path:
                return "Error: Model path is required."
            if not vocab_path:
                 return "Error: Vocabulary path is required."

            # Load vocabulary first, as it might be needed for model loading/resizing
            self.load_vocabulary(vocab_path)
            model_info_text += f"Vocabulary loaded from: {vocab_path}\n"
            model_info_text += f"Tokenizer Vocab Size: {self.tokenizer.get_vocab_size() if self.tokenizer else 'N/A'}\n\n"

            # Load model (now uses checkpoint's config)
            self.load_model(model_path)
            model_info_text += f"Model loaded from: {model_path}\n"
            # Display the *actual* config of the loaded model
            model_info_text += f"Loaded Model Config: {self.model.config.to_dict() if self.model else 'N/A'}\n"
            model_info_text += f"Model Vocab Size (after potential resize): {self.model.get_input_embeddings().weight.size(0) if self.model else 'N/A'}\n"

            # Update internal state (paths)
            self.model_path = model_path
            self.vocab_file = vocab_path

            return f"Model and vocabulary loaded successfully.\n\n{model_info_text}"

        except Exception as e:
            logger.error(f"Error loading model/vocab via UI: {e}", exc_info=True)
            # Return the specific error message (e.g., from compatibility check)
            return f"Error: {e}"

    def handle_create_model(
        self,
        model_size: str,
        multimodal: bool,
        use_expert_system: bool,
        vocab_size: int,
        save_dir: str,
        progress=gr.Progress()
    ) -> str:
        """Handles creating a new model from the UI."""
        if not save_dir:
            return "Error: Save directory is required."
        if vocab_size <= 0:
             return "Error: Vocabulary size must be positive."

        try:
            progress(0, desc="Creating output directory...")
            os.makedirs(save_dir, exist_ok=True)

            progress(0.2, desc="Creating model configuration...")
            # Map model_size string to actual parameters
            size_map = {
                "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
            }
            if model_size not in size_map:
                 return f"Error: Invalid model size \t'{model_size}\t'. Choose from {list(size_map.keys())}."
            model_params = size_map[model_size]

            # Explicitly create config with dimensions from size_map
            config = ApertisConfig(
                vocab_size=vocab_size,
                hidden_size=model_params["hidden_size"],
                num_hidden_layers=model_params["num_hidden_layers"],
                num_attention_heads=model_params["num_attention_heads"],
                intermediate_size=model_params["intermediate_size"],
                multimodal=multimodal,
                use_expert_system=use_expert_system,
                model_size=model_size # Store the size string too
            )

            progress(0.4, desc="Initializing model...")
            model = ApertisForCausalLM(config)
            model.eval() # Set to eval mode

            progress(0.7, desc="Saving model configuration...")
            # Use the config's save method
            config.save_pretrained(save_dir)
            # config_path = os.path.join(save_dir, "config.json")
            # with open(config_path, "w") as f:
            #     json.dump(config.to_dict(), f, indent=4)

            progress(0.9, desc="Saving initial model weights...")
            model_path = os.path.join(save_dir, "model.pt")
            torch.save(model.state_dict(), model_path)

            progress(1, desc="Model created successfully!")
            return f"New model created successfully and saved in '{save_dir}'.\nConfig: {config.to_dict()}"

        except Exception as e:
            logger.error(f"Error creating new model: {e}", exc_info=True)
            return f"Error creating model: {e}"


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
                            chatbot = gr.Chatbot(label="Chat History", height=500)
                            with gr.Row():
                                message = gr.Textbox(
                                    placeholder="Type your message here...",
                                    show_label=False,
                                    scale=5
                                )
                                submit_btn = gr.Button("Send", scale=1)

                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")

                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="filepath",
                                label="Upload Image (optional)",
                                visible=self.multimodal # Show only if interface launched in multimodal mode
                            )
                            with gr.Accordion("Generation Settings", open=False):
                                max_length = gr.Slider(
                                    minimum=10,
                                    maximum=2048, # Increased max length
                                    value=150,
                                    step=10,
                                    label="Max New Tokens",
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
                            # Use model_size dropdown for creating/training new models
                            train_model_size = gr.Dropdown(
                                choices=["small", "base", "large"],
                                value="base",
                                label="Model Size (for new training)",
                            )
                            train_multimodal_checkbox = gr.Checkbox(
                                value=False, # Default to False for new training
                                label="Multimodal (Text + Image)",
                            )
                            train_use_expert_system = gr.Checkbox(
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
                            train_image_dir = gr.Textbox(
                                label="Image Directory (for multimodal training)",
                                placeholder="/path/to/images",
                                visible=False, # Start hidden
                            )

                            # Update image_dir visibility based on multimodal checkbox
                            train_multimodal_checkbox.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[train_multimodal_checkbox],
                                outputs=[train_image_dir],
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("## Training Parameters")
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=64,
                                value=4,
                                step=1,
                                label="Batch Size (per GPU)",
                            )
                            learning_rate = gr.Slider(
                                minimum=1e-6,
                                maximum=1e-3,
                                value=5e-5,
                                step=1e-6,
                                label="Learning Rate",
                                elem_id="learning_rate_slider" # Added for potential JS interaction
                            )
                            num_epochs = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=3,
                                step=1,
                                label="Number of Epochs",
                            )

                            # Add checkpoint frequency controls
                            with gr.Accordion("Checkpoint Settings", open=False):
                                checkpoint_steps = gr.Slider(
                                    minimum=0,
                                    maximum=10000, # Increased max
                                    value=1000,
                                    step=100,
                                    label="Global Step Checkpoint Frequency (0 to disable)",
                                )
                                iteration_checkpoint_steps = gr.Slider(
                                    minimum=0,
                                    maximum=5000, # Increased max
                                    value=0, # Default to 0 (disabled)
                                    step=10,
                                    label="Iteration Checkpoint Frequency (0 to disable)",
                                )
                                gr.Markdown("""
                                * **Global Step Checkpoint**: Save checkpoint every N global steps (across epochs)
                                * **Iteration Checkpoint**: Save checkpoint every N iterations within each epoch
                                """, elem_classes=["small-text"])

                            # Add GPU management controls
                            with gr.Accordion("GPU Settings", open=False):
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
                                        value=[gpu_choices[0]] if gpu_choices else [],  # Default to first GPU if available
                                        label="Select GPUs for Training",
                                    )

                                    distributed_training = gr.Checkbox(
                                        value=False,
                                        label="Use Distributed Training (DDP - Recommended for >1 GPU)",
                                        visible=len(gpu_choices) > 1 # Only show if multiple GPUs exist
                                    )

                                    # Update distributed training visibility based on selection count
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
                                    # Create hidden elements to avoid errors in handler
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
                                    value=0.9, # Default higher
                                    step=0.05,
                                    label="GPU Memory Fraction (Informational, not strict limit)",
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

                            # Training Controls and Output
                            with gr.Row():
                                train_btn = gr.Button("Start Training", variant="primary")
                                stop_btn = gr.Button("Stop Training", variant="stop") # Added Stop Button
                            training_output = gr.Textbox(
                                label="Training Status / Output",
                                interactive=False,
                                lines=10,
                                placeholder="Training logs will appear here..."
                            )

                # Model tab
                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model & Vocabulary")
                            model_path_input = gr.Textbox(
                                label="Model Path (Directory or .pt file)",
                                value=self.model_path or "",
                                placeholder="/path/to/checkpoint_dir or /path/to/model.pt",
                            )
                            vocab_path_input = gr.Textbox(
                                label="Vocabulary Path (JSON)",
                                value=self.vocab_file or "",
                                placeholder="/path/to/vocab.json",
                            )

                            load_model_btn = gr.Button("Load Model & Vocab", variant="primary")
                            model_info = gr.Textbox(
                                label="Load Status / Model Information",
                                interactive=False,
                                lines=15, # Increased lines
                                placeholder="Status messages will appear here..."
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
                            new_model_save_dir = gr.Textbox(
                                label="Save Directory",
                                placeholder="/path/to/save/new_model",
                            )
                            create_model_btn = gr.Button("Create & Save Model", variant="primary")
                            create_model_output = gr.Textbox(
                                label="Create Model Status",
                                interactive=False,
                                lines=5,
                                placeholder="Status messages will appear here..."
                            )

            # --- Define Event Handlers ---

            # Chat Tab Handlers
            submit_btn.click(
                fn=self.chat,
                inputs=[
                    message,
                    chatbot, # Pass chatbot history
                    image_input,
                    max_length,
                    temperature,
                    top_k,
                    top_p,
                ],
                outputs=[message, chatbot], # Clear message box, update chatbot
            )
            message.submit( # Allow submitting with Enter key
                 fn=self.chat,
                 inputs=[
                     message,
                     chatbot,
                     image_input,
                     max_length,
                     temperature,
                     top_k,
                     top_p,
                 ],
                 outputs=[message, chatbot],
            )
            clear_btn.click(
                fn=self.reset_chat,
                inputs=[],
                outputs=[chatbot, message], # Clear chatbot and message box
            )

            # Training Tab Handlers
            train_btn.click(
                fn=self.start_training,
                inputs=[
                    train_model_size, # Use the training-specific dropdown
                    train_multimodal_checkbox,
                    train_use_expert_system,
                    train_data,
                    val_data,
                    vocab_data,
                    train_image_dir,
                    batch_size,
                    learning_rate,
                    num_epochs,
                    checkpoint_steps,
                    iteration_checkpoint_steps,
                    gpu_selection,
                    distributed_training,
                    gpu_memory_fraction,
                    output_dir,
                    use_wandb,
                    wandb_project,
                ],
                outputs=[training_output],
                # Add api_name="start_training" ?
            )
            # Stop Button Handler
            stop_btn.click(
                fn=self.stop_training,
                inputs=[],
                outputs=[training_output],
                # Add api_name="stop_training" ?
            )

            # Model Tab Handlers
            load_model_btn.click(
                fn=self.handle_load_model,
                inputs=[
                    model_path_input,
                    vocab_path_input,
                ],
                outputs=[model_info],
                # Add api_name="load_model" ?
            )
            create_model_btn.click(
                fn=self.handle_create_model,
                inputs=[
                    new_model_size,
                    new_model_multimodal,
                    new_model_expert,
                    new_model_vocab_size,
                    new_model_save_dir,
                ],
                outputs=[create_model_output],
                # Add api_name="create_model" ?
            )

        # Launch the interface with queuing enabled
        interface.queue().launch(server_name="0.0.0.0", server_port=self.port, share=self.share)

# Example usage (if run directly)
if __name__ == "__main__":
    # Example: Launch web interface without preloading model/vocab
    # User will load via the UI
    # Example: python src/inference/interface.py --web
    import argparse
    parser = argparse.ArgumentParser(description="Apertis AI Interface")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-load model")
    parser.add_argument("--vocab_file", type=str, default=None, help="Path to pre-load vocabulary")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal features")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., 'cpu', 'cuda:0')")
    parser.add_argument("--web", action="store_true", help="Launch the web interface")
    parser.add_argument("--port", type=int, default=7860, help="Port for the web interface")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    args = parser.parse_args()

    # Create and potentially launch the interface
    app = ApertisInterface(
        model_path=args.model_path,
        vocab_file=args.vocab_file,
        multimodal=args.multimodal,
        device=args.device,
        web=args.web, # Control launch via argument
        port=args.port,
        share=args.share,
    )

    # If --web was not provided, the interface object is created but not launched.
    # You could potentially use app methods programmatically here.
    if not args.web:
        logger.info("Interface object created. Use --web to launch the Gradio UI.")

