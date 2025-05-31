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
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model
from src.training.pipeline import YoloStyleTrainingPipeline, ApertisPretrainDataset, get_available_gpus, create_sample_config as create_training_sample_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApertisInterface:
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
        self.model_path_arg = model_path 
        self.vocab_file_fallback_arg = vocab_file 
        self.multimodal = multimodal
        self.web = web
        self.port = port
        self.share = share

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")

        self.model: Optional[ApertisForCausalLM] = None
        self.vocab: Optional[Dict[str, int]] = None 
        self.reverse_vocab: Optional[Dict[int, str]] = None 
        self.token_mapping: Optional[Dict[int, int]] = None 

        self.hf_tokenizer_chat = None
        self.actual_model_path_loaded = None 
        self.actual_tokenizer_path_loaded = None 


        if self.model_path_arg is not None:
            self.load_model_and_tokenizer_from_path(self.model_path_arg, vocab_file_override=self.vocab_file_fallback_arg)
        else:
            logger.info("No initial model path provided. Please load a model via the UI or CLI.")
            self._create_dummy_model_and_vocab_for_ui_startup()


        self.chat_history: List[Dict[str,str]] = []

        self.standard_training_stop_event = threading.Event()
        self.azr_training_stop_event = threading.Event()
        self.finetune_training_stop_event = threading.Event()
        self.standard_training_thread: Optional[threading.Thread] = None
        self.azr_training_thread: Optional[threading.Thread] = None
        self.finetune_training_thread: Optional[threading.Thread] = None


        if web:
            self.launch_web_interface()

    def _create_dummy_model_and_vocab_for_ui_startup(self):
        """Creates a minimal model and vocab if none loaded, so UI can start."""
        logger.info("Creating a dummy model and vocab for UI startup as no model was specified.")
        dummy_config = ApertisConfig(vocab_size=100, hidden_size=64, num_hidden_layers=1, num_attention_heads=1, intermediate_size=128)
        self.model = ApertisForCausalLM(dummy_config)
        self.model.to(self.device)
        self.model.eval()
        self.actual_model_path_loaded = "Dummy Startup Model"

        self.vocab = {f"<dummy_token_{i}>": i for i in range(100)}
        self.vocab.update({"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3})
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.actual_tokenizer_path_loaded = "Dummy Startup Vocab"
        self.multimodal = dummy_config.multimodal


    def _attempt_load_hf_tokenizer(self, path_to_check: str) -> bool:
        try:
            from transformers import AutoTokenizer
            is_tokenizer_dir = os.path.isdir(path_to_check) and (
                os.path.exists(os.path.join(path_to_check, "tokenizer.json")) or
                (os.path.exists(os.path.join(path_to_check, "vocab.json")) and os.path.exists(os.path.join(path_to_check, "merges.txt"))) or
                os.path.exists(os.path.join(path_to_check, "tokenizer_config.json"))
            )

            if is_tokenizer_dir:
                self.hf_tokenizer_chat = AutoTokenizer.from_pretrained(path_to_check)
                logger.info(f"Successfully loaded Hugging Face tokenizer from directory: {path_to_check}")
                self.actual_tokenizer_path_loaded = path_to_check
                return True
            
            # Try loading as a Hub ID or a direct file path (less common for AutoTokenizer but worth a try)
            # This needs to be guarded, as from_pretrained can raise if path_to_check is not a valid ID / local path
            if not os.path.isdir(path_to_check): # If it's not a directory, it might be a Hub ID
                try:
                    self.hf_tokenizer_chat = AutoTokenizer.from_pretrained(path_to_check)
                    logger.info(f"Successfully loaded Hugging Face tokenizer from Hub ID/path: {path_to_check}")
                    self.actual_tokenizer_path_loaded = path_to_check 
                    return True
                except EnvironmentError: # Handles cases where it's not a valid Hub ID or local file/dir
                    logger.debug(f"'{path_to_check}' is not a local tokenizer directory and failed to load as Hub ID.")
                    pass # Fall through, it wasn't a Hub ID or directly loadable path
        except ImportError:
            logger.error("Hugging Face Transformers library not found. Cannot load HF tokenizer.")
        except Exception as e:
            logger.warning(f"Could not load Hugging Face tokenizer from {path_to_check}: {e}")
        return False

    def load_model_and_tokenizer_from_path(self, model_path_or_name: str, vocab_file_override: Optional[str] = None):
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.token_mapping = None
        self.hf_tokenizer_chat = None
        self.actual_model_path_loaded = model_path_or_name 
        self.vocab_file_fallback_arg = vocab_file_override

        tokenizer_found_and_loaded = False
        potential_tokenizer_base_dir = model_path_or_name
        if os.path.isfile(model_path_or_name): # If model path is file, tokenizer is likely in same dir
            potential_tokenizer_base_dir = os.path.dirname(model_path_or_name)
        
        # Priority 1: Check the determined base directory for HF tokenizer files
        if os.path.isdir(potential_tokenizer_base_dir):
            logger.info(f"Checking for HF tokenizer in potential base directory: {potential_tokenizer_base_dir}")
            if self._attempt_load_hf_tokenizer(potential_tokenizer_base_dir):
                tokenizer_found_and_loaded = True
        
        # Priority 2: If model_path_or_name itself was a different string (e.g. Hub ID, or user *thought* it was tokenizer path)
        if not tokenizer_found_and_loaded and model_path_or_name != potential_tokenizer_base_dir:
            logger.info(f"Base directory tokenizer check failed or paths differ. Checking original path '{model_path_or_name}' as HF tokenizer source.")
            if self._attempt_load_hf_tokenizer(model_path_or_name):
                tokenizer_found_and_loaded = True
        
        # --- Model Loading ---
        self.load_model(model_path_or_name) # This will use self.hf_tokenizer_chat if loaded

        # --- Fallback to Manual Vocab if HF Tokenizer wasn't loaded by this point ---
        if self.model and not self.hf_tokenizer_chat:
            manual_vocab_path_to_try = None
            if vocab_file_override: # User explicitly provided a vocab.json
                manual_vocab_path_to_try = vocab_file_override
            elif os.path.isdir(potential_tokenizer_base_dir): # Check for vocab.json alongside model
                manual_vocab_path_to_try = os.path.join(potential_tokenizer_base_dir, "vocab.json")
            
            if manual_vocab_path_to_try and os.path.exists(manual_vocab_path_to_try):
                logger.info(f"HF tokenizer not found/loaded. Attempting to load manual vocab from: {manual_vocab_path_to_try}")
                self.load_manual_vocabulary(manual_vocab_path_to_try)
            else:
                logger.warning("No Hugging Face tokenizer found/loaded, and no fallback manual vocab.json could be automatically determined or provided. Chat may not work correctly.")
                self._create_fallback_vocab()
                self.actual_tokenizer_path_loaded = "Fallback minimal vocab for chat"
        
        elif self.model and self.hf_tokenizer_chat:
            if hasattr(self.model.config, "vocab_size") and self.model.config.vocab_size != self.hf_tokenizer_chat.vocab_size:
                logger.error(
                    f"CRITICAL MISMATCH: Model final config vocab_size ({self.model.config.vocab_size}) != "
                    f"loaded HF tokenizer vocab_size ({self.hf_tokenizer_chat.vocab_size}). "
                    "This can happen if model loading failed to properly align with the tokenizer. Inference will likely be broken."
                )
            else:
                 logger.info(f"Model vocab size ({self.model.config.vocab_size}) matches HF tokenizer vocab size ({self.hf_tokenizer_chat.vocab_size}).")

        final_tokenizer_source = "None"
        if self.hf_tokenizer_chat:
            final_tokenizer_source = f"Hugging Face ({self.actual_tokenizer_path_loaded or self.hf_tokenizer_chat.name_or_path})"
        elif self.vocab:
            final_tokenizer_source = f"Manual Vocabulary ({self.actual_tokenizer_path_loaded or 'unknown source'})"
        logger.info(f"Tokenizer for chat: {final_tokenizer_source}")

    def load_model(self, model_weights_path_input: str) -> None:
        try:
            logger.info(f"Loading model. Input path/name: {model_weights_path_input}")
            config_for_model_instantiation: Optional[ApertisConfig] = None
            
            # Determine base directory for config.json
            model_config_base_dir = model_weights_path_input
            if os.path.isfile(model_weights_path_input):
                model_config_base_dir = os.path.dirname(model_weights_path_input)
            
            config_json_path = os.path.join(model_config_base_dir, "config.json")
            if os.path.exists(config_json_path):
                config_for_model_instantiation = ApertisConfig.from_pretrained(config_json_path)
                logger.info(f"Loaded config.json from {config_json_path}. Initial vocab_size from file: {config_for_model_instantiation.vocab_size if config_for_model_instantiation else 'N/A'}")
            else:
                logger.warning(f"config.json not found in {model_config_base_dir}. Will attempt to infer or use defaults if creating new.")

            # Align config_for_model_instantiation with self.hf_tokenizer_chat if it's loaded
            if config_for_model_instantiation and self.hf_tokenizer_chat:
                logger.info("HF tokenizer is loaded. Aligning model config with its properties.")
                config_for_model_instantiation.vocab_size = self.hf_tokenizer_chat.vocab_size
                if self.hf_tokenizer_chat.pad_token_id is not None: config_for_model_instantiation.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                if self.hf_tokenizer_chat.bos_token_id is not None: config_for_model_instantiation.bos_token_id = self.hf_tokenizer_chat.bos_token_id
                if self.hf_tokenizer_chat.eos_token_id is not None: config_for_model_instantiation.eos_token_id = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.unk_token_id is not None: config_for_model_instantiation.unk_token_id = self.hf_tokenizer_chat.unk_token_id
                logger.info(f"Model config aligned with HF tokenizer. New vocab_size: {config_for_model_instantiation.vocab_size}")
            
            # Determine actual model weights file path
            state_dict_path_final = None
            if os.path.isdir(model_weights_path_input):
                bin_path = os.path.join(model_weights_path_input, "pytorch_model.bin")
                pt_path = os.path.join(model_weights_path_input, "model.pt") # Legacy
                if os.path.exists(bin_path): state_dict_path_final = bin_path
                elif os.path.exists(pt_path): state_dict_path_final = pt_path
            elif os.path.isfile(model_weights_path_input) and \
                 (model_weights_path_input.endswith(".pt") or model_weights_path_input.endswith(".bin")):
                state_dict_path_final = model_weights_path_input
            
            if not config_for_model_instantiation and state_dict_path_final:
                logger.info(f"No config.json. Attempting to infer config from state_dict: {state_dict_path_final}")
                config_for_model_instantiation = self._infer_config_from_state_dict(state_dict_path_final)
                if config_for_model_instantiation and self.hf_tokenizer_chat: # Align inferred config if HF tokenizer exists
                    config_for_model_instantiation.vocab_size = self.hf_tokenizer_chat.vocab_size
                    if self.hf_tokenizer_chat.pad_token_id is not None: config_for_model_instantiation.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                    # ... (bos, eos, unk)
                    logger.info(f"Inferred config aligned with HF tokenizer. Vocab_size: {config_for_model_instantiation.vocab_size}")


            if config_for_model_instantiation:
                self.model = ApertisForCausalLM(config_for_model_instantiation)
                if state_dict_path_final and os.path.exists(state_dict_path_final):
                    loaded_state_dict = torch.load(state_dict_path_final, map_location=self.device, weights_only=True)
                    
                    # Check vocab size of embeddings in state_dict
                    emb_weights_key = "model.token_embeddings.weight"
                    lm_head_weights_key = "lm_head.weight"
                    sd_vocab_size = -1
                    if emb_weights_key in loaded_state_dict:
                        sd_vocab_size = loaded_state_dict[emb_weights_key].shape[0]
                    elif lm_head_weights_key in loaded_state_dict and self.model.config.tie_word_embeddings:
                         sd_vocab_size = loaded_state_dict[lm_head_weights_key].shape[0]
                    
                    if sd_vocab_size != -1 and self.model.config.vocab_size != sd_vocab_size:
                        logger.warning(
                            f"Model instance's config vocab_size ({self.model.config.vocab_size}) "
                            f"differs from vocab_size in state_dict's embeddings ({sd_vocab_size}). "
                            "This suggests a mismatch between the loaded config/tokenizer and the model weights. "
                            "Attempting to resize model to match state_dict."
                        )
                        self.model.resize_token_embeddings(sd_vocab_size) # Resize model to match state_dict
                        logger.info(f"Model resized to vocab_size {sd_vocab_size} to match state_dict.")
                    
                    load_result = self.model.load_state_dict(loaded_state_dict, strict=False)
                    logger.info(f"Loaded model weights from {state_dict_path_final}. Final model config vocab_size: {self.model.config.vocab_size}")
                    if load_result.missing_keys: logger.warning(f"Missing keys during load: {load_result.missing_keys}")
                    if load_result.unexpected_keys: logger.warning(f"Unexpected keys during load: {load_result.unexpected_keys}")
                    self.actual_model_path_loaded = state_dict_path_final
                else:
                    logger.info(f"Instantiated model from config (source: {config_json_path or 'inferred/default'}), but no weights file was found/loaded at '{state_dict_path_final}'. Model is initialized randomly.")
                    self.actual_model_path_loaded = config_json_path or "Inferred/Default Config (No Weights)"
            else:
                logger.error(f"Could not load or infer model configuration for {model_weights_path_input}. Model not loaded.")
                self.model = None
                self.actual_model_path_loaded = None

            if self.model:
                self.model.to(self.device)
                self.model.eval()
                self.multimodal = self.model.config.multimodal
            logger.info("Model loading process completed.")

        except Exception as e:
            logger.error(f"Error during model loading: {e}", exc_info=True)
            self.model = None
            self.actual_model_path_loaded = None
            logger.info("Falling back to a dummy model due to loading error.")
            self._create_dummy_model_and_vocab_for_ui_startup()


    def _infer_config_from_state_dict(self, pt_path:str) -> Optional[ApertisConfig]:
        logger.info(f"Attempting to infer model config from state_dict file: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            
            # Infer vocab_size
            vocab_size = 0
            if "model.token_embeddings.weight" in state_dict:
                vocab_size = state_dict["model.token_embeddings.weight"].shape[0]
            elif "lm_head.weight" in state_dict: # If tied or older format
                vocab_size = state_dict["lm_head.weight"].shape[0]
            else: # Fallback if essential layers are missing
                vocab_size = self.hf_tokenizer_chat.vocab_size if self.hf_tokenizer_chat else 32000
            
            # Infer hidden_size
            hidden_size = 0
            if "model.token_embeddings.weight" in state_dict:
                hidden_size = state_dict["model.token_embeddings.weight"].shape[1]
            elif "model.layers.0.attention.q_proj.weight" in state_dict: # More robust
                hidden_size = state_dict["model.layers.0.attention.q_proj.weight"].shape[1]
            elif "lm_head.weight" in state_dict: # Fallback
                hidden_size = state_dict["lm_head.weight"].shape[1]
            else: hidden_size = 768


            layer_count = 0; layer_prefixes = set()
            for k in state_dict.keys():
                if k.startswith("model.layers."):
                    parts = k.split('.'); 
                    if len(parts) > 2 and parts[2].isdigit(): layer_prefixes.add(parts[2])
            layer_count = len(layer_prefixes) if layer_prefixes else 12

            num_attn_heads = hidden_size // 64 if hidden_size > 0 and hidden_size % 64 == 0 else 12
            if hidden_size > 0 and hidden_size % num_attn_heads != 0:
                 for i in range(num_attn_heads, 0, -1):
                    if hidden_size % i == 0: num_attn_heads = i; break
                 if hidden_size % num_attn_heads != 0: num_attn_heads = 1
            
            intermediate_size = hidden_size * 4 if hidden_size > 0 else 3072
            use_expert_system = any("experts" in k for k in state_dict.keys())
            multimodal_inferred = any("multimodal_encoder" in k or "vision_projection" in k for k in state_dict.keys())

            logger.info(f"Inferred params: vocab={vocab_size}, hidden={hidden_size}, layers={layer_count}, heads={num_attn_heads}, intermediate={intermediate_size}, experts={use_expert_system}, multimodal={multimodal_inferred}")

            conf_params = {
                "vocab_size":vocab_size, "hidden_size":hidden_size, "num_hidden_layers":layer_count,
                "num_attention_heads":num_attn_heads, "intermediate_size":intermediate_size,
                "use_expert_system":use_expert_system, "multimodal":multimodal_inferred
            }
            # Use current interface setting for multimodal if not clearly inferable
            if not multimodal_inferred and hasattr(self, 'multimodal'):
                conf_params["multimodal"] = self.multimodal
            
            # Create config with inferred, allowing defaults for others
            conf = ApertisConfig(**conf_params)
            
            # If HF tokenizer is already loaded, align inferred config's special tokens
            if self.hf_tokenizer_chat:
                if self.hf_tokenizer_chat.pad_token_id is not None: conf.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                if self.hf_tokenizer_chat.bos_token_id is not None: conf.bos_token_id = self.hf_tokenizer_chat.bos_token_id
                if self.hf_tokenizer_chat.eos_token_id is not None: conf.eos_token_id = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.unk_token_id is not None: conf.unk_token_id = self.hf_tokenizer_chat.unk_token_id
            return conf
            
        except Exception as e:
            logger.error(f"Error inferring config from {pt_path}: {e}", exc_info=True)
            return None

    def load_manual_vocabulary(self, vocab_file: str) -> None:
        try:
            logger.info(f"Loading manual vocabulary from {vocab_file}")
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            
            loaded_vocab_dict = {}
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    loaded_vocab_dict = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                else:
                    loaded_vocab_dict = vocab_data
            else: raise ValueError(f"Unsupported vocabulary format in {vocab_file}: {type(vocab_data)}")

            if not loaded_vocab_dict:
                logger.warning(f"Manual vocabulary file {vocab_file} is empty or invalid. Using fallback.")
                self._create_fallback_vocab()
                return

            self.vocab = loaded_vocab_dict
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"Manual vocabulary loaded with {len(self.vocab)} distinct tokens.")
            self.actual_tokenizer_path_loaded = vocab_file

            if self.model and hasattr(self.model.config, "vocab_size"):
                model_cfg_vocab_size = self.model.config.vocab_size
                
                max_id_in_loaded_vocab = 0
                if self.vocab: max_id_in_loaded_vocab = max(self.vocab.values() or [-1]) # Handle empty self.vocab
                effective_loaded_vocab_size = max_id_in_loaded_vocab + 1
                
                if model_cfg_vocab_size != effective_loaded_vocab_size :
                    logger.warning(
                        f"Model config vocab_size ({model_cfg_vocab_size}) != "
                        f"effective manual vocab_file size ({effective_loaded_vocab_size}). "
                        "Token mapping may be needed if used directly. "
                        "Consider resizing the model or ensuring vocab consistency."
                    )
                    # For chat, self.tokenize and self.detokenize will handle mapping to model's vocab space.
                    # No need to create self.token_mapping here as chat methods manage this.
                    self.token_mapping = None # Or a more explicit mapping could be built if strict alignment is needed
                else:
                    logger.info(f"Model config vocab_size matches manual vocab effective size: {model_cfg_vocab_size}")
                    self.token_mapping = None 
            else: 
                self.token_mapping = None
        except Exception as e:
            logger.error(f"Error loading manual vocabulary from {vocab_file}: {e}", exc_info=True)
            self._create_fallback_vocab()

    def _create_fallback_vocab(self):
        logger.info("Creating minimal manual vocabulary as fallback.")
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        for i in range(4, 100): self.vocab[f"<tok{i}>"] = i # Add some dummy tokens
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_mapping = None # No mapping needed for this self-consistent fallback
        self.actual_tokenizer_path_loaded = "Fallback minimal vocab (100 tokens)"

    def tokenize(self, text: str) -> List[int]: # For generating prompt IDs for model
        if not self.model:
            logger.error("Model not loaded. Cannot tokenize.")
            return []
        
        if self.hf_tokenizer_chat:
            return self.hf_tokenizer_chat.encode(text, add_special_tokens=False)

        if not self.vocab: # Manual vocab not loaded
            logger.warning("Manual vocab not loaded for tokenize. Using fallback tokenization.")
            # Very basic fallback: map words to unk if not in tiny vocab
            base_unk_id = self.model.config.unk_token_id
            return [self.vocab.get(word, base_unk_id) if self.vocab else base_unk_id for word in text.split()]

        # Manual vocab tokenization: map to model's expected ID range
        model_config = self.model.config
        raw_token_ids_from_vocab_file = [
            self.vocab.get(word, self.vocab.get("<unk>", model_config.unk_token_id)) for word in text.split()
        ]
        
        # Ensure all token IDs are within the model's configured vocab_size
        final_token_ids = []
        for tid in raw_token_ids_from_vocab_file:
            if tid >= model_config.vocab_size:
                final_token_ids.append(model_config.unk_token_id)
            else:
                final_token_ids.append(tid)
        return final_token_ids

    def detokenize(self, token_ids: List[int]) -> str: # For decoding model output IDs
        if not self.model:
            return f"[DetokenizeError: Model not loaded. IDs: {token_ids[:5]}...]"

        if self.hf_tokenizer_chat:
            return self.hf_tokenizer_chat.decode(token_ids, skip_special_tokens=True)

        if not self.reverse_vocab: # Manual reverse_vocab not available
            return f"[DetokenizeError: Manual Reverse Vocab missing. IDs: {token_ids[:5]}...]"

        words = []
        model_config = self.model.config
        default_unk_str = "<unk>" # String representation of unk from original vocab file
        if "<unk>" in self.vocab:
            unk_id_in_vocab_file = self.vocab["<unk>"]
            if unk_id_in_vocab_file in self.reverse_vocab:
                default_unk_str = self.reverse_vocab[unk_id_in_vocab_file]
        
        for model_token_id in token_ids:
            if model_token_id == model_config.pad_token_id or \
               model_token_id == model_config.bos_token_id or \
               model_token_id == model_config.eos_token_id:
                continue # Skip special control tokens not meant for display
            
            word = self.reverse_vocab.get(model_token_id)
            if word is not None:
                words.append(word)
            else: # model_token_id not in reverse_vocab (e.g. if model vocab > file vocab)
                words.append(f"[{default_unk_str.upper()}_ID:{model_token_id}]")
        return " ".join(words)


    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            from torchvision import transforms
            img_size = self.model.config.image_size if self.model and hasattr(self.model.config, 'image_size') else 224
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = Image.open(image_path).convert("RGB")
            return transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            default_size = self.model.config.image_size if self.model and hasattr(self.model.config, 'image_size') else 224
            return torch.zeros(1, 3, default_size, default_size)

    def generate_response(
        self, prompt: str, image_path: Optional[str] = None, max_length: int = 100,
        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
    ) -> str:
        if not self.model: return "Model not loaded."
        if not (self.hf_tokenizer_chat or self.vocab): return "Tokenizer/Vocabulary not loaded."

        try:
            input_ids_list_for_model: List[int]
            if self.hf_tokenizer_chat:
                input_ids_list_for_model = self.hf_tokenizer_chat.encode(prompt, add_special_tokens=True) # HF handles BOS
            else: # Manual vocab
                tokenized_prompt = self.tokenize(prompt) # Already mapped to model's vocab space
                bos_id_for_model = self.model.config.bos_token_id
                if not tokenized_prompt or tokenized_prompt[0] != bos_id_for_model:
                    input_ids_list_for_model = [bos_id_for_model] + tokenized_prompt
                else:
                    input_ids_list_for_model = tokenized_prompt

            input_t = torch.tensor([input_ids_list_for_model], dtype=torch.long).to(self.device)
            attention_mask_t = torch.ones_like(input_t)

            pixel_values_t = None
            model_is_multimodal = self.model.config.multimodal if hasattr(self.model.config, 'multimodal') else False
            if image_path and model_is_multimodal:
                pixel_values_t = self.preprocess_image(image_path).to(self.device)
            elif image_path and not model_is_multimodal:
                 logger.warning("Image provided but model not in multimodal mode.")

            eos_token_id_for_gen = self.model.config.eos_token_id
            pad_token_id_for_gen = self.model.config.pad_token_id
            
            if self.hf_tokenizer_chat: # Prefer HF tokenizer's special IDs if available
                if self.hf_tokenizer_chat.eos_token_id is not None: eos_token_id_for_gen = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.pad_token_id is not None: pad_token_id_for_gen = self.hf_tokenizer_chat.pad_token_id
            
            output_ids = self.model.generate(
                input_ids=input_t, attention_mask=attention_mask_t, pixel_values=pixel_values_t,
                max_new_tokens=max_length, do_sample=temperature > 0.001,
                temperature=temperature if temperature > 0.001 else 1.0,
                top_k=top_k if top_k > 0 else 0, # Pass 0 if disabled
                top_p=top_p if top_p < 1.0 else 1.0, # Pass 1.0 if disabled
                use_cache=True,
                eos_token_id=eos_token_id_for_gen,
                pad_token_id=pad_token_id_for_gen
            )
            
            # output_ids includes the prompt. Slice to get only generated part.
            generated_part_ids = output_ids[0, input_t.shape[1]:].tolist()
            
            if self.hf_tokenizer_chat:
                return self.hf_tokenizer_chat.decode(generated_part_ids, skip_special_tokens=True)
            else:
                return self.detokenize(generated_part_ids)

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def chat(
        self, message: str, image_path: Optional[str] = None, max_length: int = 100,
        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
    ) -> str:

        full_prompt_parts = []
        for entry in self.chat_history:
            full_prompt_parts.append(f"{entry['role'].capitalize()}: {entry['content']}")
        full_prompt_parts.append(f"User: {message}")
        full_prompt_parts.append("Assistant:") # Prompt the model for assistant's turn

        prompt_text_for_model = "\n".join(full_prompt_parts)

        response = self.generate_response(prompt_text_for_model, image_path, max_length, temperature, top_k, top_p)

        self.chat_history.append({"role": "user", "content": message})
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def reset_chat(self) -> None: self.chat_history = []

    def launch_web_interface(self) -> None:
        logger.info("Launching web interface")
        with gr.Blocks(title="Apertis AI Studio", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# Apertis AI Studio")
            with gr.Tabs():
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            chatbot_ui = gr.Chatbot(height=500, label="Apertis Chat")
                            with gr.Row():
                                msg_textbox = gr.Textbox(placeholder="Type message...", show_label=False, scale=7)
                                submit_btn_chat = gr.Button("Send", scale=1)
                            clear_btn_chat = gr.Button("Clear Chat")
                        with gr.Column(scale=1):
                            img_input_chat = gr.Image(type="filepath", label="Upload Image (optional)")
                            with gr.Accordion("Generation Settings", open=False):
                                max_new_tokens_slider = gr.Slider(10, 1024, 100, step=10, label="Max New Tokens")
                                temp_slider_chat = gr.Slider(0.0, 2.0, 0.7, step=0.05, label="Temperature")
                                top_k_slider_chat = gr.Slider(0, 100, 50, step=1, label="Top K (0=disable)")
                                top_p_slider_chat = gr.Slider(0.0, 1.0, 0.9, step=0.05, label="Top P (1.0=disable)")

                with gr.TabItem("Pre-training"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Config")
                            model_size_train_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            attn_type_train_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            multimodal_train_cb = gr.Checkbox(label="Multimodal")
                            expert_sys_train_cb = gr.Checkbox(label="Use Expert System")

                            gr.Markdown("## Data (Pre-training)")
                            train_file_up = gr.File(label="Train Data (JSONL, field: 'text')", file_types=[".jsonl"])
                            val_file_up = gr.File(label="Val Data (JSONL, optional)", file_types=[".jsonl"])
                            vocab_file_up_std_train = gr.File(label="Vocab File (.json, e.g. {'<pad>':0,...} or {'tokens':['<pad>',...]} )", file_types=[".json"])
                            img_dir_train_tb = gr.Textbox(label="Image Dir (for multimodal)", placeholder="/path/to/images", visible=False)
                            multimodal_train_cb.change(lambda x: gr.update(visible=x), inputs=[multimodal_train_cb], outputs=[img_dir_train_tb])

                        with gr.Column(scale=1):
                            gr.Markdown("## Pre-training Params")
                            batch_size_train_sl = gr.Slider(1, 64, 4, step=1, label="Batch Size")
                            lr_train_sl = gr.Slider(1e-6, 1e-3, 5e-5, step=1e-6, label="Learning Rate", info="Logarithmic scale if available in Gradio version")
                            epochs_train_sl = gr.Slider(1, 100, 3, step=1, label="Epochs")
                            eval_epochs_train_sl = gr.Slider(0, 10, 1, step=1, label="Eval Every N Epochs (0=disable)")
                            with gr.Accordion("Checkpoints", open=False):
                                chkpt_steps_sl = gr.Slider(0, 5000, 1000, step=100, label="Global Step Freq (0=disable)")
                                iter_chkpt_steps_sl = gr.Slider(0, 1000, 0, step=10, label="Iteration Freq (0=disable)")
                            with gr.Accordion("GPU", open=False):
                                available_gpus_list = get_available_gpus()
                                gpu_md_text = "### GPUs:\n" + ("\n".join([f"- {g['id']}: {g['name']} ({g['total_memory']:.1f}GB)" for g in available_gpus_list]) or "None detected.")
                                gr.Markdown(gpu_md_text)
                                gpu_choices = [str(g['id']) for g in available_gpus_list]
                                gpu_select_train_cbg = gr.CheckboxGroup(choices=gpu_choices, value=[gpu_choices[0]] if gpu_choices else [], label="Select GPUs", visible=bool(gpu_choices))
                                dist_train_cb = gr.Checkbox(label="Distributed Training", visible=len(gpu_choices)>1)
                                gpu_select_train_cbg.change(lambda g_list: gr.update(visible=len(g_list)>1), inputs=gpu_select_train_cbg, outputs=dist_train_cb)
                                gpu_mem_frac_sl = gr.Slider(0.1,1.0,0.7,step=0.05, label="GPU Mem Fraction")
                            output_dir_train_tb = gr.Textbox("output_pretraining", label="Output Dir")
                            wandb_train_cb = gr.Checkbox(label="Log to W&B")
                            wandb_proj_train_tb = gr.Textbox("apertis-pretraining", label="W&B Project", visible=False)
                            wandb_train_cb.change(lambda x: gr.update(visible=x), inputs=[wandb_train_cb], outputs=[wandb_proj_train_tb])

                            with gr.Row():
                                start_train_btn = gr.Button("Start Pre-training", variant="primary")
                                stop_train_btn = gr.Button("Stop Pre-training")
                            train_status_tb = gr.Textbox(label="Pre-training Status", interactive=False, lines=10, autoscroll=True)

                with gr.TabItem("Fine-tuning"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Base Model for Fine-tuning")
                            ft_base_model_path_tb = gr.Textbox(label="Path to Pre-trained Apertis Model Directory/File", placeholder="e.g., ./models/my_pretrained_model OR ./models/my_model/pytorch_model.bin")

                            gr.Markdown("## Fine-tuning Data")
                            ft_data_file_up = gr.File(label="Fine-tuning Data (JSONL, fields: 'instruction', 'output')", file_types=[".jsonl"])
                            ft_val_file_up = gr.File(label="Validation Data (JSONL, optional)", file_types=[".jsonl"])

                            gr.Markdown("## Tokenizer for Fine-tuning")
                            ft_tokenizer_option_dd = gr.Dropdown(["Use HF Tokenizer (from base model dir or Hub)", "Use Manual Vocab (from base model or new)"], value="Use HF Tokenizer (from base model dir or Hub)", label="Tokenizer Option")
                            ft_hf_tokenizer_name_tb = gr.Textbox(label="HF Tokenizer Name/Path", placeholder="e.g., gpt2, or path to base model's dir", visible=True)
                            ft_manual_vocab_path_tb = gr.File(label="Manual Vocab Path (.json)", file_types=[".json"], visible=False)

                            def toggle_tokenizer_input_ft(choice):
                                if "Use HF Tokenizer" in choice:
                                    return gr.update(visible=True), gr.update(visible=False)
                                return gr.update(visible=False), gr.update(visible=True)
                            ft_tokenizer_option_dd.change(toggle_tokenizer_input_ft, ft_tokenizer_option_dd, [ft_hf_tokenizer_name_tb, ft_manual_vocab_path_tb])

                            ft_prompt_template_tb = gr.Textbox(value="User: {instruction}\nAssistant: {output}", label="Prompt Template")

                        with gr.Column(scale=1):
                            gr.Markdown("## Fine-tuning Params")
                            ft_batch_size_sl = gr.Slider(1, 64, 2, step=1, label="Batch Size")
                            ft_lr_sl = gr.Slider(1e-7, 1e-4, 2e-5, step=1e-7, label="Learning Rate", info="Logarithmic scale if available")
                            ft_epochs_sl = gr.Slider(1, 50, 3, step=1, label="Epochs")
                            ft_eval_epochs_sl = gr.Slider(0, 10, 1, step=1, label="Eval Every N Epochs (0=disable)")

                            with gr.Accordion("GPU (Fine-tuning)", open=False):
                                # Re-fetch available_gpus_list here to ensure it's current if UI reloads
                                available_gpus_list_ft_re = get_available_gpus()
                                gpu_md_text_ft_re = "### GPUs:\n" + ("\n".join([f"- {g['id']}: {g['name']} ({g['total_memory']:.1f}GB)" for g in available_gpus_list_ft_re]) or "None detected.")
                                gr.Markdown(gpu_md_text_ft_re)
                                ft_gpu_choices_re = [str(g['id']) for g in available_gpus_list_ft_re]

                                ft_gpu_select_cbg = gr.CheckboxGroup(choices=ft_gpu_choices_re, value=[ft_gpu_choices_re[0]] if ft_gpu_choices_re else [], label="Select GPUs", visible=bool(ft_gpu_choices_re))
                                ft_dist_train_cb = gr.Checkbox(label="Distributed Training", visible=len(ft_gpu_choices_re)>1)
                                ft_gpu_select_cbg.change(lambda g_list_ft: gr.update(visible=len(g_list_ft)>1), inputs=ft_gpu_select_cbg, outputs=ft_dist_train_cb)
                                ft_gpu_mem_frac_sl = gr.Slider(0.1,1.0,0.7,step=0.05, label="GPU Mem Fraction")

                            ft_output_dir_tb = gr.Textbox("output_finetuning", label="Output Dir")
                            ft_wandb_cb = gr.Checkbox(label="Log to W&B")
                            ft_wandb_proj_tb = gr.Textbox("apertis-finetuning", label="W&B Project", visible=False)
                            ft_wandb_cb.change(lambda x: gr.update(visible=x), inputs=[ft_wandb_cb], outputs=[ft_wandb_proj_tb])

                            with gr.Row():
                                start_ft_btn = gr.Button("Start Fine-tuning", variant="primary")
                                stop_ft_btn = gr.Button("Stop Fine-tuning")
                            ft_status_tb = gr.Textbox(label="Fine-tuning Status", interactive=False, lines=10, autoscroll=True)


                with gr.TabItem("Absolute Zero Reasoner"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Absolute Zero Reasoner Training")
                            gr.Markdown("Train your model using the Absolute Zero Reasoner method.")

                            gr.Markdown("## Model Config (for AZR internal model)")
                            azr_model_size_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            azr_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            azr_multimodal_cb = gr.Checkbox(label="Multimodal (Note: AZR tasks are text-based)")
                            azr_expert_sys_cb = gr.Checkbox(label="Use Expert System")

                            gr.Markdown("## Tokenizer (for AZR internal model)")
                            azr_tokenizer_name_tb = gr.Textbox(
                                value="bert-base-uncased",
                                label="Hugging Face Tokenizer Name",
                                placeholder="e.g., bert-base-uncased, gpt2, meta-llama/Llama-2-7b-hf"
                            )

                            gr.Markdown("## Seed Data (Optional)")
                            azr_seed_tasks_up = gr.File(label="Seed Tasks (JSONL, optional, fields: 'task', 'type')", file_types=[".jsonl"])
                            azr_seed_prob_sl = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Seed Task Probability")

                        with gr.Column(scale=1):
                            gr.Markdown("## AZR Training Parameters")
                            azr_iterations_sl = gr.Slider(10, 500, 100, step=10, label="Number of Iterations")
                            azr_tasks_per_iter_sl = gr.Slider(1, 20, 5, step=1, label="Tasks Per Iteration")

                            with gr.Accordion("Task Generation", open=False):
                                azr_task_types_cbg = gr.CheckboxGroup(["abduction", "deduction", "induction"], value=["abduction", "deduction", "induction"], label="Task Types")
                                azr_task_dist_abduction_sl = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Abduction Weight")
                                azr_task_dist_deduction_sl = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Deduction Weight")
                                azr_task_dist_induction_sl = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Induction Weight")
                                azr_max_attempts_sl = gr.Slider(1, 10, 3, step=1, label="Max Generation Attempts")
                                azr_temperature_sl = gr.Slider(0.1, 1.5, 0.7, step=0.05, label="Generation Temperature")
                                azr_top_p_sl = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-P (1.0 = disable)")

                            with gr.Accordion("Rewards", open=False):
                                gr.Markdown("### Learnability Reward"); azr_learn_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight")
                                gr.Markdown("### Accuracy Reward"); azr_acc_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight"); azr_partial_credit_cb = gr.Checkbox(value=True, label="Allow Partial Credit")
                                gr.Markdown("### Diversity Reward"); azr_div_weight_sl = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Weight"); azr_history_size_sl = gr.Slider(1, 50, 10, step=1, label="History Size")
                                gr.Markdown("### Complexity Reward"); azr_complex_weight_sl = gr.Slider(0.0, 2.0, 0.3, step=0.1, label="Weight"); azr_target_complex_sl = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Target Complexity"); azr_tolerance_sl = gr.Slider(0.0, 0.5, 0.2, step=0.05, label="Tolerance")

                            with gr.Accordion("Python Executor (for some task/solution validation)", open=False):
                                azr_timeout_sl = gr.Slider(1, 30, 5, step=1, label="Execution Timeout (seconds)")
                                azr_max_output_sl = gr.Slider(1000, 50000, 10000, step=1000, label="Max Output Size (chars)")

                            with gr.Accordion("GPU (AZR Training)", open=False):
                                # Re-fetch available_gpus_list here
                                available_gpus_list_azr_re = get_available_gpus()
                                gpu_md_text_azr_re = "### GPUs:\n" + ("\n".join([f"- {g['id']}: {g['name']} ({g['total_memory']:.1f}GB)" for g in available_gpus_list_azr_re]) or "None detected.")
                                gr.Markdown(gpu_md_text_azr_re)
                                azr_gpu_choices_re = [str(g['id']) for g in available_gpus_list_azr_re]

                                azr_gpu_select_dd = gr.Dropdown(choices=azr_gpu_choices_re + ["cpu"], value=azr_gpu_choices_re[0] if azr_gpu_choices_re else "cpu", label="Select Device (GPU ID or CPU)")
                                # AZR currently simpler, often single GPU or CPU. No explicit DDP for AZR in this UI.
                                # azr_gpu_mem_frac_sl = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="GPU Memory Fraction (Informational)")

                            azr_output_dir_tb = gr.Textbox("output_azr", label="Output Directory")
                            azr_wandb_cb = gr.Checkbox(label="Log to W&B")
                            azr_wandb_proj_tb = gr.Textbox("apertis-azr", label="W&B Project", visible=False)
                            azr_wandb_cb.change(lambda x: gr.update(visible=x), inputs=[azr_wandb_cb], outputs=[azr_wandb_proj_tb])

                            with gr.Row():
                                azr_start_btn = gr.Button("Start AZR Training", variant="primary")
                                azr_stop_btn = gr.Button("Stop AZR Training")
                            azr_status_tb = gr.Textbox(label="AZR Training Status", interactive=False, lines=10, autoscroll=True)

                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model for Chat")
                            model_path_load_tb = gr.Textbox(self.model_path_arg or "", label="Model Path/Name (HF Hub ID or Local Dir/File)", placeholder="e.g. ./models/my_model OR ./models/my_model/pytorch_model.bin")
                            vocab_path_load_tb = gr.Textbox(self.vocab_file_fallback_arg or "", label="Manual Vocab Path (.json, Fallback ONLY if no HF tokenizer found with model)")
                            load_model_btn_ui = gr.Button("Load Model")
                            model_info_load_tb = gr.Textbox(label="Loaded Model Info", interactive=False, lines=10, autoscroll=True)
                        with gr.Column(scale=1):
                            gr.Markdown("## Create New Model (for Pre-training)")
                            new_model_size_dd = gr.Dropdown(["small","base","large"], value="base", label="Model Size")
                            new_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            new_multimodal_cb = gr.Checkbox(label="Multimodal")
                            new_expert_cb = gr.Checkbox(label="Use Expert System")
                            new_vocab_size_num = gr.Number(32000, label="Vocab Size (for manual vocab)", precision=0)
                            new_model_out_tb = gr.Textbox("models/new_apertis_model", label="Save Path for New Model Files")
                            create_model_btn_ui = gr.Button("Create & Save New Model Files")
                            create_model_status_tb = gr.Textbox(label="Creation Status", interactive=False, lines=3)

            def ui_chat_handler(msg, img, max_new, temp, tk, tp, hist):
                if not self.model:
                    gr.Warning("No model loaded. Please load a model from the 'Models' tab.")
                    return hist, "" # Return current history and empty input box
                if not msg.strip() and not img: 
                    gr.Info("Please provide a message or an image.")
                    return hist, ""
                
                response_from_model = self.chat(msg, img, max_new, temp, tk, tp)
                
                # Gradio chatbot expects a list of (user_msg, assistant_msg) tuples
                # We manage self.chat_history internally. We need to convert it for display.
                gradio_display_history = []
                temp_internal_history_for_display = self.chat_history.copy() # Use the updated history
                
                user_turn_text_for_display = None
                for entry_idx in range(len(temp_internal_history_for_display)):
                    if temp_internal_history_for_display[entry_idx]["role"] == "user":
                        user_turn_text_for_display = temp_internal_history_for_display[entry_idx]["content"]
                    elif temp_internal_history_for_display[entry_idx]["role"] == "assistant" and user_turn_text_for_display is not None:
                        gradio_display_history.append( (user_turn_text_for_display, temp_internal_history_for_display[entry_idx]["content"]) )
                        user_turn_text_for_display = None # Reset for next pair

                return gradio_display_history, "" # Clear input textbox


            submit_btn_chat.click(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            msg_textbox.submit(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])

            def ui_clear_chat_handler():
                self.reset_chat()
                return [], "", None # Clear chatbot, message box, image input
            clear_btn_chat.click(ui_clear_chat_handler, outputs=[chatbot_ui, msg_textbox, img_input_chat])


            def ui_load_model_handler(m_path_ui, v_path_override_ui):
                if not m_path_ui:
                    return "Please provide a model path or name."
                self.load_model_and_tokenizer_from_path(m_path_ui, v_path_override_ui if v_path_override_ui else None)
                
                info_parts = [f"Attempted to load model using input path/name: {m_path_ui}"]
                info_parts.append(f"  Actual model resource path: {self.actual_model_path_loaded or 'N/A'}")
                
                if self.model and hasattr(self.model.config, 'to_dict'):
                    cfg_dict = self.model.config.to_dict()
                    # Display a few key config items for brevity in UI
                    brief_cfg = {
                        "model_type": cfg_dict.get("model_type"), "vocab_size": cfg_dict.get("vocab_size"),
                        "hidden_size": cfg_dict.get("hidden_size"), "num_hidden_layers": cfg_dict.get("num_hidden_layers"),
                        "multimodal": cfg_dict.get("multimodal"), "attention_type": cfg_dict.get("attention_type"),
                        "pad_token_id":cfg_dict.get("pad_token_id"), "bos_token_id":cfg_dict.get("bos_token_id"),
                        "eos_token_id":cfg_dict.get("eos_token_id"), "unk_token_id":cfg_dict.get("unk_token_id"),
                    }
                    info_parts.append("\nModel Config (Brief):\n" + json.dumps(brief_cfg, indent=2))
                else:
                    info_parts.append("\nFailed to load model or model has no config.")

                if self.hf_tokenizer_chat:
                    info_parts.append(f"\nUsing Hugging Face Tokenizer: {self.actual_tokenizer_path_loaded or self.hf_tokenizer_chat.name_or_path}")
                    info_parts.append(f"  Vocab Size: {self.hf_tokenizer_chat.vocab_size}, PAD: {self.hf_tokenizer_chat.pad_token_id}, EOS: {self.hf_tokenizer_chat.eos_token_id}")
                elif self.vocab:
                    info_parts.append(f"\nUsing Manual Vocab: {len(self.vocab)} tokens from {self.actual_tokenizer_path_loaded or 'unknown source'}.")
                    if self.token_mapping: info_parts.append("  Token mapping active for manual vocab.")
                else:
                    info_parts.append("\nNo tokenizer/vocabulary loaded for chat.")
                return "\n".join(info_parts)

            load_model_btn_ui.click(ui_load_model_handler, [model_path_load_tb, vocab_path_load_tb], [model_info_load_tb])

            def ui_create_model_handler(size_ui, attn_type_ui, multi_ui, expert_ui, v_size_ui, out_path_ui):
                try:
                    if not out_path_ui: return "Output path for new model files is required."
                    v_size_int = int(v_size_ui) if v_size_ui is not None else 32000
                    
                    # Create model instance
                    new_model_instance = create_apertis_model(
                        model_size=size_ui, vocab_size_override=v_size_int,
                        multimodal=multi_ui, use_expert_system=expert_ui,
                        attention_type_override=attn_type_ui
                    )
                    # Save model weights and config.json
                    new_model_instance.save_pretrained(out_path_ui)
                    
                    # Create a dummy vocab.json consistent with v_size_int
                    dummy_vocab_content = {f"<token_{i}>": i for i in range(v_size_int)}
                    # Add default special tokens if they fit within vocab_size
                    default_specials = {
                        "<pad>": new_model_instance.config.pad_token_id, 
                        "<bos>": new_model_instance.config.bos_token_id, 
                        "<eos>": new_model_instance.config.eos_token_id, 
                        "<unk>": new_model_instance.config.unk_token_id
                    }
                    for tok_str, tok_id in default_specials.items():
                        if tok_id < v_size_int:
                            dummy_vocab_content[tok_str] = tok_id
                        else: # Should not happen if ApertisConfig defaults are low
                            logger.warning(f"Default special token {tok_str} ID {tok_id} too large for vocab_size {v_size_int}. Not adding to dummy vocab.")
                    
                    with open(os.path.join(out_path_ui, "vocab.json"),"w",encoding="utf-8") as f:
                        json.dump(dummy_vocab_content, f, indent=2)
                        
                    return f"Model files (pytorch_model.bin, config.json) and a basic vocab.json (with {v_size_int} tokens) " \
                           f"created at '{out_path_ui}'.\n" \
                           "IMPORTANT: For actual use, replace the dummy vocab.json with a real one, or ensure your " \
                           "training/inference pipeline loads an appropriate Hugging Face tokenizer."
                except Exception as e:
                    logger.error(f"Error creating model: {e}", exc_info=True)
                    return f"Error: {str(e)}"
            create_model_btn_ui.click(ui_create_model_handler,
                                     [new_model_size_dd, new_attn_type_dd, new_multimodal_cb, new_expert_cb, new_vocab_size_num, new_model_out_tb],
                                     [create_model_status_tb])

            def ui_start_training_handler(
                m_s, attn_t, m_m, exp_s, tr_f_obj, v_f_obj, voc_f_std_obj, img_d, b_s, learn_r, eps, eval_ep,
                c_steps, iter_c_steps, g_sel, d_train, g_mem_f, out_d, use_wb, wb_p):
                
                current_status = ""
                if not tr_f_obj: current_status += "Training data file is required.\n"
                if not voc_f_std_obj: current_status += "Vocabulary file (.json) is required for pre-training.\n"
                if m_m and not img_d: current_status += "Image directory is required for multimodal pre-training.\n"
                if not out_d: current_status += "Output directory is required.\n"
                if current_status: return current_status.strip()

                if self.standard_training_thread and self.standard_training_thread.is_alive():
                    return "Pre-training is already in progress."

                self.standard_training_stop_event.clear()
                tmp_dir = tempfile.mkdtemp()
                try:
                    train_p = os.path.join(tmp_dir, "train.jsonl"); shutil.copy(tr_f_obj.name, train_p)
                    vocab_p_std = os.path.join(tmp_dir, "vocab.json"); shutil.copy(voc_f_std_obj.name, vocab_p_std)
                    val_p = None
                    if v_f_obj: val_p = os.path.join(tmp_dir, "val.jsonl"); shutil.copy(v_f_obj.name, val_p)

                    sel_gpus = [int(gid) for gid in g_sel] if g_sel else None
                    dist_training_eff = d_train if sel_gpus and len(sel_gpus) > 1 else False

                    cfg = {
                        "data_config": {"train_data_path":train_p, "tokenizer_path":vocab_p_std, "val_data_path":val_p,
                                        "max_length":512, "multimodal":m_m, "image_dir":img_d if m_m else None,
                                        "image_size": 224},
                        "model_config": { # These are for NEW models for pre-training
                            "model_size":m_s, "attention_type": attn_t, "multimodal":m_m, "use_expert_system":exp_s
                        },
                        "training_config": {
                            "task_type": "pretrain",
                            "output_dir":out_d, "batch_size":b_s, "learning_rate":learn_r, "num_epochs":eps,
                            "warmup_steps":0, "gradient_accumulation_steps":4, "max_grad_norm": 1.0,
                            "eval_every_n_epochs":eval_ep, "use_wandb":use_wb, "wandb_project":wb_p if use_wb else None,
                            "wandb_run_name": None, "fp16":True, "device":None,
                            "gpu_memory_fraction":g_mem_f,
                            "use_gradient_checkpointing":True, "dynamic_batch_sizing":True,
                            "checkpoint_steps":c_steps, "iteration_checkpoint_steps":iter_c_steps,
                            "gpu_ids":sel_gpus, "distributed_training":dist_training_eff, "local_rank": -1
                        }
                    }
                    
                    cfg_path = os.path.join(tmp_dir, "run_cfg_pretrain.json");
                    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                    
                    os.makedirs(out_d, exist_ok=True)
                    final_cfg_path_in_output = os.path.join(out_d, "run_cfg_pretrain_used.json")
                    shutil.copy(cfg_path, final_cfg_path_in_output)


                    def _thread_train_job(c_p_arg, t_d_arg, stop_event_arg, status_box_arg):
                        try:
                            from src.training.pipeline import train_from_config
                            status_box_arg.value = f"Pre-training started. Output: {out_d}. Config: {final_cfg_path_in_output}\nFollow logs in console/wandb."
                            train_from_config(c_p_arg, stop_event_arg)
                            if stop_event_arg.is_set():
                                status_box_arg.value += "\nPre-training stopped by user."
                            else:
                                status_box_arg.value += "\nPre-training completed."
                        except Exception as e_thread:
                             logger.error(f"Error in Pre-training thread: {e_thread}", exc_info=True)
                             status_box_arg.value += f"\nError in Pre-training thread: {e_thread}"
                        finally:
                            shutil.rmtree(t_d_arg)
                            self.standard_training_thread = None
                    
                    train_status_tb.value = "Initializing pre-training..." # Initial feedback
                    self.standard_training_thread = threading.Thread(target=_thread_train_job, args=(cfg_path, tmp_dir, self.standard_training_stop_event, train_status_tb), daemon=True)
                    self.standard_training_thread.start()
                    return f"Pre-training initiated. Output will be in '{out_d}'. Monitor console/W&B for progress. Copied config to '{final_cfg_path_in_output}'."
                except Exception as e_start:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    logger.error(f"Failed to start pre-training: {e_start}", exc_info=True)
                    return f"Failed to start pre-training: {e_start}"


            def ui_stop_training_handler():
                if self.standard_training_thread and self.standard_training_thread.is_alive():
                    self.standard_training_stop_event.set()
                    return "Stop request sent to Pre-training. Please wait for current step/epoch to finish."
                return "No Pre-training in progress or thread already completed."

            start_train_btn.click(ui_start_training_handler,
                                 [model_size_train_dd, attn_type_train_dd, multimodal_train_cb, expert_sys_train_cb,
                                  train_file_up, val_file_up, vocab_file_up_std_train, img_dir_train_tb,
                                  batch_size_train_sl, lr_train_sl, epochs_train_sl, eval_epochs_train_sl,
                                  chkpt_steps_sl, iter_chkpt_steps_sl, gpu_select_train_cbg, dist_train_cb,
                                  gpu_mem_frac_sl, output_dir_train_tb, wandb_train_cb, wandb_proj_train_tb],
                                 [train_status_tb])
            stop_train_btn.click(ui_stop_training_handler, outputs=[train_status_tb])

            def ui_start_finetuning_handler(
                base_model_path_ui, ft_data_f_obj, ft_val_f_obj,
                tokenizer_opt_ui, hf_tokenizer_name_ui, ft_manual_vocab_path_obj, prompt_template_ui,
                batch_s_ui, learn_r_ui, eps_ui, eval_ep_ui,
                g_sel_ui, d_train_ui, g_mem_f_ui, out_d_ui, use_wb_ui, wb_p_ui
            ):
                current_status = ""
                if not base_model_path_ui: current_status += "Base model path required for fine-tuning.\n"
                if not ft_data_f_obj: current_status += "Fine-tuning data file required.\n"
                if "Use HF Tokenizer" in tokenizer_opt_ui and not hf_tokenizer_name_ui:
                    current_status += "HF Tokenizer Name/Path required if selected for fine-tuning.\n"
                if "Use Manual Vocab" in tokenizer_opt_ui and not ft_manual_vocab_path_obj:
                    current_status += "Manual Vocab Path required if selected for fine-tuning.\n"
                if not out_d_ui: current_status += "Output directory is required.\n"
                if current_status: return current_status.strip()


                if self.finetune_training_thread and self.finetune_training_thread.is_alive():
                    return "Fine-tuning is already in progress."

                self.finetune_training_stop_event.clear()
                tmp_dir = tempfile.mkdtemp()
                try:
                    ft_train_p = os.path.join(tmp_dir, "ft_train.jsonl"); shutil.copy(ft_data_f_obj.name, ft_train_p)
                    ft_val_p = None
                    if ft_val_f_obj: ft_val_p = os.path.join(tmp_dir, "ft_val.jsonl"); shutil.copy(ft_val_f_obj.name, ft_val_p)

                    tokenizer_path_for_config_ft = ""
                    use_hf_for_ft_config = False
                    if "Use HF Tokenizer" in tokenizer_opt_ui:
                        tokenizer_path_for_config_ft = hf_tokenizer_name_ui
                        use_hf_for_ft_config = True
                    elif ft_manual_vocab_path_obj :
                        tokenizer_path_for_config_ft = os.path.join(tmp_dir, "ft_manual_vocab.json")
                        shutil.copy(ft_manual_vocab_path_obj.name, tokenizer_path_for_config_ft)
                    else: # Should be caught by initial checks, but defensive
                        return "Tokenizer configuration error for fine-tuning."

                    sel_gpus = [int(gid) for gid in g_sel_ui] if g_sel_ui else None
                    dist_training_eff = d_train_ui if sel_gpus and len(sel_gpus) > 1 else False

                    cfg = {
                        "data_config": {
                            "train_data_path": ft_train_p, "val_data_path": ft_val_p,
                            "tokenizer_path": tokenizer_path_for_config_ft,
                            "use_hf_tokenizer_for_finetune": use_hf_for_ft_config,
                            "prompt_template": prompt_template_ui, "max_length": 512
                        },
                        "model_config": {}, # For FT, base model structure is loaded, but minor overrides can be here
                        "training_config": {
                            "task_type": "finetune",
                            "pretrained_model_path_for_finetune": base_model_path_ui,
                            "output_dir": out_d_ui, "batch_size": batch_s_ui, "learning_rate": learn_r_ui,
                            "num_epochs": eps_ui, "eval_every_n_epochs": eval_ep_ui,
                            "warmup_steps": 0, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0,
                            "use_wandb": use_wb_ui, "wandb_project": wb_p_ui if use_wb_ui else None,
                            "fp16": True, "device": None, "gpu_memory_fraction": g_mem_f_ui,
                            "use_gradient_checkpointing": True, "dynamic_batch_sizing": True,
                            "gpu_ids": sel_gpus, "distributed_training": dist_training_eff, "local_rank": -1
                        }
                    }

                    cfg_path = os.path.join(tmp_dir, "run_cfg_finetune.json")
                    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                    os.makedirs(out_d_ui, exist_ok=True)
                    final_cfg_path_in_output_ft = os.path.join(out_d_ui, "run_cfg_finetune_used.json")
                    shutil.copy(cfg_path, final_cfg_path_in_output_ft)


                    def _thread_finetune_job(c_p_arg, t_d_arg, stop_event_arg, status_box_arg):
                        try:
                            from src.training.pipeline import train_from_config
                            status_box_arg.value = f"Fine-tuning started. Output: {out_d_ui}. Config: {final_cfg_path_in_output_ft}\nFollow logs."
                            train_from_config(c_p_arg, stop_event_arg)
                            if stop_event_arg.is_set(): status_box_arg.value += "\nFine-tuning stopped by user."
                            else: status_box_arg.value += "\nFine-tuning completed."
                        except Exception as e_thread_ft:
                             logger.error(f"Error in Fine-tuning thread: {e_thread_ft}", exc_info=True)
                             status_box_arg.value += f"\nError in Fine-tuning thread: {e_thread_ft}"
                        finally:
                            shutil.rmtree(t_d_arg)
                            self.finetune_training_thread = None
                    
                    ft_status_tb.value = "Initializing fine-tuning..."
                    self.finetune_training_thread = threading.Thread(target=_thread_finetune_job, args=(cfg_path, tmp_dir, self.finetune_training_stop_event, ft_status_tb), daemon=True)
                    self.finetune_training_thread.start()
                    return f"Fine-tuning initiated. Output will be in '{out_d_ui}'. Monitor console/W&B. Copied config to '{final_cfg_path_in_output_ft}'."
                except Exception as e_start_ft:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    logger.error(f"Failed to start fine-tuning: {e_start_ft}", exc_info=True)
                    return f"Failed to start fine-tuning: {e_start_ft}"


            def ui_stop_finetuning_handler():
                if self.finetune_training_thread and self.finetune_training_thread.is_alive():
                    self.finetune_training_stop_event.set()
                    return "Stop request sent to Fine-tuning. Please wait for current step/epoch to finish."
                return "No Fine-tuning in progress or thread already completed."

            start_ft_btn.click(ui_start_finetuning_handler,
                [ft_base_model_path_tb, ft_data_file_up, ft_val_file_up,
                 ft_tokenizer_option_dd, ft_hf_tokenizer_name_tb, ft_manual_vocab_path_tb, ft_prompt_template_tb,
                 ft_batch_size_sl, ft_lr_sl, ft_epochs_sl, ft_eval_epochs_sl,
                 ft_gpu_select_cbg, ft_dist_train_cb, ft_gpu_mem_frac_sl,
                 ft_output_dir_tb, ft_wandb_cb, ft_wandb_proj_tb],
                [ft_status_tb])
            stop_ft_btn.click(ui_stop_finetuning_handler, outputs=[ft_status_tb])

            def ui_start_azr_training_handler(
                m_s, attn_t, m_m, exp_s, tokenizer_name_hf, seed_f_obj, seed_prob,
                iterations, tasks_per_iter, task_types_list, abduction_w, deduction_w, induction_w,
                max_attempts, temperature, top_p_gen, learn_weight, acc_weight, partial_credit,
                div_weight, history_size, complex_weight, target_complex, tolerance,
                timeout_exec, max_output_exec, azr_device_select_ui, out_d, use_wb, wb_p
            ):
                current_status = ""
                if not tokenizer_name_hf.strip(): current_status += "Hugging Face Tokenizer Name is required for AZR.\n"
                if not task_types_list: current_status += "At least one AZR Task Type must be selected.\n"
                if not out_d: current_status += "Output directory is required.\n"
                if current_status: return current_status.strip()
                
                if self.azr_training_thread and self.azr_training_thread.is_alive():
                    return "AZR training is already in progress."

                self.azr_training_stop_event.clear()
                tmp_dir = tempfile.mkdtemp()
                try:
                    seed_p = None
                    if seed_f_obj:
                        seed_p = os.path.join(tmp_dir, "seed_tasks.jsonl")
                        shutil.copy(seed_f_obj.name, seed_p)

                    task_dist_map = {"abduction": abduction_w, "deduction": deduction_w, "induction": induction_w}
                    final_task_types = [tt for tt in task_types_list if tt in task_dist_map]
                    final_task_dist_weights = [task_dist_map[tt] for tt in final_task_types]

                    if not final_task_types: # Should be caught by initial check
                        return "No valid task types selected or weights configured for AZR."
                    
                    sum_weights = sum(final_task_dist_weights)
                    if sum_weights > 0:
                        final_task_dist_weights = [w / sum_weights for w in final_task_dist_weights]
                    else: # All selected task types have 0 weight, distribute equally
                        equal_w = 1.0 / len(final_task_types)
                        final_task_dist_weights = [equal_w] * len(final_task_types)
                    
                    azr_model_cfg_base = create_apertis_model(model_size=m_s, attention_type_override=attn_t, multimodal=m_m, use_expert_system=exp_s).config.to_dict()

                    cfg = {
                        "data": {"tokenizer_name": tokenizer_name_hf.strip()},
                        "model": azr_model_cfg_base, # AZR pipeline will set vocab_size from HF tokenizer
                        "training": {"method": "azr", "output_dir": out_d, "device": azr_device_select_ui},
                        "azr": {
                            "num_iterations": iterations, "tasks_per_iteration": tasks_per_iter,
                            "checkpoint_interval": 10, "checkpoint_dir": os.path.join(out_d, "azr_checkpoints"),
                            "force_accept_tasks": True, "force_accept_solutions": True,
                            "force_accept_threshold": 10, "min_valid_tasks_before_validation": 20,
                            "log_level": "INFO", "log_file": os.path.join(out_d, "azr_training.log"),
                            "python_executor": {"timeout": timeout_exec, "max_output_size": max_output_exec},
                            "task_generator": {
                                "task_types": final_task_types, "task_distribution": final_task_dist_weights,
                                "max_attempts": max_attempts, "seed_tasks_path": seed_p,
                                "seed_task_probability": seed_prob,
                                "base_prompt": "Generate a challenging reasoning problem.",
                                "max_new_tokens": 100, "temperature": temperature, "top_p": top_p_gen
                            },
                            "task_validator": {"min_length": 10, "max_length": 2500, "min_complexity": 0.1, "max_complexity": 1.0, "min_clarity": 0.3},
                            "solution_generator": {"max_attempts": max_attempts, "base_prompt": "Solve the following problem step by step:", "include_task_type_hint": True, "max_new_tokens": 1024, "temperature": temperature, "top_p": top_p_gen},
                            "solution_validator": {"min_coherence": 0.3, "min_relevance": 0.3, "min_structure": 0.2},
                            "learnability_reward": {"weight": learn_weight},
                            "accuracy_reward": {"weight": acc_weight, "partial_credit": partial_credit},
                            "diversity_reward": {"weight": div_weight, "history_size": history_size},
                            "complexity_reward": {"weight": complex_weight, "target_complexity": target_complex, "tolerance": tolerance},
                            "use_wandb": use_wb, "wandb_project": wb_p if use_wb else "apertis-azr",
                        }
                    }

                    os.makedirs(out_d, exist_ok=True)
                    cfg_path = os.path.join(tmp_dir, "azr_config.json")
                    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                    final_cfg_path_in_output_azr = os.path.join(out_d, "azr_config_used.json")
                    shutil.copy(cfg_path, final_cfg_path_in_output_azr)

                    def _thread_azr_train(c_p_arg, t_d_arg, stop_event_arg, status_box_arg):
                        try:
                            from src.training import train_from_config as azr_entry_train
                            status_box_arg.value = f"AZR Training started. Output: {out_d}. Config: {final_cfg_path_in_output_azr}\nFollow logs."
                            azr_entry_train(c_p_arg, stop_event_arg) # Uses the training.__init__.py dispatcher
                            if stop_event_arg.is_set(): status_box_arg.value += "\nAZR Training stopped by user."
                            else: status_box_arg.value += "\nAZR Training completed."
                        except Exception as e_thread_azr:
                            logger.error(f"Error in AZR training thread: {e_thread_azr}", exc_info=True)
                            status_box_arg.value += f"\nError in AZR training thread: {e_thread_azr}"
                        finally:
                            shutil.rmtree(t_d_arg)
                            self.azr_training_thread = None
                    
                    azr_status_tb.value = "Initializing AZR training..."
                    self.azr_training_thread = threading.Thread(target=_thread_azr_train, args=(cfg_path, tmp_dir, self.azr_training_stop_event, azr_status_tb), daemon=True)
                    self.azr_training_thread.start()
                    return f"AZR Training initiated. Output will be in '{out_d}'. Monitor console/W&B. Copied config to '{final_cfg_path_in_output_azr}'."
                except Exception as e_start_azr:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    logger.error(f"Failed to start AZR training: {e_start_azr}", exc_info=True)
                    return f"Failed to start AZR training: {e_start_azr}"


            def ui_stop_azr_training_handler():
                if self.azr_training_thread and self.azr_training_thread.is_alive():
                    self.azr_training_stop_event.set()
                    return "Stop request sent to AZR Training. Please wait for current iteration to finish."
                return "No AZR Training in progress or thread already completed."

            azr_start_btn.click(
                ui_start_azr_training_handler,
                [
                    azr_model_size_dd, azr_attn_type_dd, azr_multimodal_cb, azr_expert_sys_cb,
                    azr_tokenizer_name_tb, azr_seed_tasks_up, azr_seed_prob_sl,
                    azr_iterations_sl, azr_tasks_per_iter_sl, azr_task_types_cbg,
                    azr_task_dist_abduction_sl, azr_task_dist_deduction_sl, azr_task_dist_induction_sl,
                    azr_max_attempts_sl, azr_temperature_sl, azr_top_p_sl,
                    azr_learn_weight_sl, azr_acc_weight_sl, azr_partial_credit_cb,
                    azr_div_weight_sl, azr_history_size_sl, azr_complex_weight_sl,
                    azr_target_complex_sl, azr_tolerance_sl, azr_timeout_sl, azr_max_output_sl,
                    azr_gpu_select_dd, azr_output_dir_tb,
                    azr_wandb_cb, azr_wandb_proj_tb
                ],
                [azr_status_tb]
            )
            azr_stop_btn.click(ui_stop_azr_training_handler, outputs=[azr_status_tb])

        try:
            interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share, max_threads=80, prevent_thread_lock=True)
        except OSError as e:
            if ("Can't assign requested address" in str(e) or "Address already in use" in str(e)) and self.port == 7860 :
                logger.warning(f"Port {self.port} might be in use. Trying 7861...")
                self.port = 7861
                interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share, max_threads=80, prevent_thread_lock=True)
            else:
                logger.error(f"Failed to launch Gradio interface: {e}", exc_info=True)
                print(f"Failed to launch Gradio interface on port {self.port}. Please check if the port is available or try another one.")
        except Exception as e_launch:
            logger.error(f"An unexpected error occurred during Gradio launch: {e_launch}", exc_info=True)