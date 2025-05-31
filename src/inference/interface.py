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
from src.training.pipeline import YoloStyleTrainingPipeline, ApertisPretrainDataset, get_available_gpus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApertisInterface:
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_file: Optional[str] = None, # This is now primarily a fallback for non-HF models
        multimodal: bool = False,
        device: Optional[str] = None,
        web: bool = False,
        port: int = 7860,
        share: bool = False,
    ):
        self.model_path_arg = model_path # Store the original argument
        self.vocab_file_fallback_arg = vocab_file # Store the original argument
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
        self.vocab: Optional[Dict[str, int]] = None # For manual vocab
        self.reverse_vocab: Optional[Dict[int, str]] = None # For manual vocab
        self.token_mapping: Optional[Dict[int, int]] = None # For manual vocab mapping

        self.hf_tokenizer_chat = None
        self.actual_model_path_loaded = None # Path from where model was actually loaded
        self.actual_vocab_path_loaded = None # Path from where vocab/tokenizer was actually loaded


        if self.model_path_arg is not None:
            self.load_model_and_tokenizer_from_path(self.model_path_arg, vocab_file_override=self.vocab_file_fallback_arg)

        self.chat_history: List[Dict[str,str]] = []

        self.standard_training_stop_event = threading.Event()
        self.azr_training_stop_event = threading.Event()
        self.finetune_training_stop_event = threading.Event()
        self.standard_training_thread: Optional[threading.Thread] = None
        self.azr_training_thread: Optional[threading.Thread] = None
        self.finetune_training_thread: Optional[threading.Thread] = None


        if web:
            self.launch_web_interface()

    def _attempt_load_hf_tokenizer(self, path_to_check: str) -> bool:
        """Helper to attempt loading HF tokenizer from a given path."""
        try:
            from transformers import AutoTokenizer
            # Check for presence of key files that indicate a full HF tokenizer save
            # tokenizer.json is the most indicative for modern tokenizers.
            # vocab.json + merges.txt for older BPEs like GPT-2.
            # tokenizer_config.json is usually always there.
            is_tokenizer_dir = os.path.isdir(path_to_check) and (
                os.path.exists(os.path.join(path_to_check, "tokenizer.json")) or
                (os.path.exists(os.path.join(path_to_check, "vocab.json")) and os.path.exists(os.path.join(path_to_check, "merges.txt"))) or
                os.path.exists(os.path.join(path_to_check, "tokenizer_config.json"))
            )

            if is_tokenizer_dir:
                self.hf_tokenizer_chat = AutoTokenizer.from_pretrained(path_to_check)
                logger.info(f"Successfully loaded Hugging Face tokenizer from directory: {path_to_check}")
                self.actual_vocab_path_loaded = path_to_check
                return True
            # If path_to_check is not a directory or doesn't seem to contain tokenizer files,
            # try loading it as a Hub ID or a direct path to a specific tokenizer file (less common for AutoTokenizer)
            # This part is more for "model_path_or_name" being a Hub ID.
            if not os.path.isdir(path_to_check): # Try as Hub ID if not a directory
                 self.hf_tokenizer_chat = AutoTokenizer.from_pretrained(path_to_check)
                 logger.info(f"Successfully loaded Hugging Face tokenizer from Hub ID/path: {path_to_check}")
                 self.actual_vocab_path_loaded = path_to_check # Could be a Hub ID string
                 return True

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
        self.vocab_file_fallback_arg = vocab_file_override # Store this for potential fallback

        tokenizer_found_and_loaded = False

        # Determine the base directory to check for tokenizer files
        # This is important if model_path_or_name is a path to a .bin or .pt file
        potential_tokenizer_dir = model_path_or_name
        if os.path.isfile(model_path_or_name):
            potential_tokenizer_dir = os.path.dirname(model_path_or_name)

        # --- Tokenizer Loading Logic ---
        # Priority 1: Try to load HF tokenizer from the determined directory
        if os.path.isdir(potential_tokenizer_dir):
            logger.info(f"Checking for HF tokenizer in directory: {potential_tokenizer_dir}")
            if self._attempt_load_hf_tokenizer(potential_tokenizer_dir):
                tokenizer_found_and_loaded = True

        # Priority 2: If model_path_or_name itself was not a directory (could be Hub ID) or above failed, try it directly
        if not tokenizer_found_and_loaded and model_path_or_name != potential_tokenizer_dir:
            logger.info(f"Checking if '{model_path_or_name}' is an HF Hub ID or direct tokenizer path.")
            if self._attempt_load_hf_tokenizer(model_path_or_name):
                tokenizer_found_and_loaded = True
        
        # --- Model Loading ---
        self.load_model(model_path_or_name) # model_weights_path is the original model_path_or_name

        # --- Fallback to Manual Vocab if HF Tokenizer wasn't loaded ---
        if self.model and not self.hf_tokenizer_chat:
            if self.vocab_file_fallback_arg:
                logger.info(f"HF tokenizer not successfully loaded. Falling back to manual vocab provided: {self.vocab_file_fallback_arg}")
                self.load_vocabulary(self.vocab_file_fallback_arg)
                self.actual_vocab_path_loaded = self.vocab_file_fallback_arg
            else:
                logger.warning("No Hugging Face tokenizer found/loaded and no fallback manual vocab_file provided. Chat may not work correctly.")
                self._create_fallback_vocab()
                self.actual_vocab_path_loaded = "Fallback minimal vocab"
        
        elif self.model and self.hf_tokenizer_chat:
            # If HF tokenizer was loaded, ensure model config matches its vocab size
            # (This should have been handled in load_model by aligning config_to_use)
            if hasattr(self.model.config, "vocab_size") and self.model.config.vocab_size != self.hf_tokenizer_chat.vocab_size:
                logger.error(f"CRITICAL MISMATCH: Model config vocab_size ({self.model.config.vocab_size}) != "
                               f"loaded HF tokenizer vocab_size ({self.hf_tokenizer_chat.vocab_size}). "
                               "This should not happen if model loading correctly aligned config. Inference will be broken.")
            else:
                logger.info(f"Model vocab size ({self.model.config.vocab_size}) matches HF tokenizer vocab size ({self.hf_tokenizer_chat.vocab_size}).")


        final_tokenizer_source = "None"
        if self.hf_tokenizer_chat:
            final_tokenizer_source = f"Hugging Face ({self.actual_vocab_path_loaded or self.hf_tokenizer_chat.name_or_path})"
        elif self.vocab:
            final_tokenizer_source = f"Manual Vocabulary ({self.actual_vocab_path_loaded or 'unknown source'})"
        logger.info(f"Tokenizer for chat: {final_tokenizer_source}")


    def load_model(self, model_weights_path_input: str) -> None:
        try:
            logger.info(f"Loading model, input path: {model_weights_path_input}")
            config_to_use = None
            state_dict_path_final = None
            # Determine the directory that should contain config.json (and potentially tokenizer files)
            model_config_base_dir = model_weights_path_input
            if os.path.isfile(model_weights_path_input):
                model_config_base_dir = os.path.dirname(model_weights_path_input)

            if os.path.isdir(model_config_base_dir):
                config_json_path = os.path.join(model_config_base_dir, "config.json")
                if os.path.exists(config_json_path):
                    config_to_use = ApertisConfig.from_pretrained(config_json_path)
                    logger.info(f"Loaded config.json from {config_json_path}. Initial vocab_size: {config_to_use.vocab_size if config_to_use else 'N/A'}")
                else:
                    logger.warning(f"config.json not found in {model_config_base_dir}.")
            else:
                 logger.warning(f"Determined model config base directory '{model_config_base_dir}' is not a valid directory.")


            # Determine the actual model weights file (.bin or .pt)
            if os.path.isdir(model_weights_path_input): # If input path is a directory
                pt_model_file = os.path.join(model_weights_path_input, "pytorch_model.bin")
                if not os.path.exists(pt_model_file):
                    pt_model_file = os.path.join(model_weights_path_input, "model.pt")
                if os.path.exists(pt_model_file):
                    state_dict_path_final = pt_model_file
                else: # If it's a dir but no model file, config might be the only thing to load
                    logger.warning(f"Directory {model_weights_path_input} provided, but no model weights file (pytorch_model.bin or model.pt) found directly within it.")
                    # config_to_use might have been loaded if config.json was there.
            elif os.path.isfile(model_weights_path_input) and \
                 (model_weights_path_input.endswith(".pt") or model_weights_path_input.endswith(".bin")):
                state_dict_path_final = model_weights_path_input
                # config_to_use should have been loaded from its sibling dir already
            else: # Input is not a dir, not a .bin/.pt file -> could be HF Hub ID for model *weights*
                  # This case is less common for local setup, usually HF Hub ID implies full model repo.
                  # For now, assume local paths or directories. If it's an HF ID, AutoModel...from_pretrained would be used.
                  # Our current structure assumes weights are local.
                  pass


            # If an HF tokenizer was already loaded by the calling function, align config_to_use with it
            if config_to_use and self.hf_tokenizer_chat and hasattr(config_to_use, "vocab_size"):
                if config_to_use.vocab_size != self.hf_tokenizer_chat.vocab_size:
                    logger.info(f"Aligning model config vocab_size from {config_to_use.vocab_size} "
                                f"to successfully loaded HF tokenizer vocab_size {self.hf_tokenizer_chat.vocab_size}.")
                    config_to_use.vocab_size = self.hf_tokenizer_chat.vocab_size
                # Also update special token IDs in config from HF tokenizer if they exist
                if self.hf_tokenizer_chat.pad_token_id is not None: config_to_use.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                if self.hf_tokenizer_chat.bos_token_id is not None: config_to_use.bos_token_id = self.hf_tokenizer_chat.bos_token_id
                if self.hf_tokenizer_chat.eos_token_id is not None: config_to_use.eos_token_id = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.unk_token_id is not None: config_to_use.unk_token_id = self.hf_tokenizer_chat.unk_token_id


            if not config_to_use and state_dict_path_final: # No config.json, but have weights file
                logger.info(f"No config.json found, attempting to infer config from weights file: {state_dict_path_final}")
                config_to_use = self._infer_config_from_state_dict(state_dict_path_final)
                if config_to_use and self.hf_tokenizer_chat: # If inferred and HF tokenizer exists, align vocab
                    logger.info(f"Aligning inferred config vocab_size with HF tokenizer vocab_size {self.hf_tokenizer_chat.vocab_size}.")
                    config_to_use.vocab_size = self.hf_tokenizer_chat.vocab_size
                    if self.hf_tokenizer_chat.pad_token_id is not None: config_to_use.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                    # etc. for bos, eos, unk


            if config_to_use and state_dict_path_final:
                self.model = ApertisForCausalLM(config_to_use)
                loaded_state_dict = torch.load(state_dict_path_final, map_location=self.device, weights_only=True)
                
                # Final check: ensure model instance's vocab size matches the state_dict about to be loaded
                expected_vocab_size_in_sd = -1
                if "model.token_embeddings.weight" in loaded_state_dict:
                    expected_vocab_size_in_sd = loaded_state_dict["model.token_embeddings.weight"].shape[0]
                elif "lm_head.weight" in loaded_state_dict: # For models without separate embedding matrix (older/simpler)
                     expected_vocab_size_in_sd = loaded_state_dict["lm_head.weight"].shape[0]

                if expected_vocab_size_in_sd != -1 and self.model.config.vocab_size != expected_vocab_size_in_sd:
                    logger.warning(
                        f"Model instance's final config vocab size ({self.model.config.vocab_size}) "
                        f"STILL differs from vocab size in state_dict ({expected_vocab_size_in_sd}). "
                        f"This is unexpected if HF tokenizer alignment worked. "
                        f"Forcing model instance to match state_dict vocab size: {expected_vocab_size_in_sd}"
                    )
                    self.model.config.vocab_size = expected_vocab_size_in_sd
                    self.model = ApertisForCausalLM(self.model.config) # Re-instantiate with corrected vocab

                self.model.load_state_dict(loaded_state_dict, strict=True)
                logger.info(f"Loaded model weights from {state_dict_path_final}. Final model config vocab_size: {self.model.config.vocab_size}")

            elif config_to_use and not state_dict_path_final:
                 self.model = ApertisForCausalLM(config_to_use)
                 logger.info(f"Created model from config (found at {model_config_base_dir}), but no weights file was loaded/found.")
            else:
                logger.error(f"Could not load model: No valid config or weights file found for {model_weights_path_input}")
                self.model = None

            if self.model:
                self.model.to(self.device)
                self.model.eval()
                self.multimodal = self.model.config.multimodal # Sync multimodal state
                self.actual_model_path_loaded = state_dict_path_final or model_config_base_dir
            logger.info("Model loading process completed.")

        except Exception as e:
            logger.error(f"Error during model loading: {e}", exc_info=True)
            self.model = None
            # Fallback model creation
            logger.info("Creating default 'small' model as fallback due to loading error.")
            fallback_vocab_size = self.hf_tokenizer_chat.vocab_size if self.hf_tokenizer_chat else 32000
            fb_config = ApertisConfig(model_type="apertis", vocab_size=fallback_vocab_size, hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048, multimodal=self.multimodal)
            if self.hf_tokenizer_chat: # Align fallback config with HF tokenizer if present
                if self.hf_tokenizer_chat.pad_token_id is not None: fb_config.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                if self.hf_tokenizer_chat.bos_token_id is not None: fb_config.bos_token_id = self.hf_tokenizer_chat.bos_token_id
                if self.hf_tokenizer_chat.eos_token_id is not None: fb_config.eos_token_id = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.unk_token_id is not None: fb_config.unk_token_id = self.hf_tokenizer_chat.unk_token_id

            self.model = ApertisForCausalLM(fb_config)
            if self.model:
                self.model.to(self.device)
                self.model.eval()
            self.actual_model_path_loaded = "Fallback default model"


    def _infer_config_from_state_dict(self, pt_path:str) -> Optional[ApertisConfig]:
        logger.info(f"Attempting to infer model config from state_dict file: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            use_expert_system = any("experts" in k for k in state_dict.keys())

            vocab_s_tensor = state_dict.get("model.token_embeddings.weight", state_dict.get("lm_head.weight"))
            vocab_size = vocab_s_tensor.size(0) if vocab_s_tensor is not None else (self.hf_tokenizer_chat.vocab_size if self.hf_tokenizer_chat else 32000)

            hidden_s_tensor = state_dict.get("model.token_embeddings.weight", state_dict.get("lm_head.weight")) # Use lm_head as secondary source for hidden_size
            hidden_size = hidden_s_tensor.size(1) if hidden_s_tensor is not None else 768


            layer_count = 0
            layer_prefixes = set()
            for k in state_dict.keys():
                if k.startswith("model.layers."):
                    parts = k.split('.')
                    if len(parts) > 2 and parts[2].isdigit():
                        layer_prefixes.add(parts[2])
            layer_count = len(layer_prefixes)
            if layer_count == 0: layer_count = 12

            num_attn_heads = hidden_size // 64 if hidden_size > 0 and hidden_size % 64 == 0 else 12
            if hidden_size > 0 and hidden_size % num_attn_heads != 0 : # Ensure divisibility
                 for i in range(num_attn_heads, 0, -1):
                    if hidden_size % i == 0:
                        num_attn_heads = i
                        break
                 if hidden_size % num_attn_heads != 0: num_attn_heads = 1


            intermediate_size = hidden_size * 4 if hidden_size > 0 else 3072


            logger.info(f"Inferred params: vocab={vocab_size}, hidden={hidden_size}, layers={layer_count}, heads={num_attn_heads}, intermediate={intermediate_size}")

            # Create a base config and then update with inferred values
            # This ensures all other default params from ApertisConfig are set
            conf = ApertisConfig(
                vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=layer_count,
                num_attention_heads=num_attn_heads, intermediate_size=intermediate_size,
                multimodal=self.multimodal, # Use current interface setting
                use_expert_system=use_expert_system
            )
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


    def load_vocabulary(self, vocab_file: str) -> None: # This is for manual vocab
        try:
            logger.info(f"Loading manual vocabulary from {vocab_file}")
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list): # Handle simple list of tokens
                    self.vocab = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                else: # Assume it's a token:id dict
                    self.vocab = vocab_data
            else: raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"Manual vocabulary loaded with {len(self.vocab)} tokens")
            self.actual_vocab_path_loaded = vocab_file

            if self.model and hasattr(self.model.config, "vocab_size"):
                model_cfg_vocab_size = self.model.config.vocab_size
                
                # Determine effective size of loaded manual vocab
                max_id_in_loaded_vocab = 0
                if self.vocab: max_id_in_loaded_vocab = max(self.vocab.values() or [0])
                effective_loaded_vocab_size = max_id_in_loaded_vocab + 1
                
                if model_cfg_vocab_size != effective_loaded_vocab_size :
                    logger.warning(f"Model config vocab_size ({model_cfg_vocab_size}) != "
                                   f"effective manual vocab_file size ({effective_loaded_vocab_size}). Will create token mapping.")
                    self.create_token_mapping(model_cfg_vocab_size, len(self.vocab)) # Pass entry count for map log
                else:
                    self.token_mapping = None # Sizes match, no mapping needed
            elif not self.model: # Model not loaded yet, can't compare sizes
                self.token_mapping = None
        except Exception as e:
            logger.error(f"Error loading manual vocabulary from {vocab_file}: {e}", exc_info=True)
            self._create_fallback_vocab()

    def _create_fallback_vocab(self):
        logger.info("Creating minimal manual vocabulary as fallback.")
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_mapping = None
        self.actual_vocab_path_loaded = "Fallback minimal vocab"


    def create_token_mapping(self, model_cfg_vocab_size: int, loaded_vocab_file_entry_count: int) -> None:
        if self.vocab is None: logger.error("Cannot create token mapping: manual vocabulary not loaded."); return
        self.token_mapping = {}
        special_tokens_map = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        model_unk_id = min(special_tokens_map["<unk>"], model_cfg_vocab_size -1) if model_cfg_vocab_size > 0 else 0

        for token_str, ideal_id in special_tokens_map.items():
            if token_str in self.vocab:
                vocab_file_id = self.vocab[token_str]
                if ideal_id < model_cfg_vocab_size:
                    self.token_mapping[vocab_file_id] = ideal_id
                elif vocab_file_id < model_cfg_vocab_size:
                    self.token_mapping[vocab_file_id] = vocab_file_id
                else:
                    self.token_mapping[vocab_file_id] = model_unk_id

        for vocab_file_token_id in self.vocab.values():
            if vocab_file_token_id not in self.token_mapping:
                if vocab_file_token_id < model_cfg_vocab_size:
                    self.token_mapping[vocab_file_token_id] = vocab_file_token_id
                else:
                    self.token_mapping[vocab_file_token_id] = model_unk_id
        logger.info(f"Token mapping created for manual vocab. Model expects {model_cfg_vocab_size} tokens. Vocab file has {loaded_vocab_file_entry_count} entries.")


    def tokenize(self, text: str) -> List[int]:
        if self.hf_tokenizer_chat:
            # For generation, usually add_special_tokens=False for the raw prompt part,
            # BOS/EOS handled by generate or chat history formatting.
            return self.hf_tokenizer_chat.encode(text, add_special_tokens=False)

        if not self.vocab or not self.model :
            logger.warning("Manual vocab or model not loaded for tokenize. Using placeholder.")
            return [3] * len(text.split())

        loaded_vocab_unk_id = self.vocab.get("<unk>", 3)
        if not isinstance(loaded_vocab_unk_id, int): loaded_vocab_unk_id = 3
        raw_token_ids = [self.vocab.get(word, loaded_vocab_unk_id) for word in text.split()]

        if self.token_mapping:
            model_unk_for_mapping = self.token_mapping.get(loaded_vocab_unk_id, self.model.config.unk_token_id)
            return [self.token_mapping.get(tid, model_unk_for_mapping) for tid in raw_token_ids]
        else:
            model_vocab_size = self.model.config.vocab_size
            model_unk_direct = self.model.config.unk_token_id
            return [tid if tid < model_vocab_size else model_unk_direct for tid in raw_token_ids]

    def detokenize(self, token_ids: List[int]) -> str:
        if self.hf_tokenizer_chat:
            return self.hf_tokenizer_chat.decode(token_ids, skip_special_tokens=True)

        if not self.reverse_vocab or not self.model:
            return f"[DetokenizeError: Manual Vocab/Model missing. IDs: {token_ids[:5]}...]"

        words = []
        reverse_id_map_for_detok = {v: k for k, v in self.token_mapping.items()} if self.token_mapping else None
        
        original_vocab_unk_id = self.vocab.get("<unk>", 3)
        if not isinstance(original_vocab_unk_id, int): original_vocab_unk_id = 3
        loaded_vocab_unk_string = self.reverse_vocab.get(original_vocab_unk_id, "<unk>")


        for model_token_id in token_ids:
            id_to_lookup_in_reverse_vocab = model_token_id
            if reverse_id_map_for_detok:
                id_to_lookup_in_reverse_vocab = reverse_id_map_for_detok.get(model_token_id, model_token_id)

            word = self.reverse_vocab.get(id_to_lookup_in_reverse_vocab)

            if word is not None and word not in ["<pad>", "<bos>", "<eos>"]:
                if word == loaded_vocab_unk_string and id_to_lookup_in_reverse_vocab != original_vocab_unk_id :
                     words.append(f"[{loaded_vocab_unk_string.upper()}_MAPPED_FROM_ID:{model_token_id}]")
                else:
                    words.append(word)
            elif word is None: # Token ID from model has no corresponding word
                 words.append(f"[UNKNOWN_ID:{model_token_id}]")
        return " ".join(words)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            from torchvision import transforms
            img_size = self.model.config.image_size if self.model else 224
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = Image.open(image_path).convert("RGB")
            return transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            default_size = self.model.config.image_size if self.model else 224
            return torch.zeros(1, 3, default_size, default_size)

    def generate_response(
        self, prompt: str, image_path: Optional[str] = None, max_length: int = 100,
        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
    ) -> str:
        if not self.model: return "Model not loaded."
        if not self.hf_tokenizer_chat and not self.vocab: return "Tokenizer/Vocabulary not loaded."

        try:
            input_ids_list_for_model: List[int]
            if self.hf_tokenizer_chat:
                # For generate, HF tokenizers often handle BOS/EOS via their specific config
                # or expect them to be part of the prompt template logic.
                # `encode` typically handles adding special tokens if `add_special_tokens=True`.
                # Let's ensure the prompt passed to `generate` has special tokens if model expects them.
                # Most CausalLMs expect BOS at the start of a sequence.
                input_ids_list_for_model = self.hf_tokenizer_chat.encode(prompt, add_special_tokens=True)
            else: # Manual vocab
                tokenized_prompt = self.tokenize(prompt) # This already uses mapped IDs if mapping exists
                bos_id_for_model = self.model.config.bos_token_id # Get model's expected BOS ID
                
                # If token_mapping is active, the BOS ID from vocab needs to be mapped
                if self.token_mapping and self.vocab and "<bos>" in self.vocab:
                    bos_id_for_model = self.token_mapping.get(self.vocab["<bos>"], self.model.config.bos_token_id)

                if not tokenized_prompt or tokenized_prompt[0] != bos_id_for_model:
                    input_ids_list_for_model = [bos_id_for_model] + tokenized_prompt
                else:
                    input_ids_list_for_model = tokenized_prompt

            input_t = torch.tensor([input_ids_list_for_model], dtype=torch.long).to(self.device)
            attention_mask_t = torch.ones_like(input_t)

            pixel_values_t = None
            if image_path and self.multimodal and self.model.config.multimodal:
                pixel_values_t = self.preprocess_image(image_path).to(self.device)
            elif image_path and (not self.multimodal or not self.model.config.multimodal) :
                 logger.warning("Image provided but model/interface not in multimodal mode.")

            eos_token_id_for_gen = self.model.config.eos_token_id
            if self.hf_tokenizer_chat and self.hf_tokenizer_chat.eos_token_id is not None:
                eos_token_id_for_gen = self.hf_tokenizer_chat.eos_token_id
            elif not self.hf_tokenizer_chat and self.vocab and self.vocab.get("<eos>") is not None:
                 eos_from_vocab = self.vocab.get("<eos>")
                 eos_token_id_for_gen = self.token_mapping.get(eos_from_vocab, eos_from_vocab) if self.token_mapping else eos_from_vocab

            pad_token_id_for_gen = self.model.config.pad_token_id
            if self.hf_tokenizer_chat and self.hf_tokenizer_chat.pad_token_id is not None:
                pad_token_id_for_gen = self.hf_tokenizer_chat.pad_token_id
            elif not self.hf_tokenizer_chat and self.vocab and self.vocab.get("<pad>") is not None:
                pad_from_vocab = self.vocab.get("<pad>")
                pad_token_id_for_gen = self.token_mapping.get(pad_from_vocab, pad_from_vocab) if self.token_mapping else pad_from_vocab


            output_ids = self.model.generate(
                input_ids=input_t, attention_mask=attention_mask_t, pixel_values=pixel_values_t,
                max_new_tokens=max_length, do_sample=temperature > 0.001,
                temperature=temperature if temperature > 0.001 else 1.0,
                top_k=top_k if top_k > 0 else None,
                top_p=top_p if top_p < 1.0 else None,
                use_cache=True,
                eos_token_id=eos_token_id_for_gen,
                pad_token_id=pad_token_id_for_gen
            )
            response_ids = output_ids[0, input_t.shape[1]:].tolist()

            if self.hf_tokenizer_chat:
                return self.hf_tokenizer_chat.decode(response_ids, skip_special_tokens=True)
            else:
                return self.detokenize(response_ids)

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
        full_prompt_parts.append("Assistant:")

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
                                top_p_slider_chat = gr.Slider(0.0, 1.0, 0.9, step=0.05, label="Top P (0=disable)")

                with gr.TabItem("Pre-training"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Config")
                            model_size_train_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            attn_type_train_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            multimodal_train_cb = gr.Checkbox(label="Multimodal")
                            expert_sys_train_cb = gr.Checkbox(label="Use Expert System")

                            gr.Markdown("## Data")
                            train_file_up = gr.File(label="Train Data (JSONL, field: 'text')", file_types=[".jsonl"])
                            val_file_up = gr.File(label="Val Data (JSONL, optional)", file_types=[".jsonl"])
                            vocab_file_up_std_train = gr.File(label="Vocab File (JSON)", file_types=[".json"])
                            img_dir_train_tb = gr.Textbox(label="Image Dir (for multimodal)", placeholder="/path/to/images", visible=False)
                            multimodal_train_cb.change(lambda x: gr.update(visible=x), inputs=[multimodal_train_cb], outputs=[img_dir_train_tb])

                        with gr.Column(scale=1):
                            gr.Markdown("## Pre-training Params")
                            batch_size_train_sl = gr.Slider(1, 64, 4, step=1, label="Batch Size")
                            lr_train_sl = gr.Slider(1e-6, 1e-3, 5e-5, step=1e-6, label="Learning Rate")
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
                                gpu_select_train_cbg.change(lambda g: gr.update(visible=len(g)>1), inputs=gpu_select_train_cbg, outputs=dist_train_cb)
                                gpu_mem_frac_sl = gr.Slider(0.1,1.0,0.7,step=0.05, label="GPU Mem Fraction")
                            output_dir_train_tb = gr.Textbox("output_pretraining", label="Output Dir")
                            wandb_train_cb = gr.Checkbox(label="Log to W&B")
                            wandb_proj_train_tb = gr.Textbox("apertis-pretraining", label="W&B Project", visible=False)
                            wandb_train_cb.change(lambda x: gr.update(visible=x), inputs=wandb_train_cb, outputs=wandb_proj_train_tb)

                            with gr.Row():
                                start_train_btn = gr.Button("Start Pre-training", variant="primary")
                                stop_train_btn = gr.Button("Stop Pre-training")
                            train_status_tb = gr.Textbox(label="Pre-training Status", interactive=False, lines=10)

                with gr.TabItem("Fine-tuning"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Base Model for Fine-tuning")
                            ft_base_model_path_tb = gr.Textbox(label="Path to Pre-trained Apertis Model Directory/File")

                            gr.Markdown("## Fine-tuning Data")
                            ft_data_file_up = gr.File(label="Fine-tuning Data (JSONL, fields: 'instruction', 'output')", file_types=[".jsonl"])
                            ft_val_file_up = gr.File(label="Validation Data (JSONL, optional)", file_types=[".jsonl"])

                            gr.Markdown("## Tokenizer for Fine-tuning")
                            ft_tokenizer_option_dd = gr.Dropdown(["Use HF Tokenizer (Recommended from model dir or Hub)", "Use Manual Vocab (from base model or provided)"], value="Use HF Tokenizer (Recommended from model dir or Hub)", label="Tokenizer Option")
                            ft_hf_tokenizer_name_tb = gr.Textbox(label="HF Tokenizer Name/Path (if selected)", placeholder="e.g., gpt2, or path to fine-tuned model's dir", visible=True)
                            ft_manual_vocab_path_tb = gr.File(label="Manual Vocab Path (.json, if selected)", file_types=[".json"], visible=False)

                            def toggle_tokenizer_input_ft(choice):
                                if "Use HF Tokenizer" in choice:
                                    return gr.update(visible=True), gr.update(visible=False)
                                return gr.update(visible=False), gr.update(visible=True)
                            ft_tokenizer_option_dd.change(toggle_tokenizer_input_ft, ft_tokenizer_option_dd, [ft_hf_tokenizer_name_tb, ft_manual_vocab_path_tb])

                            ft_prompt_template_tb = gr.Textbox(value="User: {instruction}\nAssistant: {output}", label="Prompt Template")

                        with gr.Column(scale=1):
                            gr.Markdown("## Fine-tuning Params")
                            ft_batch_size_sl = gr.Slider(1, 64, 2, step=1, label="Batch Size")
                            ft_lr_sl = gr.Slider(1e-7, 1e-4, 2e-5, step=1e-7, label="Learning Rate")
                            ft_epochs_sl = gr.Slider(1, 50, 3, step=1, label="Epochs")
                            ft_eval_epochs_sl = gr.Slider(0, 10, 1, step=1, label="Eval Every N Epochs (0=disable)")

                            with gr.Accordion("GPU (Fine-tuning)", open=False):
                                available_gpus_list_ft = get_available_gpus()
                                gpu_md_text_ft = "### GPUs:\n" + ("\n".join([f"- {g['id']}: {g['name']} ({g['total_memory']:.1f}GB)" for g in available_gpus_list_ft]) or "None detected.")
                                gr.Markdown(gpu_md_text_ft)
                                ft_gpu_choices = [str(g['id']) for g in available_gpus_list_ft]
                                ft_gpu_select_cbg = gr.CheckboxGroup(choices=ft_gpu_choices, value=[ft_gpu_choices[0]] if ft_gpu_choices else [], label="Select GPUs", visible=bool(ft_gpu_choices))
                                ft_dist_train_cb = gr.Checkbox(label="Distributed Training", visible=len(ft_gpu_choices)>1)
                                ft_gpu_select_cbg.change(lambda g: gr.update(visible=len(g)>1), inputs=ft_gpu_select_cbg, outputs=ft_dist_train_cb)
                                ft_gpu_mem_frac_sl = gr.Slider(0.1,1.0,0.7,step=0.05, label="GPU Mem Fraction")

                            ft_output_dir_tb = gr.Textbox("output_finetuning", label="Output Dir")
                            ft_wandb_cb = gr.Checkbox(label="Log to W&B")
                            ft_wandb_proj_tb = gr.Textbox("apertis-finetuning", label="W&B Project", visible=False)
                            ft_wandb_cb.change(lambda x: gr.update(visible=x), inputs=[ft_wandb_cb], outputs=[ft_wandb_proj_tb])

                            with gr.Row():
                                start_ft_btn = gr.Button("Start Fine-tuning", variant="primary")
                                stop_ft_btn = gr.Button("Stop Fine-tuning")
                            ft_status_tb = gr.Textbox(label="Fine-tuning Status", interactive=False, lines=10)


                with gr.TabItem("Absolute Zero Reasoner"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Absolute Zero Reasoner Training")
                            gr.Markdown("Train your model using the Absolute Zero Reasoner method...")

                            gr.Markdown("## Model Config")
                            azr_model_size_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            azr_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            azr_multimodal_cb = gr.Checkbox(label="Multimodal")
                            azr_expert_sys_cb = gr.Checkbox(label="Use Expert System")

                            gr.Markdown("## Tokenizer")
                            azr_tokenizer_name_tb = gr.Textbox(
                                value="bert-base-uncased",
                                label="Hugging Face Tokenizer Name",
                                placeholder="e.g., bert-base-uncased, gpt2, meta-llama/Llama-2-7b-hf"
                            )

                            gr.Markdown("## Seed Data (Optional)")
                            azr_seed_tasks_up = gr.File(label="Seed Tasks (JSONL, optional)", file_types=[".jsonl"])
                            azr_seed_prob_sl = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Seed Task Probability")

                        with gr.Column(scale=1):
                            gr.Markdown("## AZR Training Parameters")
                            azr_iterations_sl = gr.Slider(10, 500, 100, step=10, label="Number of Iterations")
                            azr_tasks_per_iter_sl = gr.Slider(1, 20, 5, step=1, label="Tasks Per Iteration")

                            with gr.Accordion("Task Generation", open=False):
                                azr_task_types = gr.CheckboxGroup(["abduction", "deduction", "induction"], value=["abduction", "deduction", "induction"], label="Task Types")
                                azr_task_dist_abduction = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Abduction Weight")
                                azr_task_dist_deduction = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Deduction Weight")
                                azr_task_dist_induction = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Induction Weight")
                                azr_max_attempts_sl = gr.Slider(1, 10, 3, step=1, label="Max Generation Attempts")
                                azr_temperature_sl = gr.Slider(0.1, 1.5, 0.7, step=0.05, label="Generation Temperature")
                                azr_top_p_sl = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-P")

                            with gr.Accordion("Rewards", open=False):
                                gr.Markdown("### Learnability Reward"); azr_learn_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight")
                                gr.Markdown("### Accuracy Reward"); azr_acc_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight"); azr_partial_credit_cb = gr.Checkbox(value=True, label="Allow Partial Credit")
                                gr.Markdown("### Diversity Reward"); azr_div_weight_sl = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Weight"); azr_history_size_sl = gr.Slider(1, 50, 10, step=1, label="History Size")
                                gr.Markdown("### Complexity Reward"); azr_complex_weight_sl = gr.Slider(0.0, 2.0, 0.3, step=0.1, label="Weight"); azr_target_complex_sl = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Target Complexity"); azr_tolerance_sl = gr.Slider(0.0, 0.5, 0.2, step=0.05, label="Tolerance")

                            with gr.Accordion("Python Executor", open=False):
                                azr_timeout_sl = gr.Slider(1, 30, 5, step=1, label="Execution Timeout (seconds)")
                                azr_max_output_sl = gr.Slider(1000, 50000, 10000, step=1000, label="Max Output Size")

                            with gr.Accordion("GPU", open=False):
                                available_gpus_list_azr = get_available_gpus()
                                gpu_md_text_azr = "### GPUs:\n" + ("\n".join([f"- {g['id']}: {g['name']} ({g['total_memory']:.1f}GB)" for g in available_gpus_list_azr]) or "None detected.")
                                gr.Markdown(gpu_md_text_azr)
                                azr_gpu_choices = [str(g['id']) for g in available_gpus_list_azr]
                                azr_gpu_select_cbg = gr.CheckboxGroup(choices=azr_gpu_choices, value=[azr_gpu_choices[0]] if azr_gpu_choices else [], label="Select GPUs", visible=bool(azr_gpu_choices))
                                azr_gpu_mem_frac_sl = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="GPU Memory Fraction")

                            azr_output_dir_tb = gr.Textbox("output_azr", label="Output Directory")
                            azr_wandb_cb = gr.Checkbox(label="Log to W&B")
                            azr_wandb_proj_tb = gr.Textbox("apertis-azr", label="W&B Project", visible=False)
                            azr_wandb_cb.change(lambda x: gr.update(visible=x), inputs=[azr_wandb_cb], outputs=[azr_wandb_proj_tb])

                            with gr.Row():
                                azr_start_btn = gr.Button("Start AZR Training", variant="primary")
                                azr_stop_btn = gr.Button("Stop AZR Training")
                            azr_status_tb = gr.Textbox(label="AZR Training Status", interactive=False, lines=10)

                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model")
                            model_path_load_tb = gr.Textbox(self.model_path_arg or "", label="Model Path/Name (HF Hub ID or Local Dir/File)")
                            vocab_path_load_tb = gr.Textbox(self.vocab_file_fallback_arg or "", label="Manual Vocab Path (.json, Fallback only if no HF tokenizer with model)")
                            load_model_btn_ui = gr.Button("Load Model")
                            model_info_load_tb = gr.Textbox(label="Loaded Model Info", interactive=False, lines=10)
                        with gr.Column(scale=1):
                            gr.Markdown("## Create New Model")
                            new_model_size_dd = gr.Dropdown(["small","base","large"], value="base", label="Model Size")
                            new_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            new_multimodal_cb = gr.Checkbox(label="Multimodal")
                            new_expert_cb = gr.Checkbox(label="Use Expert System")
                            new_vocab_size_num = gr.Number(32000, label="Vocab Size (used if not HF tokenizer based)", precision=0)
                            new_model_out_tb = gr.Textbox("models/new_model", label="Save Path")
                            create_model_btn_ui = gr.Button("Create Model")
                            create_model_status_tb = gr.Textbox(label="Creation Status", interactive=False, lines=3)

            def ui_chat_handler(msg, img, max_new, temp, tk, tp, hist):
                if not msg.strip() and not img: return hist, ""
                response_from_model = self.chat(msg, img, max_new, temp, tk, tp)
                gradio_history = []
                temp_internal_history = self.chat_history.copy()
                user_turn_text = None
                for entry_idx in range(0, len(temp_internal_history)):
                    if temp_internal_history[entry_idx]["role"] == "user":
                        user_turn_text = temp_internal_history[entry_idx]["content"]
                    elif temp_internal_history[entry_idx]["role"] == "assistant" and user_turn_text is not None:
                        gradio_history.append((user_turn_text, temp_internal_history[entry_idx]["content"]))
                        user_turn_text = None
                return gradio_history, ""


            submit_btn_chat.click(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            msg_textbox.submit(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])

            def ui_clear_chat_handler():
                self.reset_chat()
                return [], "", None
            clear_btn_chat.click(ui_clear_chat_handler, outputs=[chatbot_ui, msg_textbox, img_input_chat])


            def ui_load_model_handler(m_path, v_path_override_ui):
                self.load_model_and_tokenizer_from_path(m_path, v_path_override_ui if v_path_override_ui else None)
                info_parts = [f"Attempted to load model from: {self.actual_model_path_loaded or 'N/A'}"]
                if self.model and hasattr(self.model.config, 'to_dict'):
                    info_parts.append("Model Config:\n" + json.dumps(self.model.config.to_dict(), indent=2))
                else:
                    info_parts.append("Failed to load model or model has no config.")

                if self.hf_tokenizer_chat:
                    info_parts.append(f"\nUsing Hugging Face Tokenizer: {self.actual_vocab_path_loaded or self.hf_tokenizer_chat.name_or_path}, Vocab Size: {self.hf_tokenizer_chat.vocab_size}")
                elif self.vocab:
                    info_parts.append(f"\nUsing Manual Vocab: {len(self.vocab)} tokens from {self.actual_vocab_path_loaded or 'unknown source'}.")
                    if self.token_mapping: info_parts.append("Token mapping active for manual vocab.")
                else:
                    info_parts.append("\nNo tokenizer/vocabulary loaded for chat.")
                return "\n".join(info_parts)
            load_model_btn_ui.click(ui_load_model_handler, [model_path_load_tb, vocab_path_load_tb], [model_info_load_tb])

            def ui_create_model_handler(size, attn_type, multi, expert, v_size, out_path):
                try:
                    v_size_int = int(v_size) if v_size is not None else 32000
                    new_model = create_apertis_model(
                        model_size=size, vocab_size_override=v_size_int,
                        multimodal=multi, use_expert_system=expert,
                        attention_type_override=attn_type
                    )
                    os.makedirs(out_path, exist_ok=True)
                    new_model.save_pretrained(out_path)
                    dummy_vocab_content = {f"<token_{i}>": i for i in range(v_size_int)}
                    default_specials = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
                    for tok_str, tok_id in default_specials.items():
                        if tok_id < v_size_int:
                            dummy_vocab_content[tok_str] = tok_id
                    with open(os.path.join(out_path, "vocab.json"),"w",encoding="utf-8") as f:
                        json.dump(dummy_vocab_content, f, indent=2)
                    return f"Model (pytorch_model.bin), config.json, and a dummy vocab.json created at {out_path}. " \
                           "Replace dummy vocab.json with a real one or use an HF tokenizer for actual use."
                except Exception as e:
                    logger.error(f"Error creating model: {e}", exc_info=True)
                    return f"Error: {str(e)}"
            create_model_btn_ui.click(ui_create_model_handler,
                                     [new_model_size_dd, new_attn_type_dd, new_multimodal_cb, new_expert_cb, new_vocab_size_num, new_model_out_tb],
                                     [create_model_status_tb])

            def ui_start_training_handler(
                m_s, attn_t, m_m, exp_s, tr_f, v_f, voc_f_std_obj, img_d, b_s, learn_r, eps, eval_ep,
                c_steps, iter_c_steps, g_sel, d_train, g_mem_f, out_d, use_wb, wb_p):
                if not tr_f or not voc_f_std_obj: return "Training & Vocab files required for pre-training."
                if self.standard_training_thread and self.standard_training_thread.is_alive():
                    return "Pre-training is already in progress."

                self.standard_training_stop_event.clear()
                tmp_dir = tempfile.mkdtemp()
                train_p = os.path.join(tmp_dir, "train.jsonl"); shutil.copy(tr_f.name, train_p)
                vocab_p_std = os.path.join(tmp_dir, "vocab.json"); shutil.copy(voc_f_std_obj.name, vocab_p_std)
                val_p = None
                if v_f: val_p = os.path.join(tmp_dir, "val.jsonl"); shutil.copy(v_f.name, val_p)

                sel_gpus = [int(gid) for gid in g_sel] if g_sel else None
                dist_training_eff = d_train if sel_gpus and len(sel_gpus) > 1 else False

                cfg = {
                    "data_config": {"train_data_path":train_p, "tokenizer_path":vocab_p_std, "val_data_path":val_p,
                                    "max_length":512, "multimodal":m_m, "image_dir":img_d if m_m else None,
                                    "image_size": 224},
                    "model_config": {},
                    "training_config": {
                        "task_type": "pretrain",
                        "output_dir":out_d, "batch_size":b_s, "learning_rate":learn_r, "num_epochs":eps,
                        "warmup_steps":0, "gradient_accumulation_steps":4, "max_grad_norm": 1.0,
                        "eval_every_n_epochs":eval_ep, "use_wandb":use_wb, "wandb_project":wb_p if use_wb else None,
                        "wandb_run_name": None,
                        "fp16":True, "device":None,
                        "gpu_memory_fraction":g_mem_f,
                        "use_gradient_checkpointing":True, "dynamic_batch_sizing":True,
                        "checkpoint_steps":c_steps, "iteration_checkpoint_steps":iter_c_steps,
                        "gpu_ids":sel_gpus, "distributed_training":dist_training_eff, "local_rank": -1
                    }
                }
                _presets = {"small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                            "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}}
                if m_s in _presets:
                    cfg["model_config"].update(_presets[m_s])
                cfg["model_config"]["attention_type"] = attn_t
                cfg["model_config"]["multimodal"] = m_m
                cfg["model_config"]["use_expert_system"] = exp_s

                cfg_path = os.path.join(tmp_dir, "run_cfg_pretrain.json");
                with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                out_cfg_path = os.path.join(out_d, "run_cfg_pretrain.json")
                os.makedirs(out_d, exist_ok=True)
                shutil.copy(cfg_path, out_cfg_path)

                def _thread_train_job(c_path, t_dir, stop_event):
                    try:
                        from src.training.pipeline import train_from_config
                        train_from_config(c_path, stop_event)
                    except Exception as e:
                         logger.error(f"Error in Pre-training thread: {e}", exc_info=True)
                    finally:
                        shutil.rmtree(t_dir)
                        self.standard_training_thread = None

                self.standard_training_thread = threading.Thread(target=_thread_train_job, args=(cfg_path, tmp_dir, self.standard_training_stop_event), daemon=True)
                self.standard_training_thread.start()
                return f"Pre-training started. Config: {out_cfg_path}. Output: {out_d}."

            def ui_stop_training_handler():
                if self.standard_training_thread and self.standard_training_thread.is_alive():
                    self.standard_training_stop_event.set()
                    return "Stop request sent to Pre-training. Please wait for current step to finish."
                return "No Pre-training in progress."

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
                if not base_model_path_ui: return "Base model path required for fine-tuning."
                if not ft_data_f_obj: return "Fine-tuning data file required."
                if tokenizer_opt_ui == "Use HF Tokenizer (Recommended from model dir or Hub)" and not hf_tokenizer_name_ui:
                    return "Hugging Face Tokenizer Name/Path required if selected for fine-tuning."

                if self.finetune_training_thread and self.finetune_training_thread.is_alive():
                    return "Fine-tuning is already in progress."

                self.finetune_training_stop_event.clear()
                tmp_dir = tempfile.mkdtemp()

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
                else:
                    tokenizer_path_for_config_ft = None

                sel_gpus = [int(gid) for gid in g_sel_ui] if g_sel_ui else None
                dist_training_eff = d_train_ui if sel_gpus and len(sel_gpus) > 1 else False

                cfg = {
                    "data_config": {
                        "train_data_path": ft_train_p,
                        "val_data_path": ft_val_p,
                        "tokenizer_path": tokenizer_path_for_config_ft,
                        "use_hf_tokenizer_for_finetune": use_hf_for_ft_config,
                        "prompt_template": prompt_template_ui,
                        "max_length": 512
                    },
                    "model_config": {},
                    "training_config": {
                        "task_type": "finetune",
                        "pretrained_model_path_for_finetune": base_model_path_ui,
                        "output_dir": out_d_ui, "batch_size": batch_s_ui, "learning_rate": learn_r_ui,
                        "num_epochs": eps_ui, "eval_every_n_epochs": eval_ep_ui,
                        "warmup_steps": 0, "gradient_accumulation_steps": 1,
                        "max_grad_norm": 1.0,
                        "use_wandb": use_wb_ui, "wandb_project": wb_p_ui if use_wb_ui else None,
                        "fp16": True, "device": None,
                        "gpu_memory_fraction": g_mem_f_ui,
                        "use_gradient_checkpointing": True, "dynamic_batch_sizing": True,
                        "gpu_ids": sel_gpus, "distributed_training": dist_training_eff, "local_rank": -1
                    }
                }

                cfg_path = os.path.join(tmp_dir, "run_cfg_finetune.json")
                with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                out_cfg_path = os.path.join(out_d_ui, "run_cfg_finetune.json")
                os.makedirs(out_d_ui, exist_ok=True)
                shutil.copy(cfg_path, out_cfg_path)

                def _thread_finetune_job(c_path, t_dir, stop_event):
                    try:
                        from src.training.pipeline import train_from_config
                        train_from_config(c_path, stop_event)
                    except Exception as e:
                         logger.error(f"Error in Fine-tuning thread: {e}", exc_info=True)
                    finally:
                        shutil.rmtree(t_dir)
                        self.finetune_training_thread = None

                self.finetune_training_thread = threading.Thread(target=_thread_finetune_job, args=(cfg_path, tmp_dir, self.finetune_training_stop_event), daemon=True)
                self.finetune_training_thread.start()
                return f"Fine-tuning started. Config: {out_cfg_path}. Output: {out_d_ui}."

            def ui_stop_finetuning_handler():
                if self.finetune_training_thread and self.finetune_training_thread.is_alive():
                    self.finetune_training_stop_event.set()
                    return "Stop request sent to Fine-tuning. Please wait for current step to finish."
                return "No Fine-tuning in progress."

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
                max_attempts, temperature, top_p, learn_weight, acc_weight, partial_credit,
                div_weight, history_size, complex_weight, target_complex, tolerance,
                timeout, max_output, g_sel, g_mem_f, out_d, use_wb, wb_p
            ):
                if not tokenizer_name_hf.strip():
                    return "Hugging Face Tokenizer Name is required for AZR."
                if self.azr_training_thread and self.azr_training_thread.is_alive():
                    return "AZR training is already in progress."

                self.azr_training_stop_event.clear()
                tmp_dir = tempfile.mkdtemp()

                seed_p = None
                if seed_f_obj:
                    seed_p = os.path.join(tmp_dir, "seed_tasks.jsonl")
                    shutil.copy(seed_f_obj.name, seed_p)

                task_dist = []
                actual_task_types_for_config = []

                if "abduction" in task_types_list:
                    task_dist.append(abduction_w)
                    actual_task_types_for_config.append("abduction")
                else: task_dist.append(0.0)
                if "deduction" in task_types_list:
                    task_dist.append(deduction_w)
                    actual_task_types_for_config.append("deduction")
                else: task_dist.append(0.0)
                if "induction" in task_types_list:
                    task_dist.append(induction_w)
                    actual_task_types_for_config.append("induction")
                else: task_dist.append(0.0)

                final_task_dist_weights = [w for t, w in zip(["abduction", "deduction", "induction"], task_dist) if t in actual_task_types_for_config]

                if actual_task_types_for_config and sum(final_task_dist_weights) > 0:
                    total_dist_weight = sum(final_task_dist_weights)
                    final_task_dist_weights = [w / total_dist_weight for w in final_task_dist_weights]
                elif actual_task_types_for_config :
                    equal_weight = 1.0 / len(actual_task_types_for_config)
                    final_task_dist_weights = [equal_weight] * len(actual_task_types_for_config)
                else:
                    actual_task_types_for_config = ["abduction", "deduction", "induction"]
                    final_task_dist_weights = [0.3, 0.3, 0.4]


                sel_gpus = [int(gid) for gid in g_sel] if g_sel else None

                azr_model_cfg_base = {
                    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                    "intermediate_size": 3072, "hidden_act": "silu", "max_position_embeddings": 4096,
                    "initializer_range": 0.02, "layer_norm_eps": 1e-6, "use_cache": True,
                    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "unk_token_id":3,
                    "tie_word_embeddings": False, "rope_theta": 10000.0,
                    "attention_probs_dropout_prob": 0.0,
                    "multimodal": m_m, "use_expert_system": exp_s, "attention_type": attn_t
                }
                _presets_azr = {
                    "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
                }
                if m_s in _presets_azr:
                    azr_model_cfg_base.update(_presets_azr[m_s])

                cfg = {
                    "data": {"tokenizer_name": tokenizer_name_hf.strip()},
                    "model": azr_model_cfg_base,
                    "training": {
                        "method": "azr", "output_dir": out_d,
                        "device": "cuda" if torch.cuda.is_available() and sel_gpus else "cpu",
                        "gpu_ids": sel_gpus
                    },
                    "azr": {
                        "num_iterations": iterations, "tasks_per_iteration": tasks_per_iter,
                        "checkpoint_interval": 10, "checkpoint_dir": os.path.join(out_d, "azr_checkpoints"),
                        "force_accept_tasks": True, "force_accept_solutions": True,
                        "force_accept_threshold": 10, "min_valid_tasks_before_validation": 20,
                        "log_level": "INFO", "log_file": os.path.join(out_d, "azr_training.log"),
                        "python_executor": {"timeout": timeout, "max_output_size": max_output},
                        "task_generator": {
                            "task_types": actual_task_types_for_config, "task_distribution": final_task_dist_weights,
                            "max_attempts": max_attempts, "seed_tasks_path": seed_p,
                            "seed_task_probability": seed_prob,
                            "base_prompt": "Generate a challenging reasoning problem.",
                            "max_new_tokens": 100, "temperature": temperature, "top_p": top_p
                        },
                        "task_validator": {"min_length": 10, "max_length": 2500, "min_complexity": 0.1, "max_complexity": 1.0, "min_clarity": 0.3},
                        "solution_generator": {"max_attempts": max_attempts, "base_prompt": "Solve the following problem step by step:", "include_task_type_hint": True, "max_new_tokens": 1024, "temperature": temperature, "top_p": top_p},
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
                out_cfg_path = os.path.join(out_d, "azr_config.json")
                shutil.copy(cfg_path, out_cfg_path)

                def _thread_azr_train(c_path, t_dir, stop_event):
                    try:
                        from src.training import train_from_config
                        train_from_config(c_path, stop_event)
                    except Exception as e:
                        logger.error(f"Error in AZR training thread: {e}", exc_info=True)
                    finally:
                        shutil.rmtree(t_dir)
                        self.azr_training_thread = None

                self.azr_training_thread = threading.Thread(target=_thread_azr_train, args=(cfg_path, tmp_dir, self.azr_training_stop_event), daemon=True)
                self.azr_training_thread.start()
                return f"AZR Training started. Config: {out_cfg_path}. Output: {out_d}."

            def ui_stop_azr_training_handler():
                if self.azr_training_thread and self.azr_training_thread.is_alive():
                    self.azr_training_stop_event.set()
                    return "Stop request sent to AZR Training. Please wait for current iteration to finish."
                return "No AZR Training in progress."

            azr_start_btn.click(
                ui_start_azr_training_handler,
                [
                    azr_model_size_dd, azr_attn_type_dd, azr_multimodal_cb, azr_expert_sys_cb,
                    azr_tokenizer_name_tb,
                    azr_seed_tasks_up, azr_seed_prob_sl,
                    azr_iterations_sl, azr_tasks_per_iter_sl, azr_task_types,
                    azr_task_dist_abduction, azr_task_dist_deduction, azr_task_dist_induction,
                    azr_max_attempts_sl, azr_temperature_sl, azr_top_p_sl,
                    azr_learn_weight_sl, azr_acc_weight_sl, azr_partial_credit_cb,
                    azr_div_weight_sl, azr_history_size_sl, azr_complex_weight_sl,
                    azr_target_complex_sl, azr_tolerance_sl, azr_timeout_sl, azr_max_output_sl,
                    azr_gpu_select_cbg, azr_gpu_mem_frac_sl, azr_output_dir_tb,
                    azr_wandb_cb, azr_wandb_proj_tb
                ],
                [azr_status_tb]
            )
            azr_stop_btn.click(ui_stop_azr_training_handler, outputs=[azr_status_tb])

        try:
            interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share, max_threads=60, prevent_thread_lock=True)
        except OSError as e:
            if "Can't assign requested address" in str(e) and self.port == 7860 :
                logger.warning(f"Port {self.port} might be in use. Trying 7861...")
                self.port = 7861
                interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share, max_threads=60, prevent_thread_lock=True)
            else:
                logger.error(f"Failed to launch Gradio interface: {e}", exc_info=True)
                print(f"Failed to launch Gradio interface on port {self.port}. Please check if the port is available or try another one.")