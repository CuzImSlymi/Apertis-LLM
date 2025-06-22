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
from src.training.pipeline import get_available_gpus
from src.training import train_from_config


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
            
            if not os.path.isdir(path_to_check):
                try:
                    self.hf_tokenizer_chat = AutoTokenizer.from_pretrained(path_to_check)
                    logger.info(f"Successfully loaded Hugging Face tokenizer from Hub ID/path: {path_to_check}")
                    self.actual_tokenizer_path_loaded = path_to_check 
                    return True
                except EnvironmentError:
                    logger.debug(f"'{path_to_check}' is not a local tokenizer directory and failed to load as Hub ID.")
                    pass
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
        if os.path.isfile(model_path_or_name):
            potential_tokenizer_base_dir = os.path.dirname(model_path_or_name)
        
        if os.path.isdir(potential_tokenizer_base_dir):
            logger.info(f"Checking for HF tokenizer in potential base directory: {potential_tokenizer_base_dir}")
            if self._attempt_load_hf_tokenizer(potential_tokenizer_base_dir):
                tokenizer_found_and_loaded = True
        
        if not tokenizer_found_and_loaded and model_path_or_name != potential_tokenizer_base_dir:
            logger.info(f"Base directory tokenizer check failed or paths differ. Checking original path '{model_path_or_name}' as HF tokenizer source.")
            if self._attempt_load_hf_tokenizer(model_path_or_name):
                tokenizer_found_and_loaded = True
        
        self.load_model(model_path_or_name)

        if self.model and not self.hf_tokenizer_chat:
            manual_vocab_path_to_try = None
            if vocab_file_override:
                manual_vocab_path_to_try = vocab_file_override
            elif os.path.isdir(potential_tokenizer_base_dir):
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
            
            model_config_base_dir = model_weights_path_input
            if os.path.isfile(model_weights_path_input):
                model_config_base_dir = os.path.dirname(model_weights_path_input)
            
            config_json_path = os.path.join(model_config_base_dir, "config.json")
            if os.path.exists(config_json_path):
                config_for_model_instantiation = ApertisConfig.from_pretrained(config_json_path)
                logger.info(f"Loaded config.json from {config_json_path}. Initial vocab_size from file: {config_for_model_instantiation.vocab_size if config_for_model_instantiation else 'N/A'}")
            else:
                logger.warning(f"config.json not found in {model_config_base_dir}. Will attempt to infer or use defaults if creating new.")

            if config_for_model_instantiation and self.hf_tokenizer_chat:
                logger.info("HF tokenizer is loaded. Aligning model config with its properties.")
                config_for_model_instantiation.vocab_size = self.hf_tokenizer_chat.vocab_size
                if self.hf_tokenizer_chat.pad_token_id is not None: config_for_model_instantiation.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                if self.hf_tokenizer_chat.bos_token_id is not None: config_for_model_instantiation.bos_token_id = self.hf_tokenizer_chat.bos_token_id
                if self.hf_tokenizer_chat.eos_token_id is not None: config_for_model_instantiation.eos_token_id = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.unk_token_id is not None: config_for_model_instantiation.unk_token_id = self.hf_tokenizer_chat.unk_token_id
                logger.info(f"Model config aligned with HF tokenizer. New vocab_size: {config_for_model_instantiation.vocab_size}")
            
            state_dict_path_final = None
            if os.path.isdir(model_weights_path_input):
                bin_path = os.path.join(model_weights_path_input, "pytorch_model.bin")
                pt_path = os.path.join(model_weights_path_input, "model.pt")
                if os.path.exists(bin_path): state_dict_path_final = bin_path
                elif os.path.exists(pt_path): state_dict_path_final = pt_path
            elif os.path.isfile(model_weights_path_input) and \
                 (model_weights_path_input.endswith(".pt") or model_weights_path_input.endswith(".bin")):
                state_dict_path_final = model_weights_path_input
            
            if not config_for_model_instantiation and state_dict_path_final:
                logger.info(f"No config.json. Attempting to infer config from state_dict: {state_dict_path_final}")
                config_for_model_instantiation = self._infer_config_from_state_dict(state_dict_path_final)
                if config_for_model_instantiation and self.hf_tokenizer_chat:
                    config_for_model_instantiation.vocab_size = self.hf_tokenizer_chat.vocab_size
                    if self.hf_tokenizer_chat.pad_token_id is not None: config_for_model_instantiation.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                    logger.info(f"Inferred config aligned with HF tokenizer. Vocab_size: {config_for_model_instantiation.vocab_size}")


            if config_for_model_instantiation:
                self.model = ApertisForCausalLM(config_for_model_instantiation)
                if state_dict_path_final and os.path.exists(state_dict_path_final):
                    loaded_state_dict = torch.load(state_dict_path_final, map_location=self.device, weights_only=True)
                    
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
                        self.model.resize_token_embeddings(sd_vocab_size)
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
            
            vocab_size = 0
            if "model.token_embeddings.weight" in state_dict:
                vocab_size = state_dict["model.token_embeddings.weight"].shape[0]
            elif "lm_head.weight" in state_dict:
                vocab_size = state_dict["lm_head.weight"].shape[0]
            else:
                vocab_size = self.hf_tokenizer_chat.vocab_size if self.hf_tokenizer_chat else 32000
            
            hidden_size = 0
            if "model.token_embeddings.weight" in state_dict:
                hidden_size = state_dict["model.token_embeddings.weight"].shape[1]
            elif "model.layers.0.attention.q_proj.weight" in state_dict:
                hidden_size = state_dict["model.layers.0.attention.q_proj.weight"].shape[1]
            elif "lm_head.weight" in state_dict:
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
            if not multimodal_inferred and hasattr(self, 'multimodal'):
                conf_params["multimodal"] = self.multimodal
            
            conf = ApertisConfig(**conf_params)
            
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
                if self.vocab: max_id_in_loaded_vocab = max(self.vocab.values() or [-1])
                effective_loaded_vocab_size = max_id_in_loaded_vocab + 1
                
                if model_cfg_vocab_size != effective_loaded_vocab_size :
                    logger.warning(
                        f"Model config vocab_size ({model_cfg_vocab_size}) != "
                        f"effective manual vocab_file size ({effective_loaded_vocab_size}). "
                        "Token mapping may be needed if used directly. "
                        "Consider resizing the model or ensuring vocab consistency."
                    )
                    self.token_mapping = None
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
        for i in range(4, 100): self.vocab[f"<tok{i}>"] = i
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_mapping = None
        self.actual_tokenizer_path_loaded = "Fallback minimal vocab (100 tokens)"

    def tokenize(self, text: str) -> List[int]:
        if not self.model:
            logger.error("Model not loaded. Cannot tokenize.")
            return []
        
        if self.hf_tokenizer_chat:
            return self.hf_tokenizer_chat.encode(text, add_special_tokens=False)

        if not self.vocab:
            logger.warning("Manual vocab not loaded for tokenize. Using fallback tokenization.")
            base_unk_id = self.model.config.unk_token_id
            return [self.vocab.get(word, base_unk_id) if self.vocab else base_unk_id for word in text.split()]

        model_config = self.model.config
        raw_token_ids_from_vocab_file = [
            self.vocab.get(word, self.vocab.get("<unk>", model_config.unk_token_id)) for word in text.split()
        ]
        
        final_token_ids = []
        for tid in raw_token_ids_from_vocab_file:
            if tid >= model_config.vocab_size:
                final_token_ids.append(model_config.unk_token_id)
            else:
                final_token_ids.append(tid)
        return final_token_ids

    def detokenize(self, token_ids: List[int]) -> str:
        if not self.model:
            return f"[DetokenizeError: Model not loaded. IDs: {token_ids[:5]}...]"

        if self.hf_tokenizer_chat:
            return self.hf_tokenizer_chat.decode(token_ids, skip_special_tokens=True)

        if not self.reverse_vocab:
            return f"[DetokenizeError: Manual Reverse Vocab missing. IDs: {token_ids[:5]}...]"

        words = []
        model_config = self.model.config
        default_unk_str = "<unk>"
        if "<unk>" in self.vocab:
            unk_id_in_vocab_file = self.vocab["<unk>"]
            if unk_id_in_vocab_file in self.reverse_vocab:
                default_unk_str = self.reverse_vocab[unk_id_in_vocab_file]
        
        for model_token_id in token_ids:
            if model_token_id == model_config.pad_token_id or \
               model_token_id == model_config.bos_token_id or \
               model_token_id == model_config.eos_token_id:
                continue
            
            word = self.reverse_vocab.get(model_token_id)
            if word is not None:
                words.append(word)
            else:
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
                input_ids_list_for_model = self.hf_tokenizer_chat.encode(prompt, add_special_tokens=True)
            else:
                tokenized_prompt = self.tokenize(prompt)
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
            
            if self.hf_tokenizer_chat:
                if self.hf_tokenizer_chat.eos_token_id is not None: eos_token_id_for_gen = self.hf_tokenizer_chat.eos_token_id
                if self.hf_tokenizer_chat.pad_token_id is not None: pad_token_id_for_gen = self.hf_tokenizer_chat.pad_token_id
            
            output_ids = self.model.generate(
                input_ids=input_t, attention_mask=attention_mask_t, pixel_values=pixel_values_t,
                max_new_tokens=max_length, do_sample=temperature > 0.001,
                temperature=temperature if temperature > 0.001 else 1.0,
                top_k=top_k if top_k > 0 else 0,
                top_p=top_p if top_p < 1.0 else 1.0,
                use_cache=True,
                eos_token_id=eos_token_id_for_gen,
                pad_token_id=pad_token_id_for_gen
            )
            
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
                                top_p_slider_chat = gr.Slider(0.0, 1.0, 0.9, step=0.05, label="Top P (1.0=disable)")

                with gr.TabItem("Pre-training"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Config")
                            # Updated model size list
                            new_model_size_list = ["50M", "100M", "250M", "500M", "750M", "1B", "1.3B", "3B", "7B", "13B", "30B", "70B"]
                            model_size_train_dd = gr.Dropdown(new_model_size_list, value="1B", label="Base Model Size")
                            attn_type_train_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            flash_attn_train_cb = gr.Checkbox(label="Use FlashAttention (for Standard MHA)", value=False, visible=True) # Visible by default
                            multimodal_train_cb = gr.Checkbox(label="Multimodal")
                            expert_sys_train_cb = gr.Checkbox(label="Use Expert System")

                            # MoE Controls (initially hidden)
                            with gr.Group(visible=False) as moe_options_group_train:
                                gr.Markdown("### MoE Configuration")
                                num_experts_train_sl = gr.Slider(1, 64, value=8, step=1, label="Number of Experts (Total)")
                                experts_per_token_train_sl = gr.Slider(1, 8, value=2, step=1, label="Active Experts per Token")

                            # Parameter and Memory Display
                            param_display_train_md = gr.Markdown("Params: N/A | Memory: N/A")

                            def _toggle_flash_attn_visibility_train(attn_type_val):
                                # Flash attention only for standard_mha
                                return gr.update(visible=(attn_type_val == "standard_mha"), value=False if attn_type_val != "standard_mha" else gr.UNCHANGED)

                            attn_type_train_dd.change(
                                _toggle_flash_attn_visibility_train,
                                inputs=[attn_type_train_dd],
                                outputs=[flash_attn_train_cb]
                            )

                            def _toggle_moe_visibility_train(use_moe_val):
                                return gr.update(visible=use_moe_val)
                            expert_sys_train_cb.change(_toggle_moe_visibility_train, inputs=[expert_sys_train_cb], outputs=[moe_options_group_train])

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
                            gr.Markdown("## 1. AZR Model & Data")
                            with gr.Accordion("Model Config (for AZR internal model)", open=True):
                                new_model_size_list_azr = ["50M", "100M", "250M", "500M", "750M", "1B", "1.3B", "3B", "7B", "13B", "30B", "70B"]
                                azr_model_size_dd = gr.Dropdown(new_model_size_list_azr, value="1B", label="Base Model Size")
                                azr_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                                azr_flash_attn_cb = gr.Checkbox(label="Use FlashAttention (for Standard MHA)", value=False, visible=True) # Visible by default
                                # Adding expert system checkbox for AZR model config
                                azr_expert_sys_cb = gr.Checkbox(label="Use Expert System (for AZR model)")

                                # MoE Controls for AZR (initially hidden)
                                with gr.Group(visible=False) as moe_options_group_azr:
                                    gr.Markdown("### MoE Configuration (AZR Model)")
                                    num_experts_azr_sl = gr.Slider(1, 64, value=8, step=1, label="Number of Experts (Total)")
                                    experts_per_token_azr_sl = gr.Slider(1, 8, value=2, step=1, label="Active Experts per Token")

                                # Parameter and Memory Display for AZR model
                                param_display_azr_md = gr.Markdown("Params: N/A | Memory: N/A")

                                def _toggle_flash_attn_visibility_azr(attn_type_val):
                                    return gr.update(visible=(attn_type_val == "standard_mha"), value=False if attn_type_val != "standard_mha" else gr.UNCHANGED)
                                azr_attn_type_dd.change(_toggle_flash_attn_visibility_azr, inputs=[azr_attn_type_dd], outputs=[azr_flash_attn_cb])

                                def _toggle_moe_visibility_azr(use_moe_val):
                                    return gr.update(visible=use_moe_val)
                                azr_expert_sys_cb.change(_toggle_moe_visibility_azr, inputs=[azr_expert_sys_cb], outputs=[moe_options_group_azr])

                            with gr.Accordion("Tokenizer & Seed Data", open=True):
                                azr_tokenizer_name_tb = gr.Textbox(value="gpt2", label="Hugging Face Tokenizer Name", placeholder="e.g., gpt2, meta-llama/Llama-2-7b-hf")
                                azr_seed_tasks_up = gr.File(label="Seed Tasks (JSONL, optional, fields: 'task', 'type')", file_types=[".jsonl"])

                            gr.Markdown("## 2. Training Parameters")
                            with gr.Accordion("Main Loop", open=True):
                                azr_iterations_sl = gr.Slider(10, 1000, 100, step=10, label="Number of Iterations")
                                azr_tasks_per_iter_sl = gr.Slider(1, 50, 5, step=1, label="Tasks Per Iteration")
                                azr_checkpoint_interval_sl = gr.Slider(1, 100, 10, step=1, label="Checkpoint Every N Iterations")
                            with gr.Accordion("Task Generation", open=False):
                                azr_task_types_cbg = gr.CheckboxGroup(["abduction", "deduction", "induction"], value=["abduction", "deduction", "induction"], label="Task Types")
                                azr_task_dist_abduction_sl = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Abduction Weight")
                                azr_task_dist_deduction_sl = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Deduction Weight")
                                azr_task_dist_induction_sl = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Induction Weight")
                                azr_max_attempts_sl = gr.Slider(1, 10, 3, step=1, label="Max Generation Attempts")
                                azr_temperature_sl = gr.Slider(0.1, 1.5, 0.7, step=0.05, label="Generation Temperature")
                                azr_top_p_sl = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-P (1.0 = disable)")
                                azr_seed_prob_sl = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Seed Task Probability")
                            with gr.Accordion("Python Executor (for validation)", open=False):
                                azr_timeout_sl = gr.Slider(1, 30, 5, step=1, label="Execution Timeout (seconds)")
                                azr_max_output_sl = gr.Slider(1000, 50000, 10000, step=1000, label="Max Output Size (chars)")

                        with gr.Column(scale=1):
                            gr.Markdown("## 3. Reward System")
                            with gr.Accordion("Task Reward Weights", open=True):
                                azr_clarity_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Clarity Weight")
                                azr_complex_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Complexity Weight")
                                azr_div_weight_sl = gr.Slider(0.0, 2.0, 0.8, step=0.1, label="Diversity Weight")
                            with gr.Accordion("Solution Reward Weights", open=True):
                                azr_acc_weight_sl = gr.Slider(0.0, 2.0, 1.5, step=0.1, label="Accuracy Weight")
                                azr_coherence_weight_sl = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Coherence Weight")
                                azr_relevance_weight_sl = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Relevance Weight")
                                azr_structure_weight_sl = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Structure Weight")
                            with gr.Accordion("Advanced Reward Parameters", open=False):
                                gr.Markdown("Complexity Target")
                                azr_target_complex_sl = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Target Complexity")
                                azr_tolerance_sl = gr.Slider(0.0, 0.5, 0.15, step=0.05, label="Tolerance")
                                gr.Markdown("Accuracy Power")
                                azr_acc_power_sl = gr.Slider(1.0, 3.0, 1.5, step=0.1, label="Partial Credit Power")

                            gr.Markdown("## 4. Execution & Output")
                            with gr.Accordion("GPU & Logging", open=False):
                                available_gpus_list_azr_re = get_available_gpus()
                                gpu_md_text_azr_re = "### GPUs:\n" + ("\n".join([f"- {g['id']}: {g['name']} ({g['total_memory']:.1f}GB)" for g in available_gpus_list_azr_re]) or "None detected.")
                                gr.Markdown(gpu_md_text_azr_re)
                                azr_gpu_select_dd = gr.Dropdown(choices=[f"cuda:{g['id']}" for g in available_gpus_list_azr_re] + ["cpu"], value=f"cuda:{available_gpus_list_azr_re[0]['id']}" if available_gpus_list_azr_re else "cpu", label="Select Device")
                                azr_log_level_dd = gr.Dropdown(["DEBUG", "INFO", "WARNING", "ERROR"], value="INFO", label="Log Level")
                                azr_wandb_cb = gr.Checkbox(label="Log to W&B")
                                azr_wandb_proj_tb = gr.Textbox("apertis-azr", label="W&B Project", visible=False)
                                azr_wandb_cb.change(lambda x: gr.update(visible=x), inputs=[azr_wandb_cb], outputs=[azr_wandb_proj_tb])
                            
                            azr_output_dir_tb = gr.Textbox("output_azr", label="Output Directory")
                            
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
                            new_model_size_list_create = ["50M", "100M", "250M", "500M", "750M", "1B", "1.3B", "3B", "7B", "13B", "30B", "70B"]
                            new_model_size_dd = gr.Dropdown(new_model_size_list_create, value="1B", label="Model Size")
                            new_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            new_flash_attn_cb = gr.Checkbox(label="Use FlashAttention (for Standard MHA)", value=False, visible=True) # Visible by default
                            new_multimodal_cb = gr.Checkbox(label="Multimodal")
                            new_expert_cb = gr.Checkbox(label="Use Expert System")

                            # MoE Controls for Create New Model (initially hidden)
                            with gr.Group(visible=False) as moe_options_group_create:
                                gr.Markdown("### MoE Configuration (New Model)")
                                num_experts_create_sl = gr.Slider(1, 64, value=8, step=1, label="Number of Experts (Total)")
                                experts_per_token_create_sl = gr.Slider(1, 8, value=2, step=1, label="Active Experts per Token")

                            new_vocab_size_num = gr.Number(32000, label="Vocab Size (for manual vocab)", precision=0)
                            new_model_out_tb = gr.Textbox("models/new_apertis_model", label="Save Path for New Model Files")

                            # Parameter and Memory Display for new model
                            param_display_create_md = gr.Markdown("Params: N/A | Memory: N/A")

                            create_model_btn_ui = gr.Button("Create & Save New Model Files")
                            create_model_status_tb = gr.Textbox(label="Creation Status", interactive=False, lines=3)

            def _toggle_flash_attn_visibility_new_model(attn_type_val):
                return gr.update(visible=(attn_type_val == "standard_mha"), value=False if attn_type_val != "standard_mha" else gr.UNCHANGED)

            new_attn_type_dd.change(
                _toggle_flash_attn_visibility_new_model,
                inputs=[new_attn_type_dd],
                outputs=[new_flash_attn_cb]
            )

            def _toggle_moe_visibility_create(use_moe_val):
                return gr.update(visible=use_moe_val)
            new_expert_cb.change(_toggle_moe_visibility_create, inputs=[new_expert_cb], outputs=[moe_options_group_create])


            # Centralized function to update parameter/memory display
            # It needs all relevant inputs from a given tab
            def update_param_memory_display(*args):
                # args: model_size, attn_type, use_flash_attn, use_expert_system,
                #       num_experts, experts_per_token, vocab_size (optional, for create new)
                #       batch_size_ui (for memory est), seq_len_ui (for memory est)

                # Determine which set of inputs we got based on length or specific None values
                # This is a bit fragile; named arguments to a Python function called by Gradio
                # via .change() is cleaner if possible, or pass a dict.
                # For now, make it flexible.

                try:
                    # Common args
                    model_size_val = args[0]
                    # attn_type_val = args[1] # Not directly used in param count, but good to have
                    # use_flash_attn_val = args[2] # Not used in param count
                    use_expert_system_val = args[3]
                    num_experts_val = args[4]
                    experts_per_token_val = args[5]

                    # Defaults for optional/context-specific args
                    vocab_size_val = 32000 # Default vocab size
                    batch_size_for_mem = 4  # Default batch for memory estimation
                    seq_len_for_mem = 512   # Default seq_len for memory estimation

                    arg_idx = 6
                    if len(args) > arg_idx and args[arg_idx] is not None and not isinstance(args[arg_idx], gr.Request): # vocab_size_override if present
                        try: vocab_size_val = int(args[arg_idx])
                        except (ValueError, TypeError): pass # Keep default if not convertible
                        arg_idx +=1

                    if len(args) > arg_idx and args[arg_idx] is not None and not isinstance(args[arg_idx], gr.Request): # batch_size for mem est
                        try: batch_size_for_mem = int(args[arg_idx])
                        except (ValueError, TypeError): pass
                        arg_idx +=1

                    if len(args) > arg_idx and args[arg_idx] is not None and not isinstance(args[arg_idx], gr.Request): # seq_len for mem est
                        try: seq_len_for_mem = int(args[arg_idx])
                        except (ValueError, TypeError): pass
                        # arg_idx +=1 # No more args after this for now

                    _presets = {
                        "50M": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 1376},
                        "100M": {"hidden_size": 768, "num_hidden_layers": 10, "num_attention_heads": 12, "intermediate_size": 2048},
                        "250M": {"hidden_size": 1024, "num_hidden_layers": 12, "num_attention_heads": 16, "intermediate_size": 2816},
                        "500M": {"hidden_size": 1280, "num_hidden_layers": 16, "num_attention_heads": 20, "intermediate_size": 3584},
                        "750M": {"hidden_size": 1536, "num_hidden_layers": 18, "num_attention_heads": 24, "intermediate_size": 4096},
                        "1B": {"hidden_size": 1536, "num_hidden_layers": 24, "num_attention_heads": 24, "intermediate_size": 4352},
                        "1.3B": {"hidden_size": 2048, "num_hidden_layers": 18, "num_attention_heads": 32, "intermediate_size": 5632},
                        "3B": {"hidden_size": 2560, "num_hidden_layers": 26, "num_attention_heads": 40, "intermediate_size": 6912},
                        "7B": {"hidden_size": 3200, "num_hidden_layers": 36, "num_attention_heads": 50, "intermediate_size": 8704},
                        "13B": {"hidden_size": 4096, "num_hidden_layers": 40, "num_attention_heads": 64, "intermediate_size": 11008},
                        "30B": {"hidden_size": 5120, "num_hidden_layers": 48, "num_attention_heads": 80, "intermediate_size": 13824},
                        "70B": {"hidden_size": 6144, "num_hidden_layers": 64, "num_attention_heads": 96, "intermediate_size": 16384},
                    }
                    base_dims = _presets.get(model_size_val, _presets["1B"]) # Default to 1B if somehow invalid

                    cfg_for_count = ApertisConfig(
                        vocab_size=int(vocab_size_val),
                        hidden_size=base_dims["hidden_size"],
                        num_hidden_layers=base_dims["num_hidden_layers"],
                        num_attention_heads=base_dims["num_attention_heads"],
                        intermediate_size=base_dims["intermediate_size"],
                        use_expert_system=bool(use_expert_system_val),
                        num_experts=int(num_experts_val) if use_expert_system_val else 0,
                        experts_per_token=int(experts_per_token_val) if use_expert_system_val else 0,
                        # other params like attention_type can be added if they affect param count significantly
                    )

                    params_info = cfg_for_count.count_parameters()
                    mem_info = cfg_for_count.estimate_memory_usage(batch_size=int(batch_size_for_mem), seq_len=int(seq_len_for_mem))

                    total_p = params_info['total_parameters']
                    active_p = params_info['active_parameters']

                    param_str = f"{total_p/1e9:.2f}B" if total_p >= 1e9 else f"{total_p/1e6:.1f}M"
                    active_param_str = f"{active_p/1e9:.2f}B" if active_p >= 1e9 else f"{active_p/1e6:.1f}M"

                    display_text = f"Total Params: {param_str}"
                    if use_expert_system_val and cfg_for_count.num_experts > 0 and cfg_for_count.experts_per_token > 0 :
                        # Calculate the size of one expert's FFN part per layer more directly
                        # H*I + I*H for one FFN
                        single_expert_ffn_params_total = (cfg_for_count.hidden_size * cfg_for_count.intermediate_size +
                                                          cfg_for_count.intermediate_size * cfg_for_count.hidden_size) * cfg_for_count.num_hidden_layers

                        expert_size_str = f"{single_expert_ffn_params_total / 1e9:.2f}B" if single_expert_ffn_params_total >=1e9 else f"{single_expert_ffn_params_total/1e6:.1f}M"
                        # Format for MoE display: "Total: 8B | Active: 1.3B | Experts: 6x1.3B" (example)
                        # Here, expert_size_str is total params for ONE expert across ALL layers.
                        # The "X x YB" usually means NumTotalExperts x SizeOfOneExpertFFN (for all layers)
                        # Or sometimes NumActiveExperts x SizeOfOneExpertFFN
                        # Let's use: NumTotalExperts x SizeOfOneExpertFFN_all_layers
                        display_text = f"Total: {param_str} | Active: {active_param_str} | Experts: {cfg_for_count.num_experts}x{expert_size_str}"

                    display_text += f"\nMem Est (B={batch_size_for_mem}, S={seq_len_for_mem}, FP16): Train {mem_info['training_gb']:.1f}GB | Infer {mem_info['inference_gb']:.1f}GB"
                    return display_text
                except Exception as e:
                    logger.error(f"Error updating param/memory display: {e}", exc_info=True)
                    return "Error calculating params/memory."

            # Wire up the param/memory display updates
            # For Pre-training Tab
            pretrain_param_inputs = [
                model_size_train_dd, attn_type_train_dd, flash_attn_train_cb, expert_sys_train_cb,
                num_experts_train_sl, experts_per_token_train_sl,
                gr.Number(value=32000, visible=False), # Placeholder for vocab_size_override (not directly editable here)
                batch_size_train_sl, # Pass the training batch size slider
                gr.Number(value=512, visible=False) # Placeholder for seq_len for memory (can make this a UI input if needed)
            ]
            # Listen to changes on components that affect parameter or memory calculation
            comps_to_listen_pretrain = [model_size_train_dd, expert_sys_train_cb, num_experts_train_sl, experts_per_token_train_sl, batch_size_train_sl]
            for comp in comps_to_listen_pretrain: # Iterate through the actual Gradio components
                 if comp: # Check if component is not None (it shouldn't be for these)
                    comp.change(update_param_memory_display, inputs=pretrain_param_inputs, outputs=[param_display_train_md])

            # For AZR Tab
            azr_param_inputs = [
                azr_model_size_dd, azr_attn_type_dd, azr_flash_attn_cb, azr_expert_sys_cb,
                num_experts_azr_sl, experts_per_token_azr_sl,
                gr.Number(value=32000, visible=False), # vocab (AZR uses HF tokenizer, so this is for consistency)
                gr.Number(value=1, visible=False),   # batch for mem (AZR generation is usually B=1)
                gr.Number(value=512, visible=False) # seq_len for mem
            ]
            comps_to_listen_azr = [azr_model_size_dd, azr_expert_sys_cb, num_experts_azr_sl, experts_per_token_azr_sl]
            for comp in comps_to_listen_azr:
                 if comp:
                    comp.change(update_param_memory_display, inputs=azr_param_inputs, outputs=[param_display_azr_md])

            # For Create New Model Tab
            create_model_param_inputs = [
                new_model_size_dd, new_attn_type_dd, new_flash_attn_cb, new_expert_cb,
                num_experts_create_sl, experts_per_token_create_sl,
                new_vocab_size_num, # Vocab size is an input here
                gr.Number(value=1, visible=False),   # batch for mem (default for creation display)
                gr.Number(value=512, visible=False) # seq_len for mem
            ]
            comps_to_listen_create = [new_model_size_dd, new_expert_cb, num_experts_create_sl, experts_per_token_create_sl, new_vocab_size_num]
            for comp in comps_to_listen_create: # Iterate through actual components
                if comp:
                    comp.change(update_param_memory_display, inputs=create_model_param_inputs, outputs=[param_display_create_md])


            def ui_chat_handler(msg, img, max_new, temp, tk, tp, hist):
                if not self.model:
                    gr.Warning("No model loaded. Please load a model from the 'Models' tab.")
                    return hist, ""
                if not msg.strip() and not img: 
                    gr.Info("Please provide a message or an image.")
                    return hist, ""
                
                response_from_model = self.chat(msg, img, max_new, temp, tk, tp)
                
                gradio_display_history = []
                temp_internal_history_for_display = self.chat_history.copy()
                
                user_turn_text_for_display = None
                for entry_idx in range(len(temp_internal_history_for_display)):
                    if temp_internal_history_for_display[entry_idx]["role"] == "user":
                        user_turn_text_for_display = temp_internal_history_for_display[entry_idx]["content"]
                    elif temp_internal_history_for_display[entry_idx]["role"] == "assistant" and user_turn_text_for_display is not None:
                        gradio_display_history.append( (user_turn_text_for_display, temp_internal_history_for_display[entry_idx]["content"]) )
                        user_turn_text_for_display = None

                return gradio_display_history, ""


            submit_btn_chat.click(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            msg_textbox.submit(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])

            def ui_clear_chat_handler():
                self.reset_chat()
                return [], "", None
            clear_btn_chat.click(ui_clear_chat_handler, outputs=[chatbot_ui, msg_textbox, img_input_chat])


            def ui_load_model_handler(m_path_ui, v_path_override_ui):
                if not m_path_ui:
                    return "Please provide a model path or name."
                # Note: Loading an existing model will use the 'use_flash_attention' from its config.json.
                # There's no UI override here for existing models' flash attention setting during load.
                self.load_model_and_tokenizer_from_path(m_path_ui, v_path_override_ui if v_path_override_ui else None)
                
                info_parts = [f"Attempted to load model using input path/name: {m_path_ui}"]
                info_parts.append(f"  Actual model resource path: {self.actual_model_path_loaded or 'N/A'}")
                
                if self.model and hasattr(self.model.config, 'to_dict'):
                    cfg_dict = self.model.config.to_dict()
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

            def ui_create_model_handler(size_ui, attn_type_ui, multi_ui, expert_ui, flash_attn_ui, v_size_ui, out_path_ui):
                try:
                    if not out_path_ui: return "Output path for new model files is required."
                    v_size_int = int(v_size_ui) if v_size_ui is not None else 32000
                    
                    # Determine effective use_flash_attention: only if attn_type is standard_mha
                    effective_flash_attn = flash_attn_ui if attn_type_ui == "standard_mha" else False

                    new_model_instance = create_apertis_model(
                        model_size=size_ui, vocab_size_override=v_size_int,
                        multimodal=multi_ui, use_expert_system=expert_ui,
                        attention_type_override=attn_type_ui,
                        use_flash_attention=effective_flash_attn # Pass the new flag
                    )
                    new_model_instance.save_pretrained(out_path_ui)
                    
                    dummy_vocab_content = {f"<token_{i}>": i for i in range(v_size_int)}
                    default_specials = {
                        "<pad>": new_model_instance.config.pad_token_id, 
                        "<bos>": new_model_instance.config.bos_token_id, 
                        "<eos>": new_model_instance.config.eos_token_id, 
                        "<unk>": new_model_instance.config.unk_token_id
                    }
                    for tok_str, tok_id in default_specials.items():
                        if tok_id < v_size_int:
                            dummy_vocab_content[tok_str] = tok_id
                        else:
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
                                     [new_model_size_dd, new_attn_type_dd, new_multimodal_cb, new_expert_cb, new_flash_attn_cb, new_vocab_size_num, new_model_out_tb],
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
                        "model_config": {
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
                    
                    train_status_tb.value = "Initializing pre-training..."
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

            # Removed the first start_train_btn.click call that used ui_start_training_handler
            # It is now superseded by the call below using ui_start_training_handler_updated
            stop_train_btn.click(ui_stop_training_handler, outputs=[train_status_tb])

            # Update the signature and logic for ui_start_training_handler
            # The old handler ui_start_training_handler and its .click() registration were removed.
            # This is the new handler.
            def ui_start_training_handler_updated(
                m_s, attn_t, flash_attn_t_cb_val, # Added flash_attn_t_cb_val
                m_m, exp_s, tr_f_obj, v_f_obj, voc_f_std_obj, img_d, b_s, learn_r, eps, eval_ep,
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

                    effective_flash_attn_train = flash_attn_t_cb_val if attn_t == "standard_mha" else False

                    cfg = {
                        "data_config": {"train_data_path":train_p, "tokenizer_path":vocab_p_std, "val_data_path":val_p,
                                        "max_length":512, "multimodal":m_m, "image_dir":img_d if m_m else None,
                                        "image_size": 224},
                        "model_config": {
                            "model_size":m_s, "attention_type": attn_t,
                            "use_flash_attention": effective_flash_attn_train, # Added here
                            "multimodal":m_m, "use_expert_system":exp_s
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

                    train_status_tb.value = "Initializing pre-training..."
                    self.standard_training_thread = threading.Thread(target=_thread_train_job, args=(cfg_path, tmp_dir, self.standard_training_stop_event, train_status_tb), daemon=True)
                    self.standard_training_thread.start()
                    return f"Pre-training initiated. Output will be in '{out_d}'. Monitor console/W&B for progress. Copied config to '{final_cfg_path_in_output}'."
                except Exception as e_start:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    logger.error(f"Failed to start pre-training: {e_start}", exc_info=True)
                    return f"Failed to start pre-training: {e_start}"

            # Update the click handler to use the new function and pass the new checkbox value
            start_train_btn.click(ui_start_training_handler_updated, # Use updated handler
                                 [model_size_train_dd, attn_type_train_dd, flash_attn_train_cb, # Added flash_attn_train_cb
                                  multimodal_train_cb, expert_sys_train_cb,
                                  train_file_up, val_file_up, vocab_file_up_std_train, img_dir_train_tb,
                                  batch_size_train_sl, lr_train_sl, epochs_train_sl, eval_epochs_train_sl,
                                  chkpt_steps_sl, iter_chkpt_steps_sl, gpu_select_train_cbg, dist_train_cb,
                                  gpu_mem_frac_sl, output_dir_train_tb, wandb_train_cb, wandb_proj_train_tb],
                                 [train_status_tb])


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
                    else:
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
                        "model_config": {},
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
                m_s, attn_t, tokenizer_name_hf, seed_f_obj,
                iterations, tasks_per_iter, checkpoint_interval,
                task_types_list, abduction_w, deduction_w, induction_w,
                max_attempts, temperature, top_p_gen, seed_prob,
                timeout_exec, max_output_exec,
                clarity_w, complex_w, div_w,
                acc_w, coherence_w, relevance_w, structure_w,
                target_complex, tolerance, acc_power,
                azr_device_select_ui, log_level_ui, use_wb, wb_p, out_d
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
                    
                    sum_weights = sum(final_task_dist_weights)
                    if sum_weights > 0:
                        final_task_dist_weights = [w / sum_weights for w in final_task_dist_weights]
                    elif final_task_types:
                        equal_w = 1.0 / len(final_task_types)
                        final_task_dist_weights = [equal_w] * len(final_task_types)
                    
                    azr_model_cfg_base = create_apertis_model(model_size=m_s, attention_type_override=attn_t).config.to_dict()

                    cfg = {
                        "data": {"tokenizer_name": tokenizer_name_hf.strip()},
                        "model": azr_model_cfg_base,
                        "training": {"method": "azr", "output_dir": out_d, "device": azr_device_select_ui},
                        "azr": {
                            "num_iterations": iterations, "tasks_per_iteration": tasks_per_iter,
                            "checkpoint_interval": checkpoint_interval,
                            "log_level": log_level_ui, "log_file": "azr_training.log",
                            "python_executor": {"timeout": timeout_exec, "max_output_size": max_output_exec},
                            "task_generator": {
                                "task_types": final_task_types, "task_distribution": final_task_dist_weights,
                                "max_attempts": max_attempts, "seed_tasks_path": seed_p,
                                "seed_task_probability": seed_prob, "max_new_tokens": 100,
                                "temperature": temperature, "top_p": top_p_gen
                            },
                            "solution_generator": {"max_new_tokens": 1024, "temperature": temperature, "top_p": top_p_gen},
                            "rewards": {
                                "clarity": {"weight": clarity_w},
                                "complexity": {"weight": complex_w, "target_complexity": target_complex, "tolerance": tolerance},
                                "diversity": {"weight": div_w},
                                "accuracy": {"weight": acc_w, "partial_credit_power": acc_power},
                                "coherence": {"weight": coherence_w},
                                "relevance": {"weight": relevance_w},
                                "structure": {"weight": structure_w},
                            },
                        }
                    }

                    os.makedirs(out_d, exist_ok=True)
                    cfg_path = os.path.join(tmp_dir, "azr_config.json")
                    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                    final_cfg_path_in_output_azr = os.path.join(out_d, "azr_config_used.json")
                    shutil.copy(cfg_path, final_cfg_path_in_output_azr)

                    def _thread_azr_train(c_p_arg, t_d_arg, stop_event_arg, status_box_arg):
                        try:
                            status_box_arg.value = f"AZR Training started. Output: {out_d}. Config: {final_cfg_path_in_output_azr}\nFollow logs."
                            train_from_config(c_p_arg, stop_event_arg)
                            if stop_event_arg.is_set(): status_box_arg.value += "\nAZR Training stopped by user."
                            else: status_box_arg.value += "\nAZR Training completed."
                        except Exception as e_thread_azr:
                            logger.error(f"Error in AZR training thread: {e_thread_azr}", exc_info=True)
                            status_box_arg.value += f"\nError in AZR training thread: {e_thread_azr}"
                        finally:
                            shutil.rmtree(t_d_arg, ignore_errors=True)
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

            # Removed the first azr_start_btn.click call that used ui_start_azr_training_handler
            # It is now superseded by the call below using ui_start_azr_training_handler_updated
            azr_stop_btn.click(ui_stop_azr_training_handler, outputs=[azr_status_tb])


            # Update ui_start_azr_training_handler signature and logic
            def ui_start_azr_training_handler_updated(
                m_s, attn_t, flash_attn_azr_cb_val, # Added flash_attn_azr_cb_val
                tokenizer_name_hf, seed_f_obj,
                iterations, tasks_per_iter, checkpoint_interval,
                task_types_list, abduction_w, deduction_w, induction_w,
                max_attempts, temperature, top_p_gen, seed_prob,
                timeout_exec, max_output_exec,
                clarity_w, complex_w, div_w,
                acc_w, coherence_w, relevance_w, structure_w,
                target_complex, tolerance, acc_power,
                azr_device_select_ui, log_level_ui, use_wb, wb_p, out_d
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

                    sum_weights = sum(final_task_dist_weights)
                    if sum_weights > 0:
                        final_task_dist_weights = [w / sum_weights for w in final_task_dist_weights]
                    elif final_task_types:
                        equal_w = 1.0 / len(final_task_types)
                        final_task_dist_weights = [equal_w] * len(final_task_types)

                    effective_flash_attn_azr = flash_attn_azr_cb_val if attn_t == "standard_mha" else False
                    # Create a base config for the AZR model, then convert to dict
                    # This ensures that if create_apertis_model has defaults, they are picked up.
                    temp_model_for_config = create_apertis_model(
                        model_size=m_s,
                        attention_type_override=attn_t,
                        use_flash_attention=effective_flash_attn_azr # Pass it here
                    )
                    azr_model_cfg_base = temp_model_for_config.config.to_dict()


                    cfg = {
                        "data": {"tokenizer_name": tokenizer_name_hf.strip()},
                        "model": azr_model_cfg_base, # Use the generated dict
                        "training": {"method": "azr", "output_dir": out_d, "device": azr_device_select_ui},
                        "azr": {
                            "num_iterations": iterations, "tasks_per_iteration": tasks_per_iter,
                            "checkpoint_interval": checkpoint_interval,
                            "log_level": log_level_ui, "log_file": "azr_training.log",
                            "python_executor": {"timeout": timeout_exec, "max_output_size": max_output_exec},
                            "task_generator": {
                                "task_types": final_task_types, "task_distribution": final_task_dist_weights,
                                "max_attempts": max_attempts, "seed_tasks_path": seed_p,
                                "seed_task_probability": seed_prob, "max_new_tokens": 100,
                                "temperature": temperature, "top_p": top_p_gen
                            },
                            "solution_generator": {"max_new_tokens": 1024, "temperature": temperature, "top_p": top_p_gen},
                            "rewards": {
                                "clarity": {"weight": clarity_w},
                                "complexity": {"weight": complex_w, "target_complexity": target_complex, "tolerance": tolerance},
                                "diversity": {"weight": div_w},
                                "accuracy": {"weight": acc_w, "partial_credit_power": acc_power},
                                "coherence": {"weight": coherence_w},
                                "relevance": {"weight": relevance_w},
                                "structure": {"weight": structure_w},
                            },
                        }
                    }

                    os.makedirs(out_d, exist_ok=True)
                    cfg_path = os.path.join(tmp_dir, "azr_config.json")
                    with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                    final_cfg_path_in_output_azr = os.path.join(out_d, "azr_config_used.json")
                    shutil.copy(cfg_path, final_cfg_path_in_output_azr)

                    def _thread_azr_train(c_p_arg, t_d_arg, stop_event_arg, status_box_arg):
                        try:
                            status_box_arg.value = f"AZR Training started. Output: {out_d}. Config: {final_cfg_path_in_output_azr}\nFollow logs."
                            # Assuming azr_pipeline's train_from_config handles this structure
                            from src.training.azr_pipeline import train_from_config as azr_train_from_config
                            azr_train_from_config(c_p_arg, stop_event_arg)
                            if stop_event_arg.is_set(): status_box_arg.value += "\nAZR Training stopped by user."
                            else: status_box_arg.value += "\nAZR Training completed."
                        except Exception as e_thread_azr:
                            logger.error(f"Error in AZR training thread: {e_thread_azr}", exc_info=True)
                            status_box_arg.value += f"\nError in AZR training thread: {e_thread_azr}"
                        finally:
                            shutil.rmtree(t_d_arg, ignore_errors=True)
                            self.azr_training_thread = None

                    azr_status_tb.value = "Initializing AZR training..."
                    self.azr_training_thread = threading.Thread(target=_thread_azr_train, args=(cfg_path, tmp_dir, self.azr_training_stop_event, azr_status_tb), daemon=True)
                    self.azr_training_thread.start()
                    return f"AZR Training initiated. Output will be in '{out_d}'. Monitor console/W&B. Copied config to '{final_cfg_path_in_output_azr}'."
                except Exception as e_start_azr:
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    logger.error(f"Failed to start AZR training: {e_start_azr}", exc_info=True)
                    return f"Failed to start AZR training: {e_start_azr}"

            # Update the click handler to use the new function and pass the new checkbox value
            azr_start_btn.click(
                ui_start_azr_training_handler_updated, # use updated handler
                [
                    azr_model_size_dd, azr_attn_type_dd, azr_flash_attn_cb, # Added azr_flash_attn_cb
                    azr_tokenizer_name_tb, azr_seed_tasks_up,
                    azr_iterations_sl, azr_tasks_per_iter_sl, azr_checkpoint_interval_sl,
                    azr_task_types_cbg, azr_task_dist_abduction_sl, azr_task_dist_deduction_sl, azr_task_dist_induction_sl,
                    azr_max_attempts_sl, azr_temperature_sl, azr_top_p_sl, azr_seed_prob_sl,
                    azr_timeout_sl, azr_max_output_sl,
                    azr_clarity_weight_sl, azr_complex_weight_sl, azr_div_weight_sl,
                    azr_acc_weight_sl, azr_coherence_weight_sl, azr_relevance_weight_sl, azr_structure_weight_sl,
                    azr_target_complex_sl, azr_tolerance_sl, azr_acc_power_sl,
                    azr_gpu_select_dd, azr_log_level_dd, azr_wandb_cb, azr_wandb_proj_tb, azr_output_dir_tb
                ],
                [azr_status_tb]
            )


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