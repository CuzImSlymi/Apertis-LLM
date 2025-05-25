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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model
from src.training.pipeline import YoloStyleTrainingPipeline, ApertisDataset, get_available_gpus # train_from_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApertisInterface:
    def __init__(
        self,
        model_path: Optional[str] = None,
        vocab_file: Optional[str] = None, # This will be less used for AZR now, but kept for direct model loading
        multimodal: bool = False,
        device: Optional[str] = None,
        web: bool = False,
        port: int = 7860,
        share: bool = False,
    ):
        self.model_path = model_path
        self.vocab_file = vocab_file
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
        self.vocab: Optional[Dict[str, int]] = None # For manual vocab loading
        self.reverse_vocab: Optional[Dict[int, str]] = None # For manual vocab loading
        self.token_mapping: Optional[Dict[int, int]] = None # For manual vocab loading

        self.hf_tokenizer_chat = None # For chat if using HF tokenizer by default or loaded
        
        if model_path is not None: # Simplified initial load
            self.load_model_and_tokenizer_from_path(model_path) # Tries to load HF tokenizer too
        
        self.chat_history: List[Dict[str,str]] = []
        if web:
            self.launch_web_interface()

    def load_model_and_tokenizer_from_path(self, model_path_or_name: str, vocab_file_override: Optional[str] = None):
        """
        Loads a model and attempts to load its associated Hugging Face tokenizer.
        Falls back to vocab_file if HF tokenizer fails or if vocab_file_override is given.
        """
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.token_mapping = None
        self.hf_tokenizer_chat = None
        self.model_path = model_path_or_name # Update internal path

        try:
            logger.info(f"Attempting to load Hugging Face tokenizer from {model_path_or_name}...")
            from transformers import AutoTokenizer
            self.hf_tokenizer_chat = AutoTokenizer.from_pretrained(model_path_or_name)
            logger.info(f"Successfully loaded Hugging Face tokenizer from {model_path_or_name}")
            # If HF tokenizer loaded, its vocab size will be used by the model if loaded from same dir.
        except Exception as e:
            logger.warning(f"Could not load Hugging Face tokenizer from {model_path_or_name}: {e}. "
                           f"Will try manual vocab if provided.")
            self.hf_tokenizer_chat = None

        self.load_model(model_path_or_name) # load_model will use self.hf_tokenizer_chat.vocab_size if available

        if self.model and not self.hf_tokenizer_chat: # If model loaded but no HF tokenizer
            vocab_to_load = vocab_file_override if vocab_file_override else self.vocab_file
            if vocab_to_load:
                logger.info(f"HF tokenizer not found or failed, loading manual vocab from: {vocab_to_load}")
                self.load_vocabulary(vocab_to_load) # For manual .pt/.bin and separate vocab.json
            else:
                logger.warning("No Hugging Face tokenizer and no vocab_file provided/found for loaded model. Chat may not work correctly.")
                self._create_fallback_vocab() # Minimal vocab for basic operation

        elif self.model and self.hf_tokenizer_chat:
             # Ensure model's config matches the loaded HF tokenizer's vocab size if they came from same dir
            if hasattr(self.model.config, "vocab_size") and self.model.config.vocab_size != self.hf_tokenizer_chat.vocab_size:
                logger.warning(f"Model config vocab_size ({self.model.config.vocab_size}) != HF tokenizer vocab_size ({self.hf_tokenizer_chat.vocab_size}). "
                               "This might happen if model and tokenizer are from different sources. Model will use its own config.")
        
        self.vocab_file = vocab_file_override if vocab_file_override else self.vocab_file # Update internal path


    def load_model(self, model_path: str) -> None:
        try:
            logger.info(f"Loading model from {model_path}")
            config_to_use = None
            state_dict_path_final = None

            if os.path.isdir(model_path):
                config_json_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_json_path):
                    config_to_use = ApertisConfig.from_pretrained(config_json_path)
                    # If HF tokenizer was loaded, ensure its vocab size is respected IF model config is from same dir
                    if self.hf_tokenizer_chat and hasattr(config_to_use, "vocab_size"):
                        logger.info(f"Overriding model config vocab_size ({config_to_use.vocab_size}) "
                                    f"with HF tokenizer vocab_size ({self.hf_tokenizer_chat.vocab_size}) "
                                    f"as tokenizer was loaded from same directory/name.")
                        config_to_use.vocab_size = self.hf_tokenizer_chat.vocab_size
                        if self.hf_tokenizer_chat.pad_token_id is not None: config_to_use.pad_token_id = self.hf_tokenizer_chat.pad_token_id
                        if self.hf_tokenizer_chat.bos_token_id is not None: config_to_use.bos_token_id = self.hf_tokenizer_chat.bos_token_id
                        if self.hf_tokenizer_chat.eos_token_id is not None: config_to_use.eos_token_id = self.hf_tokenizer_chat.eos_token_id
                        if self.hf_tokenizer_chat.unk_token_id is not None: config_to_use.unk_token_id = self.hf_tokenizer_chat.unk_token_id


                pt_model_path = os.path.join(model_path, "pytorch_model.bin")
                if not os.path.exists(pt_model_path):
                    pt_model_path = os.path.join(model_path, "model.pt")
                if os.path.exists(pt_model_path):
                    state_dict_path_final = pt_model_path
                else:
                    logger.warning(f"No model file (pytorch_model.bin or model.pt) found in directory {model_path}")
            elif os.path.isfile(model_path) and (model_path.endswith(".pt") or model_path.endswith(".bin")):
                state_dict_path_final = model_path
                # Config might need to be inferred or loaded from sibling config.json
                sibling_config_path = Path(model_path).parent / "config.json"
                if os.path.exists(sibling_config_path):
                    config_to_use = ApertisConfig.from_pretrained(str(sibling_config_path))
                    if self.hf_tokenizer_chat and hasattr(config_to_use, "vocab_size"): # If tokenizer from name, override
                         config_to_use.vocab_size = self.hf_tokenizer_chat.vocab_size


            if not config_to_use and state_dict_path_final: # Infer config from state_dict if no explicit config.json
                logger.info(f"No config.json found, attempting to infer config for {state_dict_path_final}")
                config_to_use = self._infer_config_from_state_dict(state_dict_path_final)

            if config_to_use and state_dict_path_final:
                self.model = ApertisForCausalLM(config_to_use)
                self.model.load_state_dict(torch.load(state_dict_path_final, map_location=self.device, weights_only=True), strict=False)
                logger.info(f"Loaded model with config derived from {model_path}")
            elif config_to_use and not state_dict_path_final: # Only config, create model
                 self.model = ApertisForCausalLM(config_to_use)
                 logger.info(f"Created model from config at {model_path}, no state_dict loaded.")
            else:
                logger.error(f"Could not determine config or model file for {model_path}")
                self.model = None
            
            if self.model: 
                self.model.to(self.device)
                self.model.eval()
                self.multimodal = self.model.config.multimodal
            logger.info("Model loading process completed.")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            self.model = None
            logger.info("Creating default model as fallback due to loading error.")
            fallback_vocab_size = self.hf_tokenizer_chat.vocab_size if self.hf_tokenizer_chat else 32000
            self.model = create_apertis_model(model_size="small", multimodal=self.multimodal, vocab_size_override=fallback_vocab_size)
            if self.model:
                self.model.to(self.device)
                self.model.eval()

    def _infer_config_from_state_dict(self, pt_path:str) -> Optional[ApertisConfig]:
        logger.info(f"Attempting to infer model config from state_dict file: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            use_expert_system = any("experts" in k for k in state_dict.keys())
            
            vocab_s_tensor = state_dict.get("model.token_embeddings.weight", state_dict.get("lm_head.weight"))
            vocab_size = vocab_s_tensor.size(0) if vocab_s_tensor is not None else (self.hf_tokenizer_chat.vocab_size if self.hf_tokenizer_chat else 32000)
            
            hidden_s_tensor = state_dict.get("model.token_embeddings.weight")
            hidden_size = hidden_s_tensor.size(1) if hidden_s_tensor is not None else 768 # Default base
            
            layer_count = 0
            i = 0
            while any(f"model.layers.{i}.{suffix}" in state_dict for suffix in ["attention.pre_norm.weight", "feed_forward.pre_norm.weight"]):
                layer_count += 1; i += 1
            if layer_count == 0: layer_count = 12 # Default base
            
            num_attn_heads = hidden_size // 64 if hidden_size % 64 == 0 else hidden_size // (hidden_size // 12 if hidden_size // 12 > 0 else 8) # Heuristic
            if hidden_size % num_attn_heads != 0: num_attn_heads = 1

            intermediate_size = hidden_size * 4

            logger.info(f"Inferred params: vocab={vocab_size}, hidden={hidden_size}, layers={layer_count}, heads={num_attn_heads}, intermediate={intermediate_size}")

            return ApertisConfig(
                vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=layer_count,
                num_attention_heads=num_attn_heads, intermediate_size=intermediate_size,
                multimodal=self.multimodal, use_expert_system=use_expert_system,
                # Use ApertisConfig defaults for other params unless they can be inferred
            )
        except Exception as e:
            logger.error(f"Error inferring config from {pt_path}: {e}", exc_info=True)
            return None


    def load_vocabulary(self, vocab_file: str) -> None: # For manual vocab.json if HF tokenizer not used/available
        try:
            logger.info(f"Loading manual vocabulary from {vocab_file}")
            with open(vocab_file, "r", encoding="utf-8") as f: 
                vocab_data = json.load(f)
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    self.vocab = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                else: self.vocab = vocab_data
            else: raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"Manual vocabulary loaded with {len(self.vocab)} tokens")
            
            if self.model and hasattr(self.model.config, "vocab_size"):
                model_cfg_vocab_size = self.model.config.vocab_size
                loaded_vocab_actual_len = len(self.vocab)
                if model_cfg_vocab_size != loaded_vocab_actual_len:
                    logger.warning(f"Model config vocab_size ({model_cfg_vocab_size}) != actual entries in manual vocab_file ({loaded_vocab_actual_len}).")
                    self.create_token_mapping(model_cfg_vocab_size, loaded_vocab_actual_len)
                else:
                    self.token_mapping = None 
            elif not self.model:
                self.token_mapping = None
        except Exception as e:
            logger.error(f"Error loading manual vocabulary from {vocab_file}: {e}", exc_info=True)
            self._create_fallback_vocab()

    def _create_fallback_vocab(self):
        logger.info("Creating minimal manual vocabulary as fallback.")
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_mapping = None

    def create_token_mapping(self, model_cfg_vocab_size: int, loaded_vocab_actual_len: int) -> None:
        if self.vocab is None: logger.error("Cannot create token mapping: manual vocabulary not loaded."); return
        self.token_mapping = {}
        special_tokens_map = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3} 
        model_unk_id = min(special_tokens_map["<unk>"], model_cfg_vocab_size -1) if model_cfg_vocab_size > 0 else 0

        for token_str, ideal_id in special_tokens_map.items():
            if token_str in self.vocab:
                vocab_file_id = self.vocab[token_str]
                if ideal_id < model_cfg_vocab_size: self.token_mapping[vocab_file_id] = ideal_id
                elif vocab_file_id < model_cfg_vocab_size: self.token_mapping[vocab_file_id] = vocab_file_id
                else: self.token_mapping[vocab_file_id] = model_unk_id
        for vocab_file_token_id in self.vocab.values(): 
            if vocab_file_token_id not in self.token_mapping: 
                if vocab_file_token_id < model_cfg_vocab_size:
                    self.token_mapping[vocab_file_token_id] = vocab_file_token_id
                else: self.token_mapping[vocab_file_token_id] = model_unk_id
        logger.info(f"Token mapping created for manual vocab. Model expects {model_cfg_vocab_size} tokens. Vocab file has {loaded_vocab_actual_len} entries.")

    def tokenize(self, text: str) -> List[int]: # Used by chat if manual vocab
        if self.hf_tokenizer_chat: # Prioritize HF tokenizer if available
            return self.hf_tokenizer_chat.encode(text, add_special_tokens=False) # No BOS/EOS here, handled by chat logic

        if not self.vocab or not self.model : 
            logger.warning("Manual vocab or model not loaded for tokenize. Using placeholder.")
            return [3] * len(text.split()) 

        loaded_vocab_unk_id = self.vocab.get("<unk>", self.model.config.unk_token_id if self.model else 3)
        raw_token_ids = [self.vocab.get(word, loaded_vocab_unk_id) for word in text.split()]

        if self.token_mapping:
            model_unk_for_mapping = self.token_mapping.get(loaded_vocab_unk_id, self.model.config.unk_token_id)
            return [self.token_mapping.get(tid, model_unk_for_mapping) for tid in raw_token_ids]
        else: 
            model_vocab_size = self.model.config.vocab_size
            model_unk_direct = self.model.config.unk_token_id
            return [tid if tid < model_vocab_size else model_unk_direct for tid in raw_token_ids]

    def detokenize(self, token_ids: List[int]) -> str: # Used by chat if manual vocab
        if self.hf_tokenizer_chat: # Prioritize HF tokenizer
            return self.hf_tokenizer_chat.decode(token_ids, skip_special_tokens=True)

        if not self.reverse_vocab or not self.model: 
            return f"[DetokenizeError: Manual Vocab/Model missing. IDs: {token_ids[:5]}...]"
        
        words = []
        reverse_id_map = {v: k for k, v in self.token_mapping.items()} if self.token_mapping else None
        loaded_vocab_unk_id = self.vocab.get("<unk>", self.model.config.unk_token_id if self.model else 3)

        for model_token_id in token_ids:
            id_to_lookup = model_token_id
            if reverse_id_map:
                id_to_lookup = reverse_id_map.get(model_token_id, model_token_id) 
            
            word = self.reverse_vocab.get(id_to_lookup)
            
            if word is not None and word not in ["<pad>", "<bos>", "<eos>"]:
                if word == "<unk>" and id_to_lookup != loaded_vocab_unk_id: 
                    words.append(f"[UNK_MAP:{model_token_id}->{id_to_lookup}]") 
                else: words.append(word)
            elif word is None: 
                 words.append(f"[ID:{model_token_id}]")
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
            if self.hf_tokenizer_chat:
                # For HF Tokenizer, add special tokens like BOS if the model expects it.
                # Some models handle this internally in generate, others expect it in input_ids.
                # Let's assume for Apertis, BOS is good.
                # Check if tokenizer has bos_token and bos_token_id
                bos_token_id_to_use = self.hf_tokenizer_chat.bos_token_id
                if bos_token_id_to_use is None: # Fallback if tokenizer doesn't define one
                    bos_token_id_to_use = self.model.config.bos_token_id

                input_ids_list = self.hf_tokenizer_chat.encode(prompt, add_special_tokens=False)
                if not input_ids_list or input_ids_list[0] != bos_token_id_to_use:
                    input_ids_list = [bos_token_id_to_use] + input_ids_list
            else: # Manual vocab
                input_ids_list = self.tokenize(prompt) # Manual tokenize defined above
                bos_id = self.model.config.bos_token_id
                if self.token_mapping and self.vocab.get("<bos>") is not None:
                    bos_id = self.token_mapping.get(self.vocab["<bos>"], self.model.config.bos_token_id)
                if not input_ids_list or input_ids_list[0] != bos_id:
                    input_ids_list = [bos_id] + input_ids_list
            
            input_t = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)
            attention_mask_t = torch.ones_like(input_t) # Simple attention mask
            
            pixel_values_t = None
            if image_path and self.multimodal and self.model.config.multimodal:
                pixel_values_t = self.preprocess_image(image_path).to(self.device)
            elif image_path and (not self.multimodal or not self.model.config.multimodal) :
                 logger.warning("Image provided but model/interface not in multimodal mode.")

            # Determine eos_token_id for generation
            eos_token_id_for_gen = self.model.config.eos_token_id
            if self.hf_tokenizer_chat and self.hf_tokenizer_chat.eos_token_id is not None:
                eos_token_id_for_gen = self.hf_tokenizer_chat.eos_token_id
            
            pad_token_id_for_gen = self.model.config.pad_token_id
            if self.hf_tokenizer_chat and self.hf_tokenizer_chat.pad_token_id is not None:
                pad_token_id_for_gen = self.hf_tokenizer_chat.pad_token_id


            output_ids = self.model.generate(
                input_ids=input_t, attention_mask=attention_mask_t, pixel_values=pixel_values_t,
                max_new_tokens=max_length, do_sample=temperature > 0.001, 
                temperature=temperature if temperature > 0.001 else 1.0, 
                top_k=top_k, top_p=top_p, use_cache=True,
                eos_token_id=eos_token_id_for_gen,
                pad_token_id=pad_token_id_for_gen
            )
            response_ids = output_ids[0, input_t.shape[1]:].tolist() # Correct slicing for new tokens

            if self.hf_tokenizer_chat:
                return self.hf_tokenizer_chat.decode(response_ids, skip_special_tokens=True)
            else:
                return self.detokenize(response_ids) # Manual detokenize

        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def chat(
        self, message: str, image_path: Optional[str] = None, max_length: int = 100,
        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
    ) -> str:
        # Construct prompt from chat history.
        # This needs to be compatible with how the model was trained (e.g. specific roles, separators)
        # For generic chat, a simple history concatenation is a starting point.
        # If using HF tokenizer, it might have specific chat templates.
        
        full_prompt_parts = []
        for entry in self.chat_history:
            full_prompt_parts.append(f"{entry['role'].capitalize()}: {entry['content']}")
        full_prompt_parts.append(f"User: {message}")
        full_prompt_parts.append("Assistant:") # Prompt for model to complete
        
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
                
                with gr.TabItem("Training"):
                    # ... (Standard Training UI - unchanged for this fix) ...
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Config")
                            model_size_train_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            attn_type_train_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            multimodal_train_cb = gr.Checkbox(label="Multimodal")
                            expert_sys_train_cb = gr.Checkbox(label="Use Expert System")
                            
                            gr.Markdown("## Data")
                            train_file_up = gr.File(label="Train Data (JSONL)", file_types=[".jsonl"])
                            val_file_up = gr.File(label="Val Data (JSONL, optional)", file_types=[".jsonl"])
                            vocab_file_up_std_train = gr.File(label="Vocab File (JSON) - For Standard Training", file_types=[".json"]) # Renamed for clarity
                            img_dir_train_tb = gr.Textbox(label="Image Dir (for multimodal)", placeholder="/path/to/images", visible=False)
                            multimodal_train_cb.change(lambda x: gr.update(visible=x), inputs=[multimodal_train_cb], outputs=[img_dir_train_tb])

                        with gr.Column(scale=1):
                            gr.Markdown("## Training Params")
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
                            output_dir_train_tb = gr.Textbox("output_training", label="Output Dir")
                            wandb_train_cb = gr.Checkbox(label="Log to W&B")
                            wandb_proj_train_tb = gr.Textbox("apertis-training", label="W&B Project", visible=False)
                            wandb_train_cb.change(lambda x: gr.update(visible=x), inputs=wandb_train_cb, outputs=wandb_proj_train_tb)
                            start_train_btn = gr.Button("Start Training")
                            train_status_tb = gr.Textbox(label="Training Status", interactive=False, lines=10)


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
                                # available_gpus_list defined earlier for Training tab
                                azr_gpu_choices = [str(g['id']) for g in available_gpus_list]
                                azr_gpu_select_cbg = gr.CheckboxGroup(choices=azr_gpu_choices, value=[azr_gpu_choices[0]] if azr_gpu_choices else [], label="Select GPUs", visible=bool(azr_gpu_choices))
                                azr_gpu_mem_frac_sl = gr.Slider(0.1, 1.0, 0.7, step=0.05, label="GPU Memory Fraction")
                            
                            azr_output_dir_tb = gr.Textbox("output_azr", label="Output Directory")
                            azr_wandb_cb = gr.Checkbox(label="Log to W&B")
                            azr_wandb_proj_tb = gr.Textbox("apertis-azr", label="W&B Project", visible=False)
                            azr_wandb_cb.change(lambda x: gr.update(visible=x), inputs=[azr_wandb_cb], outputs=[azr_wandb_proj_tb])
                            
                            azr_start_btn = gr.Button("Start AZR Training")
                            azr_status_tb = gr.Textbox(label="AZR Training Status", interactive=False, lines=10)

                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model")
                            model_path_load_tb = gr.Textbox(self.model_path or "", label="Model Path/Name (HF or local dir)")
                            vocab_path_load_tb = gr.Textbox(self.vocab_file or "", label="Manual Vocab Path (.json, optional fallback)")
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
                # The 'chat' method constructs the full prompt from history
                response = self.chat(msg, img, max_new, temp, tk, tp) 
                # Gradio chatbot expects history as list of tuples [(user_msg, bot_msg), ...]
                # self.chat_history stores dicts. We need to convert for display.
                display_history = []
                temp_history = self.chat_history.copy() # Use a copy for display conversion
                user_turn = None
                for entry_idx in range(0, len(temp_history)):
                    if temp_history[entry_idx]["role"] == "user":
                        user_turn = f"{temp_history[entry_idx]['content']}{' (Image)' if img and entry_idx == len(temp_history)-2 else ''}" # Mark image on user turn that provided it
                    elif temp_history[entry_idx]["role"] == "assistant" and user_turn is not None:
                        display_history.append((user_turn, temp_history[entry_idx]["content"]))
                        user_turn = None
                return display_history, "" # Clear textbox

            submit_btn_chat.click(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            msg_textbox.submit(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            
            def ui_clear_chat_handler():
                self.reset_chat()
                return [], "", None # chatbot_ui, msg_textbox, img_input_chat
            clear_btn_chat.click(ui_clear_chat_handler, outputs=[chatbot_ui, msg_textbox, img_input_chat])


            def ui_load_model_handler(m_path, v_path_override):
                self.load_model_and_tokenizer_from_path(m_path, v_path_override if v_path_override else None)
                info_parts = [f"Attempted to load model: {self.model_path or 'N/A'}"]
                if self.model and hasattr(self.model.config, 'to_dict'):
                    info_parts.append("Model Config:\n" + json.dumps(self.model.config.to_dict(), indent=2))
                else:
                    info_parts.append("Failed to load model or model has no config.")
                
                if self.hf_tokenizer_chat:
                    info_parts.append(f"\nHugging Face Tokenizer: {self.hf_tokenizer_chat.name_or_path}, Vocab Size: {self.hf_tokenizer_chat.vocab_size}")
                elif self.vocab:
                    info_parts.append(f"\nManual Vocab: {len(self.vocab)} tokens.")
                    if self.token_mapping: info_parts.append("Token mapping active for manual vocab.")
                else:
                    info_parts.append("\nNo tokenizer/vocabulary loaded for chat.")
                return "\n".join(info_parts)
            load_model_btn_ui.click(ui_load_model_handler, [model_path_load_tb, vocab_path_load_tb], [model_info_load_tb])

            def ui_create_model_handler(size, attn_type, multi, expert, v_size, out_path):
                try:
                    new_model = create_apertis_model(
                        model_size=size, vocab_size_override=int(v_size),
                        multimodal=multi, use_expert_system=expert,
                        attention_type_override=attn_type
                    )
                    os.makedirs(out_path, exist_ok=True)
                    # Save in Hugging Face format
                    new_model.save_pretrained(out_path) # This saves pytorch_model.bin and config.json
                    
                    # Create a dummy tokenizer.json for HF compatibility if needed, or just vocab.json
                    # For simplicity, creating vocab.json like before.
                    # If aiming for full HF hub compatibility, a dummy tokenizer_config.json and tokenizer.json
                    # (even if just pointing to vocab.json) would be better.
                    dummy_vocab = {f"<token_{i}>": i for i in range(int(v_size))}
                    # Add basic special tokens if vocab size allows
                    default_specials = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
                    for tok, idx in default_specials.items():
                        if idx < int(v_size): dummy_vocab[tok] = idx

                    with open(os.path.join(out_path, "vocab.json"),"w",encoding="utf-8") as f:
                        json.dump(dummy_vocab, f, indent=2)
                    
                    return f"Model (pytorch_model.bin), config.json, dummy vocab.json created at {out_path}"
                except Exception as e: 
                    logger.error(f"Error creating model: {e}", exc_info=True)
                    return f"Error: {str(e)}"
            create_model_btn_ui.click(ui_create_model_handler, 
                                     [new_model_size_dd, new_attn_type_dd, new_multimodal_cb, new_expert_cb, new_vocab_size_num, new_model_out_tb], 
                                     [create_model_status_tb])

            def ui_start_training_handler( # For standard training
                m_s, attn_t, m_m, exp_s, tr_f, v_f, voc_f_std, img_d, b_s, learn_r, eps, eval_ep,
                c_steps, iter_c_steps, g_sel, d_train, g_mem_f, out_d, use_wb, wb_p):
                if not tr_f or not voc_f_std: return "Training & Vocab files required for standard training."
                tmp_dir = tempfile.mkdtemp()
                train_p = os.path.join(tmp_dir, "train.jsonl"); shutil.copy(tr_f.name, train_p)
                vocab_p_std = os.path.join(tmp_dir, "vocab.json"); shutil.copy(voc_f_std.name, vocab_p_std)
                val_p = None
                if v_f: val_p = os.path.join(tmp_dir, "val.jsonl"); shutil.copy(v_f.name, val_p)
                
                sel_gpus = [int(gid) for gid in g_sel] if g_sel else None
                dist_training_eff = d_train if sel_gpus and len(sel_gpus) > 1 else False

                cfg = {
                    "data_config": {"train_data_path":train_p, "tokenizer_path":vocab_p_std, "val_data_path":val_p, 
                                    "max_length":512, "multimodal":m_m, "image_dir":img_d if m_m else None,
                                    "image_size": 224}, # Added default image_size
                    "model_config": { 
                        "model_size_preset": m_s, "attention_type": attn_t,
                        "multimodal":m_m, "use_expert_system":exp_s,
                    },
                    "training_config": {
                        "output_dir":out_d, "batch_size":b_s, "learning_rate":learn_r, "num_epochs":eps, 
                        "warmup_steps":0, "gradient_accumulation_steps":4, "max_grad_norm": 1.0, # Added defaults
                        "eval_every_n_epochs":eval_ep, "use_wandb":use_wb, "wandb_project":wb_p if use_wb else None,
                        "wandb_run_name": None, # Added
                        "fp16":True, "device":None, # Added device=None (trainer will pick)
                        "gpu_memory_fraction":g_mem_f, 
                        "use_gradient_checkpointing":True, "dynamic_batch_sizing":True, 
                        "checkpoint_steps":c_steps, "iteration_checkpoint_steps":iter_c_steps,
                        "gpu_ids":sel_gpus, "distributed_training":dist_training_eff, "local_rank": -1 # Added
                    }
                }
                _presets = {"small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                            "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}}
                cfg["model_config"].update(_presets[m_s])
                cfg["model_config"]["attention_type"] = attn_t 
                cfg["model_config"]["multimodal"] = m_m
                cfg["model_config"]["use_expert_system"] = exp_s
                if "model_size_preset" in cfg["model_config"]: del cfg["model_config"]["model_size_preset"]

                cfg_path = os.path.join(tmp_dir, "run_cfg_standard.json"); 
                with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                out_cfg_path = os.path.join(out_d, "run_cfg_standard.json")
                shutil.copy(cfg_path, out_cfg_path)
                
                import threading
                def _thread_train(c_path, t_dir):
                    try:
                        from src.training.pipeline import train_from_config # Standard pipeline
                        train_from_config(c_path)
                    except Exception as e:
                         logger.error(f"Error in Standard training thread: {e}", exc_info=True)
                    finally: shutil.rmtree(t_dir)
                
                threading.Thread(target=_thread_train, args=(cfg_path, tmp_dir), daemon=True).start()
                return f"Standard Training started. Config: {out_cfg_path}. Output: {out_d}."
            
            start_train_btn.click(ui_start_training_handler, 
                                 [model_size_train_dd, attn_type_train_dd, multimodal_train_cb, expert_sys_train_cb,
                                  train_file_up, val_file_up, vocab_file_up_std_train, img_dir_train_tb,
                                  batch_size_train_sl, lr_train_sl, epochs_train_sl, eval_epochs_train_sl,
                                  chkpt_steps_sl, iter_chkpt_steps_sl, gpu_select_train_cbg, dist_train_cb,
                                  gpu_mem_frac_sl, output_dir_train_tb, wandb_train_cb, wandb_proj_train_tb],
                                 [train_status_tb])
            
            def ui_start_azr_training_handler(
                m_s, attn_t, m_m, exp_s, tokenizer_name_hf, seed_f, seed_prob,
                iterations, tasks_per_iter, task_types, abduction_w, deduction_w, induction_w,
                max_attempts, temperature, top_p, learn_weight, acc_weight, partial_credit,
                div_weight, history_size, complex_weight, target_complex, tolerance,
                timeout, max_output, g_sel, g_mem_f, out_d, use_wb, wb_p
            ):
                if not tokenizer_name_hf.strip():
                    return "Hugging Face Tokenizer Name is required."

                tmp_dir = tempfile.mkdtemp()
                
                seed_p = None
                if seed_f:
                    seed_p = os.path.join(tmp_dir, "seed_tasks.jsonl")
                    shutil.copy(seed_f.name, seed_p)
                
                task_dist = []
                current_task_types_selected = task_types if isinstance(task_types, list) else []
                if "abduction" in current_task_types_selected: task_dist.append(abduction_w)
                else: task_dist.append(0.0)
                if "deduction" in current_task_types_selected: task_dist.append(deduction_w)
                else: task_dist.append(0.0)
                if "induction" in current_task_types_selected: task_dist.append(induction_w)
                else: task_dist.append(0.0)
                
                total_dist_weight = sum(task_dist)
                if total_dist_weight > 0:
                    task_dist = [w/total_dist_weight for w in task_dist]
                elif current_task_types_selected : # If types selected but weights are zero, distribute equally
                    num_selected_types = len(current_task_types_selected)
                    equal_weight = 1.0 / num_selected_types if num_selected_types > 0 else 0
                    temp_task_dist = []
                    if "abduction" in current_task_types_selected: temp_task_dist.append(equal_weight)
                    else: temp_task_dist.append(0.0)
                    if "deduction" in current_task_types_selected: temp_task_dist.append(equal_weight)
                    else: temp_task_dist.append(0.0)
                    if "induction" in current_task_types_selected: temp_task_dist.append(equal_weight)
                    else: temp_task_dist.append(0.0)
                    task_dist = temp_task_dist
                else: # No types selected or all weights zero
                    task_dist = [0.3, 0.3, 0.4] # Fallback to original default if no types given meaningful weights
                    if not current_task_types_selected: current_task_types_selected = ["abduction", "deduction", "induction"]


                sel_gpus = [int(gid) for gid in g_sel] if g_sel else None
                
                # Model config part for AZR - vocab_size will be set by pipeline
                azr_model_cfg_base = {
                    "hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12,
                    "intermediate_size": 3072, "hidden_act": "silu", "max_position_embeddings": 4096,
                    "initializer_range": 0.02, "rms_norm_eps": 1e-6, "use_cache": True,
                    "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2, "unk_token_id":3, # These will be updated by HF tokenizer
                    "tie_word_embeddings": False, "rope_theta": 10000.0, 
                    "attention_bias": False, "attention_dropout": 0.0, # HF Llama uses 0.0
                    "multimodal": m_m, "use_expert_system": exp_s
                    # vocab_size will be set by azr_pipeline from the HF tokenizer
                }
                _presets = {
                    "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
                }
                azr_model_cfg_base.update(_presets[m_s])
                azr_model_cfg_base["attention_type"] = attn_t


                cfg = {
                    "data": {"tokenizer_name": tokenizer_name_hf.strip()},
                    "model": azr_model_cfg_base,
                    "training": { # Minimal training config, AZR pipeline handles its own loop
                        "method": "azr", "output_dir": out_d,
                        "device": "cuda" if torch.cuda.is_available() else "cpu", # Passed to model setup
                        "gpu_ids": sel_gpus # For model device placement if not DDP
                    },
                    "azr": {
                        "num_iterations": iterations, "tasks_per_iteration": tasks_per_iter,
                        "checkpoint_interval": 10, "checkpoint_dir": os.path.join(out_d, "azr_checkpoints"),
                        "force_accept_tasks": True, "force_accept_solutions": True,
                        "force_accept_threshold": 10, "min_valid_tasks_before_validation": 20,
                        "log_level": "INFO", # Added log level
                        "python_executor": {"timeout": timeout, "max_output_size": max_output},
                        "task_generator": {
                            "task_types": current_task_types_selected, "task_distribution": task_dist, 
                            "max_attempts": max_attempts, "seed_tasks_path": seed_p, 
                            "seed_task_probability": seed_prob,
                            "base_prompt": "Generate a challenging reasoning problem.",
                            "max_new_tokens": 512, "temperature": temperature, "top_p": top_p
                        },
                        "task_validator": {"min_length": 10, "max_length": 2000, "min_complexity": 0.1, "max_complexity": 1.0, "min_clarity": 0.3}, # Adjusted to match defaults in TaskValidator
                        "solution_generator": {"max_attempts": max_attempts, "base_prompt": "Solve the following problem step by step:", "include_task_type_hint": True, "max_new_tokens": 1024, "temperature": temperature, "top_p": top_p},
                        "solution_validator": {"min_coherence": 0.3, "min_relevance": 0.3, "min_structure": 0.2}, # Adjusted
                        "learnability_reward": {"weight": learn_weight},
                        "accuracy_reward": {"weight": acc_weight, "partial_credit": partial_credit},
                        "diversity_reward": {"weight": div_weight, "history_size": history_size},
                        "complexity_reward": {"weight": complex_weight, "target_complexity": target_complex, "tolerance": tolerance},
                        "use_wandb": use_wb, "wandb_project": wb_p if use_wb else "apertis-azr", # Moved W&B here
                    }
                }
                
                os.makedirs(out_d, exist_ok=True)
                cfg_path = os.path.join(tmp_dir, "azr_config.json")
                with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                out_cfg_path = os.path.join(out_d, "azr_config.json") # Save a copy in output
                shutil.copy(cfg_path, out_cfg_path)
                
                import threading
                def _thread_azr_train(c_path, t_dir):
                    try:
                        from src.training import train_from_config # This now points to azr_pipeline's train_from_config
                        train_from_config(c_path)
                    except Exception as e:
                        logger.error(f"Error in AZR training thread: {e}", exc_info=True)
                    finally:
                        shutil.rmtree(t_dir) # Clean up temp dir
                
                threading.Thread(target=_thread_azr_train, args=(cfg_path, tmp_dir), daemon=True).start()
                return f"AZR Training started. Config: {out_cfg_path}. Output: {out_d}."
            
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

        interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share, max_threads=20)