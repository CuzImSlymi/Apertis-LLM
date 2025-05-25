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
        vocab_file: Optional[str] = None,
        multimodal: bool = False,
        device: Optional[str] = None,
        web: bool = False,
        port: int = 7860,
        share: bool = False,
    ):
        self.model_path = model_path
        self.vocab_file = vocab_file
        self.multimodal = multimodal # Initial default, will be updated by loaded model's config
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
        
        if model_path is not None and vocab_file is not None:
            self.load_model(model_path)
            if self.model: # Only load vocab if model loaded successfully
                self.load_vocabulary(vocab_file)
        
        self.chat_history: List[Dict[str,str]] = []
        if web:
            self.launch_web_interface()
    
    def load_model(self, model_path: str) -> None:
        try:
            logger.info(f"Loading model from {model_path}")
            if os.path.isdir(model_path):
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r", encoding="utf-8") as f: 
                        config_dict = json.load(f)
                    config = ApertisConfig.from_dict(config_dict)
                    self.model = ApertisForCausalLM(config)
                    state_dict_path = os.path.join(model_path, "pytorch_model.bin") # HF convention
                    if not os.path.exists(state_dict_path): # Fallback to model.pt
                        state_dict_path = os.path.join(model_path, "model.pt")
                    if os.path.exists(state_dict_path):
                        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device, weights_only=True), strict=True)
                        logger.info(f"Loaded model with config from {config_path}")
                    else:
                        logger.warning(f"Model state dict (pytorch_model.bin or model.pt) not found in {model_path}")
                        self.model = None # Ensure model is None if loading fails
                else:
                    logger.warning(f"config.json not found in {model_path}. Attempting to load model.pt and infer.")
                    pt_path = os.path.join(model_path, "model.pt")
                    if os.path.exists(pt_path):
                        self._load_from_pt_file(pt_path)
                    else:
                        logger.error(f"model.pt not found in {model_path} either.")
                        self.model = None
            elif os.path.isfile(model_path) and (model_path.endswith(".pt") or model_path.endswith(".bin")):
                 self._load_from_pt_file(model_path)
            else:
                logger.error(f"Invalid model_path: {model_path}. Not a directory or recognized model file.")
                self.model = None
            
            if self.model: 
                self.model.to(self.device)
                self.model.eval()
                self.multimodal = self.model.config.multimodal # Update interface multimodal status
            logger.info("Model loading process completed.")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            self.model = None
            logger.info("Creating default model as fallback due to loading error.")
            self.model = create_apertis_model(model_size="small", multimodal=self.multimodal) # Use self.multimodal as last known
            if self.model:
                self.model.to(self.device)
                self.model.eval()

    def _load_from_pt_file(self, pt_path:str):
        logger.info(f"Attempting to load model from raw state_dict file: {pt_path}")
        try:
            state_dict = torch.load(pt_path, map_location="cpu", weights_only=True)
            use_expert_system = any("experts" in k for k in state_dict.keys()) # Simplified check
            
            vocab_s = state_dict.get("model.token_embeddings.weight", state_dict.get("lm_head.weight"))
            vocab_size = vocab_s.size(0) if vocab_s is not None else 32000 # Default if not found
            
            hidden_s_tensor = state_dict.get("model.token_embeddings.weight")
            hidden_size = hidden_s_tensor.size(1) if hidden_s_tensor is not None else None
            
            layer_count = 0
            if hidden_size: # Only count layers if hidden_size is known
                i = 0
                while any(f"model.layers.{i}.{suffix}" in state_dict for suffix in ["attention.pre_norm.weight", "feed_forward.pre_norm.weight"]):
                    layer_count += 1
                    i += 1
            
            model_size_name = "base" # Default
            if hidden_size and layer_count > 0:
                if hidden_size == 512 and layer_count <= 8 : model_size_name = "small"
                elif hidden_size == 768 and layer_count <= 12: model_size_name = "base"
                elif hidden_size == 1024 : model_size_name = "large"
            logger.info(f"Inferred model size: {model_size_name} (hidden={hidden_size}, layers={layer_count})")

            # Create model using inferred parameters and create_apertis_model for consistency
            self.model = create_apertis_model(
                model_size=model_size_name,
                vocab_size_override=vocab_size,
                multimodal=self.multimodal, # Use current interface setting as guess
                use_expert_system=use_expert_system,
                # If loading a raw .pt, assume it used default ssm params for its size, or standard_mha
                # The config params ssm_d_inner etc in create_apertis_model will use their defaults
                # based on model_size and hidden_size.
            )
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            if missing_keys or unexpected_keys:
                logger.warning(f"Loaded .pt with issues: Missing={missing_keys or 'None'}, Unexpected={unexpected_keys or 'None'}")
            else: logger.info(f"Successfully loaded model from .pt using inferred '{model_size_name}' config.")
        except Exception as e:
            logger.error(f"Error in _load_from_pt_file for {pt_path}: {e}", exc_info=True)
            self.model = None # Ensure model is None on failure

    def load_vocabulary(self, vocab_file: str) -> None:
        try:
            logger.info(f"Loading vocabulary from {vocab_file}")
            with open(vocab_file, "r", encoding="utf-8") as f: 
                vocab_data = json.load(f)
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    self.vocab = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                else: self.vocab = vocab_data
            else: raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            logger.info(f"Vocabulary loaded with {len(self.vocab)} tokens")
            
            if self.model and hasattr(self.model.config, "vocab_size"):
                model_cfg_vocab_size = self.model.config.vocab_size
                loaded_vocab_actual_len = len(self.vocab)
                if model_cfg_vocab_size != loaded_vocab_actual_len:
                    logger.warning(f"Model config vocab_size ({model_cfg_vocab_size}) != actual entries in vocab_file ({loaded_vocab_actual_len}).")
                    self.create_token_mapping(model_cfg_vocab_size, loaded_vocab_actual_len)
                else:
                    self.token_mapping = None 
                    logger.info("Model vocab_size matches vocab_file entries. No token mapping needed.")
            elif not self.model:
                logger.warning("Model not loaded, cannot compare vocab sizes. Assuming vocab file is authoritative.")
                self.token_mapping = None
        except Exception as e:
            logger.error(f"Error loading vocabulary from {vocab_file}: {e}", exc_info=True)
            self._create_fallback_vocab()

    def _create_fallback_vocab(self):
        logger.info("Creating minimal vocabulary as fallback.")
        self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        self.token_mapping = None

    def create_token_mapping(self, model_cfg_vocab_size: int, loaded_vocab_actual_len: int) -> None:
        if self.vocab is None: logger.error("Cannot create token mapping: vocabulary not loaded."); return
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
        logger.info(f"Token mapping created. Model expects {model_cfg_vocab_size} tokens. Vocab file has {loaded_vocab_actual_len} entries.")

    def tokenize(self, text: str) -> List[int]:
        if not self.vocab or not self.model : 
            logger.warning("Vocab or model not loaded. Using placeholder tokenization.")
            return [3] * len(text.split()) # Fallback UNK id

        loaded_vocab_unk_id = self.vocab.get("<unk>", 3)
        raw_token_ids = [self.vocab.get(word, loaded_vocab_unk_id) for word in text.split()]

        if self.token_mapping:
            model_unk_for_mapping = self.token_mapping.get(loaded_vocab_unk_id, self.model.config.unk_token_id)
            return [self.token_mapping.get(tid, model_unk_for_mapping) for tid in raw_token_ids]
        else: # No mapping implies model_vocab_size == len(vocab) and IDs should be fine
            model_vocab_size = self.model.config.vocab_size
            model_unk_direct = self.model.config.unk_token_id
            return [tid if tid < model_vocab_size else model_unk_direct for tid in raw_token_ids]

    def detokenize(self, token_ids: List[int]) -> str:
        if not self.reverse_vocab or not self.model: 
            return f"[DetokenizeError: Vocab/Model missing. IDs: {token_ids[:5]}...]"
        
        words = []
        reverse_id_map = {v: k for k, v in self.token_mapping.items()} if self.token_mapping else None
        loaded_vocab_unk_id = self.vocab.get("<unk>", 3) # ID of <unk> in the loaded vocab file

        for model_token_id in token_ids:
            id_to_lookup = model_token_id
            if reverse_id_map:
                id_to_lookup = reverse_id_map.get(model_token_id, model_token_id) # Fallback to model_id if not in map
            
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
            transform = transforms.Compose([
                transforms.Resize((self.model.config.image_size, self.model.config.image_size)), # Use model's config
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            image = Image.open(image_path).convert("RGB")
            return transform(image).unsqueeze(0) 
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {e}")
            return torch.zeros(1, 3, self.model.config.image_size if self.model else 224, self.model.config.image_size if self.model else 224)

    def generate_response(
        self, prompt: str, image_path: Optional[str] = None, max_length: int = 100, # max_length here is max_new_tokens
        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
    ) -> str:
        if not self.model: return "Model not loaded."
        if not self.vocab: return "Vocabulary not loaded."
        try:
            input_ids_list = self.tokenize(prompt)
            bos_id = self.model.config.bos_token_id
            if self.token_mapping and self.vocab.get("<bos>") is not None: # If <bos> in vocab, map its ID
                bos_id = self.token_mapping.get(self.vocab["<bos>"], self.model.config.bos_token_id)

            if not input_ids_list or input_ids_list[0] != bos_id:
                input_ids_list = [bos_id] + input_ids_list
            
            input_t = torch.tensor([input_ids_list], dtype=torch.long).to(self.device)
            attention_mask_t = torch.ones_like(input_t)
            
            pixel_values_t = None
            if image_path and self.multimodal and self.model.config.multimodal:
                pixel_values_t = self.preprocess_image(image_path).to(self.device)
            elif image_path and (not self.multimodal or not self.model.config.multimodal) :
                 logger.warning("Image provided but model/interface not in multimodal mode.")

            output_ids = self.model.generate(
                input_ids=input_t, attention_mask=attention_mask_t, pixel_values=pixel_values_t,
                max_new_tokens=max_length, do_sample=temperature > 0.001, 
                temperature=temperature if temperature > 0.001 else 1.0, 
                top_k=top_k, top_p=top_p, use_cache=True,
                eos_token_id=self.model.config.eos_token_id, # Use model's configured eos
                pad_token_id=self.model.config.pad_token_id  # Use model's configured pad
            )
            response_ids = output_ids[0, len(input_ids_list):].tolist()
            return self.detokenize(response_ids)
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    def chat(
        self, message: str, image_path: Optional[str] = None, max_length: int = 100,
        temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
    ) -> str:
        self.chat_history.append({"role": "user", "content": message})
        prompt = "".join([f"{entry['role'].capitalize()}: {entry['content']}\n" for entry in self.chat_history]) + "Assistant: "
        response = self.generate_response(prompt, image_path, max_length, temperature, top_k, top_p)
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
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Config")
                            model_size_train_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            attn_type_train_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type") # New
                            multimodal_train_cb = gr.Checkbox(label="Multimodal")
                            expert_sys_train_cb = gr.Checkbox(label="Use Expert System")
                            
                            gr.Markdown("## Data")
                            train_file_up = gr.File(label="Train Data (JSONL)", file_types=[".jsonl"])
                            val_file_up = gr.File(label="Val Data (JSONL, optional)", file_types=[".jsonl"])
                            vocab_file_up = gr.File(label="Vocab File (JSON)", file_types=[".json"])
                            img_dir_train_tb = gr.Textbox(label="Image Dir (for multimodal)", placeholder="/path/to/images", visible=False)
                            multimodal_train_cb.change(lambda x: gr.update(visible=x), inputs=[multimodal_train_cb], outputs=[img_dir_train_tb])

                        with gr.Column(scale=1):
                            gr.Markdown("## Training Params")
                            batch_size_train_sl = gr.Slider(1, 64, 4, step=1, label="Batch Size")
                            lr_train_sl = gr.Slider(1e-6, 1e-3, 5e-5, step=1e-6, label="Learning Rate") # Removed format
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

                # New AZR Tab
                with gr.TabItem("Absolute Zero Reasoner"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Absolute Zero Reasoner Training")
                            gr.Markdown("""
                            Train your model using the Absolute Zero Reasoner method, which enables 
                            language models to improve reasoning capabilities through self-play without 
                            requiring any external training data.
                            """)
                            
                            gr.Markdown("## Model Config")
                            azr_model_size_dd = gr.Dropdown(["small", "base", "large"], value="base", label="Base Model Size")
                            azr_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type")
                            azr_multimodal_cb = gr.Checkbox(label="Multimodal")
                            azr_expert_sys_cb = gr.Checkbox(label="Use Expert System")
                            
                            gr.Markdown("## Vocabulary")
                            azr_vocab_file_up = gr.File(label="Vocab File (JSON)", file_types=[".json"])
                            
                            gr.Markdown("## Seed Data (Optional)")
                            azr_seed_tasks_up = gr.File(label="Seed Tasks (JSONL, optional)", file_types=[".jsonl"])
                            azr_seed_prob_sl = gr.Slider(0.0, 1.0, 0.2, step=0.05, label="Seed Task Probability")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("## AZR Training Parameters")
                            azr_iterations_sl = gr.Slider(10, 500, 100, step=10, label="Number of Iterations")
                            azr_tasks_per_iter_sl = gr.Slider(1, 20, 5, step=1, label="Tasks Per Iteration")
                            
                            with gr.Accordion("Task Generation", open=False):
                                azr_task_types = gr.CheckboxGroup(
                                    ["abduction", "deduction", "induction"], 
                                    value=["abduction", "deduction", "induction"],
                                    label="Task Types"
                                )
                                azr_task_dist_abduction = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Abduction Weight")
                                azr_task_dist_deduction = gr.Slider(0.0, 1.0, 0.3, step=0.05, label="Deduction Weight")
                                azr_task_dist_induction = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Induction Weight")
                                azr_max_attempts_sl = gr.Slider(1, 10, 3, step=1, label="Max Generation Attempts")
                                azr_temperature_sl = gr.Slider(0.1, 1.5, 0.7, step=0.05, label="Generation Temperature")
                                azr_top_p_sl = gr.Slider(0.1, 1.0, 0.9, step=0.05, label="Top-P")
                            
                            with gr.Accordion("Rewards", open=False):
                                gr.Markdown("### Learnability Reward")
                                azr_learn_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight")
                                
                                gr.Markdown("### Accuracy Reward")
                                azr_acc_weight_sl = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight")
                                azr_partial_credit_cb = gr.Checkbox(value=True, label="Allow Partial Credit")
                                
                                gr.Markdown("### Diversity Reward")
                                azr_div_weight_sl = gr.Slider(0.0, 2.0, 0.5, step=0.1, label="Weight")
                                azr_history_size_sl = gr.Slider(1, 50, 10, step=1, label="History Size")
                                
                                gr.Markdown("### Complexity Reward")
                                azr_complex_weight_sl = gr.Slider(0.0, 2.0, 0.3, step=0.1, label="Weight")
                                azr_target_complex_sl = gr.Slider(0.0, 1.0, 0.7, step=0.05, label="Target Complexity")
                                azr_tolerance_sl = gr.Slider(0.0, 0.5, 0.2, step=0.05, label="Tolerance")
                            
                            with gr.Accordion("Python Executor", open=False):
                                azr_timeout_sl = gr.Slider(1, 30, 5, step=1, label="Execution Timeout (seconds)")
                                azr_max_output_sl = gr.Slider(1000, 50000, 10000, step=1000, label="Max Output Size")
                            
                            with gr.Accordion("GPU", open=False):
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
                            model_path_load_tb = gr.Textbox(self.model_path or "", label="Model Path (.pt or dir)")
                            vocab_path_load_tb = gr.Textbox(self.vocab_file or "", label="Vocab Path (.json)")
                            with gr.Accordion("Advanced Load Options", open=False):
                                adapt_vocab_cb = gr.Checkbox(True, label="Auto-adapt Vocab Mismatches")
                            load_model_btn_ui = gr.Button("Load Model")
                            model_info_load_tb = gr.Textbox(label="Loaded Model Info", interactive=False, lines=10)
                        with gr.Column(scale=1):
                            gr.Markdown("## Create New Model")
                            new_model_size_dd = gr.Dropdown(["small","base","large"], value="base", label="Model Size")
                            new_attn_type_dd = gr.Dropdown(["selective_ssm", "standard_mha"], value="standard_mha", label="Attention Type") # New
                            new_multimodal_cb = gr.Checkbox(label="Multimodal")
                            new_expert_cb = gr.Checkbox(label="Use Expert System")
                            new_vocab_size_num = gr.Number(32000, label="Vocab Size", precision=0)
                            new_model_out_tb = gr.Textbox("models/new_model", label="Save Path")
                            create_model_btn_ui = gr.Button("Create Model")
                            create_model_status_tb = gr.Textbox(label="Creation Status", interactive=False, lines=3)
            
            def ui_chat_handler(msg, img, max_new, temp, tk, tp, hist):
                if not msg.strip() and not img: return hist, ""
                hist = hist + [(f"{msg}{' (Image)' if img else ''}", None)]
                response = self.chat(msg, img, max_new, temp, tk, tp)
                hist[-1] = (hist[-1][0], response)
                return hist, ""
            submit_btn_chat.click(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            msg_textbox.submit(ui_chat_handler, [msg_textbox, img_input_chat, max_new_tokens_slider, temp_slider_chat, top_k_slider_chat, top_p_slider_chat, chatbot_ui], [chatbot_ui, msg_textbox])
            clear_btn_chat.click(lambda: (self.reset_chat(), [], "", None)[1:], outputs=[chatbot_ui, msg_textbox, img_input_chat])

            def ui_load_model_handler(m_path, v_path, adapt_v_ui):
                self.model, self.vocab, self.reverse_vocab, self.token_mapping = None, None, None, None
                self.model_path, self.vocab_file = m_path, v_path
                self.load_model(m_path)
                if self.model: 
                    if not adapt_v_ui and hasattr(self, 'create_token_mapping'): # Temp disable mapping if user unchecks
                        original_create_mapping_func = self.create_token_mapping
                        self.create_token_mapping = lambda mv_size, lv_size: None 
                        self.load_vocabulary(v_path)
                        self.create_token_mapping = original_create_mapping_func
                    else:
                        self.load_vocabulary(v_path)
                info = "Model loading attempted.\n"
                if self.model and hasattr(self.model.config, 'to_dict'):
                    info += "Config:\n" + json.dumps(self.model.config.to_dict(), indent=2)
                    info += f"\n\nVocab: {len(self.vocab) if self.vocab else 'None'} tokens."
                    if self.token_mapping: info += "\nToken mapping active."
                else: info += "Failed to load model or model has no config."
                return info
            load_model_btn_ui.click(ui_load_model_handler, [model_path_load_tb, vocab_path_load_tb, adapt_vocab_cb], [model_info_load_tb])

            def ui_create_model_handler(size, attn_type, multi, expert, v_size, out_path):
                try:
                    # Use create_apertis_model which now respects attention_type_override
                    new_model = create_apertis_model(
                        model_size=size, vocab_size_override=int(v_size),
                        multimodal=multi, use_expert_system=expert,
                        attention_type_override=attn_type # Pass the selected attention type
                    )
                    os.makedirs(out_path, exist_ok=True)
                    torch.save(new_model.state_dict(), os.path.join(out_path, "pytorch_model.bin"))
                    new_model.config.save_pretrained(out_path)
                    dummy_vocab = {f"<token_{i}>": i for i in range(int(v_size))}
                    if int(v_size) >= 4:
                         dummy_vocab["<pad>"],dummy_vocab["<bos>"],dummy_vocab["<eos>"],dummy_vocab["<unk>"] = 0,1,2,3
                    with open(os.path.join(out_path, "vocab.json"),"w",encoding="utf-8") as f: json.dump(dummy_vocab,f,indent=2)
                    return f"Model, config.json, dummy vocab.json created at {out_path}"
                except Exception as e: return f"Error: {str(e)}"
            create_model_btn_ui.click(ui_create_model_handler, 
                                     [new_model_size_dd, new_attn_type_dd, new_multimodal_cb, new_expert_cb, new_vocab_size_num, new_model_out_tb], 
                                     [create_model_status_tb])

            def ui_start_training_handler(
                m_s, attn_t, m_m, exp_s, tr_f, v_f, voc_f, img_d, b_s, learn_r, eps, eval_ep,
                c_steps, iter_c_steps, g_sel, d_train, g_mem_f, out_d, use_wb, wb_p):
                if not tr_f or not voc_f: return "Training & Vocab files required."
                tmp_dir = tempfile.mkdtemp()
                train_p = os.path.join(tmp_dir, "train.jsonl"); shutil.copy(tr_f.name, train_p)
                vocab_p = os.path.join(tmp_dir, "vocab.json"); shutil.copy(voc_f.name, vocab_p)
                val_p = None
                if v_f: val_p = os.path.join(tmp_dir, "val.jsonl"); shutil.copy(v_f.name, val_p)
                
                sel_gpus = [int(gid) for gid in g_sel] if g_sel else None
                dist_training_eff = d_train if sel_gpus and len(sel_gpus) > 1 else False

                cfg = {
                    "data_config": {"train_data_path":train_p, "tokenizer_path":vocab_p, "val_data_path":val_p, 
                                    "max_length":512, "multimodal":m_m, "image_dir":img_d if m_m else None},
                    "model_config": { # create_apertis_model will fill this based on size and overrides
                        "model_size_preset": m_s, # Not a direct ApertisConfig param, used by helper
                        "attention_type": attn_t, # Pass explicitly
                        "multimodal":m_m, "use_expert_system":exp_s,
                        # vocab_size determined by pipeline from vocab_p
                    },
                    "training_config": {
                        "output_dir":out_d, "batch_size":b_s, "learning_rate":learn_r, "num_epochs":eps, 
                        "eval_every_n_epochs":eval_ep, "use_wandb":use_wb, "wandb_project":wb_p if use_wb else None,
                        "gradient_accumulation_steps":4, "fp16":True, "gpu_memory_fraction":g_mem_f, 
                        "use_gradient_checkpointing":True, "dynamic_batch_sizing":True, 
                        "checkpoint_steps":c_steps, "iteration_checkpoint_steps":iter_c_steps,
                        "gpu_ids":sel_gpus, "distributed_training":dist_training_eff,
                    }
                }
                # Explicitly set model config from presets, allowing overrides
                _presets = {"small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                            "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                            "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}}
                cfg["model_config"].update(_presets[m_s]) # Apply preset
                cfg["model_config"]["attention_type"] = attn_t # Ensure override
                cfg["model_config"]["multimodal"] = m_m
                cfg["model_config"]["use_expert_system"] = exp_s
                del cfg["model_config"]["model_size_preset"] # Remove temp key


                cfg_path = os.path.join(tmp_dir, "run_cfg.json"); 
                with open(cfg_path, "w") as f: json.dump(cfg, f, indent=2)
                
                import threading
                def _thread_train(c_path, t_dir):
                    try:
                        from src.training.pipeline import train_from_config # Re-import in thread if needed
                        train_from_config(c_path)
                    finally: shutil.rmtree(t_dir)
                
                threading.Thread(target=_thread_train, args=(cfg_path, tmp_dir), daemon=True).start()
                return f"Training started. Config in {cfg_path}. Logs in {out_d}."
            
            start_train_btn.click(ui_start_training_handler, 
                                 [model_size_train_dd, attn_type_train_dd, multimodal_train_cb, expert_sys_train_cb,
                                  train_file_up, val_file_up, vocab_file_up, img_dir_train_tb,
                                  batch_size_train_sl, lr_train_sl, epochs_train_sl, eval_epochs_train_sl,
                                  chkpt_steps_sl, iter_chkpt_steps_sl, gpu_select_train_cbg, dist_train_cb,
                                  gpu_mem_frac_sl, output_dir_train_tb, wandb_train_cb, wandb_proj_train_tb],
                                 [train_status_tb])
            
            # AZR Training Handler
            def ui_start_azr_training_handler(
                m_s, attn_t, m_m, exp_s, voc_f, seed_f, seed_prob,
                iterations, tasks_per_iter, task_types, abduction_w, deduction_w, induction_w,
                max_attempts, temperature, top_p, learn_weight, acc_weight, partial_credit,
                div_weight, history_size, complex_weight, target_complex, tolerance,
                timeout, max_output, g_sel, g_mem_f, out_d, use_wb, wb_p
            ):
                if not voc_f: return "Vocabulary file is required."
                
                # Create temporary directory for files
                tmp_dir = tempfile.mkdtemp()
                vocab_p = os.path.join(tmp_dir, "vocab.json")
                shutil.copy(voc_f.name, vocab_p)
                
                # Handle seed file if provided
                seed_p = None
                if seed_f:
                    seed_p = os.path.join(tmp_dir, "seed_tasks.jsonl")
                    shutil.copy(seed_f.name, seed_p)
                
                # Normalize task distribution weights
                task_dist = []
                if "abduction" in task_types:
                    task_dist.append(abduction_w)
                else:
                    task_dist.append(0.0)
                    
                if "deduction" in task_types:
                    task_dist.append(deduction_w)
                else:
                    task_dist.append(0.0)
                    
                if "induction" in task_types:
                    task_dist.append(induction_w)
                else:
                    task_dist.append(0.0)
                
                # Normalize to sum to 1.0
                total = sum(task_dist)
                if total > 0:
                    task_dist = [w/total for w in task_dist]
                else:
                    task_dist = [1/3, 1/3, 1/3]  # Default equal distribution
                
                # Selected GPUs
                sel_gpus = [int(gid) for gid in g_sel] if g_sel else None
                
                # Create configuration
                cfg = {
                    "model": {
                        "vocab_size": 32000,  # Will be determined from vocab file
                        "hidden_size": 768,   # Will be updated based on model size
                        "num_hidden_layers": 12,
                        "num_attention_heads": 12,
                        "intermediate_size": 3072,
                        "hidden_act": "silu",
                        "max_position_embeddings": 4096,
                        "initializer_range": 0.02,
                        "rms_norm_eps": 1e-6,
                        "use_cache": True,
                        "pad_token_id": 0,
                        "bos_token_id": 1,
                        "eos_token_id": 2,
                        "tie_word_embeddings": False,
                        "rope_theta": 10000.0,
                        "attention_bias": False,
                        "attention_dropout": 0.0,
                        "multimodal": m_m,
                        "use_expert_system": exp_s
                    },
                    "data": {
                        "tokenizer_path": vocab_p
                    },
                    "training": {
                        "method": "azr",
                        "output_dir": out_d,
                        "batch_size": 1,
                        "learning_rate": 5e-5,
                        "weight_decay": 0.01,
                        "use_wandb": use_wb,
                        "wandb_project": wb_p if use_wb else "apertis-azr",
                        "fp16": True,
                        "device": "cuda" if torch.cuda.is_available() else "cpu",
                        "gpu_memory_fraction": g_mem_f,
                        "use_gradient_checkpointing": True,
                        "gpu_ids": sel_gpus
                    },
                    "azr": {
                        "num_iterations": iterations,
                        "tasks_per_iteration": tasks_per_iter,
                        "checkpoint_interval": 10,
                        "python_executor": {
                            "timeout": timeout,
                            "max_output_size": max_output
                        },
                        "task_generator": {
                            "task_types": task_types,
                            "task_distribution": task_dist,
                            "max_attempts": max_attempts,
                            "seed_tasks_path": seed_p,
                            "seed_task_probability": seed_prob,
                            "base_prompt": "Generate a challenging reasoning problem.",
                            "max_new_tokens": 512,
                            "temperature": temperature,
                            "top_p": top_p
                        },
                        "task_validator": {
                            "min_length": 20,
                            "max_length": 1000,
                            "min_complexity": 0.3,
                            "max_complexity": 0.9,
                            "min_clarity": 0.5
                        },
                        "solution_generator": {
                            "max_attempts": max_attempts,
                            "base_prompt": "Solve the following problem step by step:",
                            "include_task_type_hint": True,
                            "max_new_tokens": 1024,
                            "temperature": temperature,
                            "top_p": top_p
                        },
                        "solution_validator": {
                            "min_coherence": 0.5,
                            "min_relevance": 0.6,
                            "min_structure": 0.4,
                            "min_output_similarity": 0.8
                        },
                        "learnability_reward": {
                            "weight": learn_weight,
                            "min_threshold": 0.0,
                            "max_threshold": 1.0
                        },
                        "accuracy_reward": {
                            "weight": acc_weight,
                            "partial_credit": partial_credit
                        },
                        "diversity_reward": {
                            "weight": div_weight,
                            "history_size": history_size
                        },
                        "complexity_reward": {
                            "weight": complex_weight,
                            "target_complexity": target_complex,
                            "tolerance": tolerance
                        },
                        "reward_calculator": {},
                        "tracker": {
                            "save_tasks": True,
                            "save_solutions": True
                        }
                    }
                }
                
                # Update model config based on selected size
                _presets = {
                    "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096}
                }
                cfg["model"].update(_presets[m_s])
                
                # Save configuration
                os.makedirs(out_d, exist_ok=True)
                cfg_path = os.path.join(tmp_dir, "azr_config.json")
                with open(cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                
                # Copy config to output dir for reference
                out_cfg_path = os.path.join(out_d, "azr_config.json")
                with open(out_cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                
                # Start training in a separate thread
                import threading
                def _thread_azr_train(c_path, t_dir):
                    try:
                        from src.training import train_from_config
                        train_from_config(c_path)
                    except Exception as e:
                        logger.error(f"Error in AZR training: {e}", exc_info=True)
                    finally:
                        shutil.rmtree(t_dir)
                
                threading.Thread(target=_thread_azr_train, args=(cfg_path, tmp_dir), daemon=True).start()
                return f"AZR Training started. Configuration saved to {out_cfg_path}. Logs and outputs will be in {out_d}."
            
            # Connect AZR training button
            azr_start_btn.click(
                ui_start_azr_training_handler,
                [
                    azr_model_size_dd, azr_attn_type_dd, azr_multimodal_cb, azr_expert_sys_cb,
                    azr_vocab_file_up, azr_seed_tasks_up, azr_seed_prob_sl,
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

        interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share)
