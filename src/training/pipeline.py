import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import logging
import json
import wandb
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
import torchvision.transforms as transforms

import sys
import os
import math
import threading
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def _load_vocabulary_and_get_size(tokenizer_path: str) -> Tuple[Dict[str, int], int]:
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        actual_vocab_dict = {}
        effective_vocab_size = 0 # This will be max_id + 1

        if isinstance(vocab_data, dict):
            if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                token_list = vocab_data["tokens"]
                actual_vocab_dict = {token: idx for idx, token in enumerate(token_list)}
                if actual_vocab_dict:
                    max_id_found = max(actual_vocab_dict.values())
                    effective_vocab_size = max_id_found + 1
                else: # Empty list
                    effective_vocab_size = 0
            else: # Assume it's a token:id dict
                actual_vocab_dict = vocab_data
                if not actual_vocab_dict:
                    return {}, 0
                
                max_id_found = -1
                seen_ids = set()
                for token, token_id in actual_vocab_dict.items():
                    if not isinstance(token_id, int) or token_id < 0:
                        raise ValueError(f"Invalid token ID '{token_id}' for token '{token}' in vocabulary file.")
                    if token_id in seen_ids:
                        raise ValueError(f"Duplicate token ID '{token_id}' found in vocabulary file.")
                    seen_ids.add(token_id)
                    if token_id > max_id_found:
                        max_id_found = token_id
                effective_vocab_size = max_id_found + 1
        else:
            raise ValueError(f"Unsupported vocabulary format in {tokenizer_path}: {type(vocab_data)}")

        return actual_vocab_dict, effective_vocab_size

    except Exception as e:
        logger.error(f"Error loading or processing vocabulary from {tokenizer_path}: {e}")
        raise

class ApertisPretrainDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        vocab_dict: Dict[str, int],
        model_config_vocab_size: int,
        max_length: int = 512,
        multimodal: bool = False,
        image_dir: Optional[str] = None,
        image_size: int = 224,
        pad_token_id_from_config: int = 0,
        unk_token_id_from_config: int = 3,
    ):
        self.data_path = data_path
        self.vocab = vocab_dict
        self.model_config_vocab_size = model_config_vocab_size # The vocab size the model is configured with
        self.max_length = max_length
        self.multimodal = multimodal
        self.image_dir = image_dir
        self.image_size = image_size
        self.data = self._load_data()

        self.pad_token_id_for_data = pad_token_id_from_config
        self.unk_token_id_for_data = unk_token_id_from_config

        if self.multimodal:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_data(self) -> List[Dict]:
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if "text" not in item:
                            logger.warning(f"Skipping line {i+1} in {self.data_path}: 'text' field missing.")
                            continue
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping line {i+1} in {self.data_path} due to JSON decode error: {e}")
                        pass
        except FileNotFoundError:
             logger.error(f"Data file not found: {self.data_path}")
             raise
        return data

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        
        # Use the UNK token ID that the model config expects
        unk_id_to_use = self.unk_token_id_for_data

        current_tokens = []
        if isinstance(text, str):
            current_tokens = text.split()
        elif isinstance(text, list): # Support pre-tokenized text if it's a list of strings/ints
            current_tokens = text
        else:
            logger.warning(f"Unexpected text type for tokenization: {type(text)}. Treating as empty.")

        for word_or_token in current_tokens:
            token_id_from_vocab_file = -1
            if isinstance(word_or_token, int): # If input is already token IDs (from vocab file's perspective)
                token_id_from_vocab_file = word_or_token
            else: # Input is string, lookup in vocab_file
                token_id_from_vocab_file = self.vocab.get(str(word_or_token), self.vocab.get("<unk>", unk_id_to_use))

            # Ensure the token_id is valid for the model's configured vocab_size
            if token_id_from_vocab_file >= self.model_config_vocab_size:
                final_token_id = unk_id_to_use
            else:
                final_token_id = token_id_from_vocab_file
            tokens.append(final_token_id)
        return tokens

    def _load_image(self, image_path: str) -> torch.Tensor:
        if self.image_dir is None: full_path = image_path
        else: full_path = os.path.join(self.image_dir, image_path)
        try:
            image = Image.open(full_path).convert('RGB')
            return self.image_transform(image)
        except FileNotFoundError:
             logger.warning(f"Image not found at {full_path}, returning blank image.")
             blank = torch.zeros(3, self.image_size, self.image_size)
             return blank
        except Exception as e:
            logger.warning(f"Error loading image {full_path}: {e}. Returning blank image.")
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item.get("text", "")

        input_ids = self._tokenize(input_text)

        pad_id_to_use = self.pad_token_id_for_data

        seq_len = len(input_ids)
        if seq_len > self.max_length: input_ids = input_ids[:self.max_length]
        elif seq_len < self.max_length: input_ids.extend([pad_id_to_use] * (self.max_length - seq_len))

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != pad_id_to_use).long()

        labels = input_ids_tensor.clone()

        output = {"input_ids": input_ids_tensor, "attention_mask": attention_mask, "labels": labels}

        if self.multimodal and "image" in item:
             if self.image_dir is None and not os.path.isabs(item["image"]):
                  logger.warning(f"Image path '{item['image']}' is relative but image_dir is not set. Assuming path is relative to data file or CWD.")
             output["pixel_values"] = self._load_image(item["image"])
        return output

class ApertisFineTuneDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any, # HF Tokenizer object or manual vocab_dict
        max_length: int = 512,
        prompt_template: str = "User: {instruction}\nAssistant: {output}",
        is_hf_tokenizer: bool = False,
        # These are only used if not is_hf_tokenizer
        model_config_vocab_size: Optional[int] = None,
        model_config_eos_token_id: Optional[int] = None,
        model_config_pad_token_id: Optional[int] = None,
        model_config_unk_token_id: Optional[int] = None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer 
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.data = self._load_data()
        self.is_hf_tokenizer = is_hf_tokenizer

        if self.is_hf_tokenizer:
            if not hasattr(self.tokenizer, 'encode') or not hasattr(self.tokenizer, 'pad_token_id') or not hasattr(self.tokenizer, 'eos_token_id'):
                raise ValueError("Provided HF tokenizer object is missing required attributes (encode, pad_token_id, eos_token_id).")
            self.pad_token_id = self.tokenizer.pad_token_id
            self.eos_token_id = self.tokenizer.eos_token_id

            if self.pad_token_id is None and self.eos_token_id is not None:
                logger.warning(f"HF tokenizer pad_token_id is None, using its eos_token_id ({self.eos_token_id}) as pad_token_id.")
                self.pad_token_id = self.eos_token_id
            elif self.pad_token_id is None and self.eos_token_id is None:
                raise ValueError("HF tokenizer has neither pad_token_id nor eos_token_id set. Cannot proceed with fine-tuning dataset.")
            if self.eos_token_id is None:
                 raise ValueError("HF tokenizer eos_token_id is None. This is required for generative fine-tuning.")
        else: # Manual vocab
            if not (isinstance(self.tokenizer, dict) and model_config_vocab_size is not None and 
                    model_config_eos_token_id is not None and model_config_pad_token_id is not None and
                    model_config_unk_token_id is not None):
                raise ValueError("For manual vocab fine-tuning, tokenizer (vocab_dict) and model_config_vocab_size/eos/pad/unk IDs must be provided.")
            self.vocab = self.tokenizer # tokenizer is the vocab_dict
            self.model_config_vocab_size = model_config_vocab_size
            self.eos_token_id = model_config_eos_token_id
            self.pad_token_id = model_config_pad_token_id
            self.unk_token_id = model_config_unk_token_id

    def _load_data(self) -> List[Dict]:
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if "instruction" not in item or "output" not in item:
                            logger.warning(f"Skipping line {i+1} in {self.data_path}: 'instruction' or 'output' field missing.")
                            continue
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping line {i+1} in {self.data_path} due to JSON decode error: {e}")
                        pass
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        return data

    def _manual_tokenize(self, text: str) -> List[int]:
        if self.is_hf_tokenizer:
            raise RuntimeError("_manual_tokenize called with HF tokenizer setup.")
        if not isinstance(self.vocab, dict):
             raise RuntimeError("_manual_tokenize called but vocab (self.tokenizer) is not a dict.")

        tokens = []
        unk_id_to_use = self.unk_token_id # Use the one from model config

        for word in text.split():
            token_id_from_vocab_file = self.vocab.get(word, self.vocab.get("<unk>", unk_id_to_use))
            
            if token_id_from_vocab_file >= self.model_config_vocab_size:
                final_token_id = unk_id_to_use
            else:
                final_token_id = token_id_from_vocab_file
            tokens.append(final_token_id)
        return tokens

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        instruction = item.get("instruction", "")
        output_text = item.get("output", "")

        full_text = ""
        prompt_part_for_len_calc = ""

        if "{instruction}" in self.prompt_template and "{output}" in self.prompt_template:
            full_text = self.prompt_template.format(instruction=instruction, output=output_text)
            prompt_part_for_len_calc = self.prompt_template.format(instruction=instruction, output="").rstrip()
        else:
            full_text = f"User: {instruction}\nAssistant: {output_text}"
            prompt_part_for_len_calc = f"User: {instruction}\nAssistant:".rstrip()
        
        prompt_tokens: List[int]
        full_tokenized: List[int]

        if self.is_hf_tokenizer:
            full_text_with_eos = full_text + self.tokenizer.eos_token if self.tokenizer.eos_token else full_text
            # Tokenize the prompt part *without* EOS to determine its length accurately
            # Add_special_tokens=False for prompt part usually, as BOS might be added by full tokenization
            prompt_tokenized_obj = self.tokenizer(prompt_part_for_len_calc, add_special_tokens=False, truncation=False)
            prompt_tokens = prompt_tokenized_obj["input_ids"]
            
            # Tokenize the full text, allowing truncation and adding special tokens (like BOS, EOS if tokenizer configured to)
            full_tokenized_obj = self.tokenizer(full_text_with_eos, add_special_tokens=True, truncation=True, max_length=self.max_length)
            full_tokenized = full_tokenized_obj["input_ids"]

            # Determine length of prompt part *within* the full_tokenized sequence
            # This needs to account for how HF tokenizer adds special tokens
            # A robust way: tokenize prompt_part_for_len_calc *with* add_special_tokens=True
            # and see if it's a prefix of full_tokenized.
            temp_prompt_tokenized_with_specials = self.tokenizer(prompt_part_for_len_calc, add_special_tokens=True, truncation=True, max_length=self.max_length)["input_ids"]
            len_prompt_tokens = 0
            if full_tokenized[:len(temp_prompt_tokenized_with_specials)] == temp_prompt_tokenized_with_specials:
                len_prompt_tokens = len(temp_prompt_tokenized_with_specials)
                # If the prompt part itself ended with EOS (e.g. if output was empty in template), exclude that EOS from masking.
                if len_prompt_tokens > 0 and temp_prompt_tokenized_with_specials[-1] == self.eos_token_id:
                    # Check if the *actual* user input part of the prompt (before assistant's turn) would have an EOS.
                    # This is complex. A simpler heuristic is to assume the tokenized prompt_part_for_len_calc
                    # is what we want to mask. If add_special_tokens=True added a BOS to it, count that.
                    if self.tokenizer.bos_token_id is not None and \
                       temp_prompt_tokenized_with_specials and \
                       temp_prompt_tokenized_with_specials[0] == self.tokenizer.bos_token_id and \
                       (not prompt_tokens or prompt_tokens[0] != self.tokenizer.bos_token_id):
                        len_prompt_tokens = len(prompt_tokens) + 1 # User text + BOS
                    else:
                        len_prompt_tokens = len(prompt_tokens)
            else: # Fallback if prefix match fails (e.g. due to truncation differences)
                len_prompt_tokens = len(prompt_tokens)
                if self.tokenizer.bos_token_id is not None and full_tokenized and full_tokenized[0] == self.tokenizer.bos_token_id:
                    len_prompt_tokens +=1 # Account for BOS if it was added to full_tokenized

        else: # Manual vocab
            prompt_tokens = self._manual_tokenize(prompt_part_for_len_calc)
            output_tokens = self._manual_tokenize(output_text)
            # For manual, explicitly add EOS to the combined sequence before truncation
            full_tokenized_raw = prompt_tokens + output_tokens + [self.eos_token_id]
            if len(full_tokenized_raw) > self.max_length:
                full_tokenized = full_tokenized_raw[:self.max_length-1] + [self.eos_token_id] # Ensure EOS if truncated
            else:
                full_tokenized = full_tokenized_raw
            len_prompt_tokens = len(prompt_tokens)

        input_ids = list(full_tokenized) # Make it a mutable list

        if len(input_ids) < self.max_length:
            input_ids.extend([self.pad_token_id] * (self.max_length - len(input_ids)))

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != self.pad_token_id).long()

        labels = input_ids_tensor.clone()
        
        # Mask prompt tokens
        # Ensure len_prompt_tokens doesn't exceed the actual sequence length before padding
        actual_seq_len_before_padding = len(full_tokenized)
        len_prompt_tokens_to_mask = min(len_prompt_tokens, actual_seq_len_before_padding)
        labels[:len_prompt_tokens_to_mask] = -100
        
        # Mask padding tokens in labels
        labels[input_ids_tensor == self.pad_token_id] = -100
        
        # Ensure EOS token is not masked if it's part of the target output
        # If the last non-pad token is EOS, and it wasn't part of the prompt, it should be a label
        if len_prompt_tokens_to_mask < actual_seq_len_before_padding and \
           input_ids_tensor[actual_seq_len_before_padding-1] == self.eos_token_id:
            labels[actual_seq_len_before_padding-1] = input_ids_tensor[actual_seq_len_before_padding-1]

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask,
            "labels": labels
        }

class ApertisTrainer:
    def __init__(
        self, model: ApertisForCausalLM, train_dataset: Union[ApertisPretrainDataset, ApertisFineTuneDataset],
        val_dataset: Optional[Union[ApertisPretrainDataset, ApertisFineTuneDataset]] = None,
        output_dir: str = "output", batch_size: int = 4, learning_rate: float = 5e-5, weight_decay: float = 0.01,
        num_epochs: int = 3, warmup_steps: int = 0, gradient_accumulation_steps: int = 4, max_grad_norm: float = 1.0,
        use_wandb: bool = False, wandb_project: str = "apertis", wandb_run_name: Optional[str] = None,
        fp16: bool = True, device: Optional[str] = None, checkpoint_steps: int = 1000, iteration_checkpoint_steps: int = 0,
        gpu_memory_fraction: float = 0.7, use_gradient_checkpointing: bool = True, eval_every_n_epochs: int = 1,
        dynamic_batch_sizing: bool = True, gpu_ids: Optional[List[int]] = None, distributed_training: bool = False, local_rank: int = -1,
        stop_event: Optional[threading.Event] = None, is_fine_tuning: bool = False,
        original_tokenizer_path_for_ft_hf: Optional[str] = None, # Path to HF tokenizer if used for FT
        original_manual_vocab_path_for_ft: Optional[str] = None # Path to manual vocab.json if used for FT
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.fp16 = fp16
        self.checkpoint_steps = checkpoint_steps
        self.iteration_checkpoint_steps = iteration_checkpoint_steps
        self.gpu_memory_fraction = gpu_memory_fraction
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.eval_every_n_epochs = eval_every_n_epochs
        self.dynamic_batch_sizing = dynamic_batch_sizing
        self.gpu_ids = gpu_ids
        self.distributed_training = distributed_training
        self.local_rank = local_rank
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.is_fine_tuning = is_fine_tuning
        self.original_tokenizer_path_for_ft_hf = original_tokenizer_path_for_ft_hf
        self.original_manual_vocab_path_for_ft = original_manual_vocab_path_for_ft


        os.makedirs(output_dir, exist_ok=True)
        self.world_size = 1
        self.is_main_process = True

        if self.distributed_training:
            if self.local_rank == -1:
                if 'LOCAL_RANK' in os.environ: self.local_rank = int(os.environ['LOCAL_RANK'])
                else: self.local_rank = 0 # Default for single-node, multi-gpu if not set
            if not dist.is_initialized():
                if torch.cuda.is_available(): dist.init_process_group(backend='nccl')
                else: dist.init_process_group(backend='gloo') # Fallback for CPU-only distributed (rare)
            self.world_size = dist.get_world_size()
            self.is_main_process = self.local_rank == 0
            logger.info(f"Distributed training: World size: {self.world_size}, Local rank: {self.local_rank}")

        if self.distributed_training:
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        elif self.gpu_ids and len(self.gpu_ids) > 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        elif device is not None: self.device = torch.device(device)
        else: self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            torch.cuda.empty_cache()
        logger.info(f"Using device: {self.device}")

        if self.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        self.model.to(self.device)

        if self.distributed_training:
            self.model = DDP(self.model, device_ids=[self.local_rank] if torch.cuda.is_available() else None, output_device=self.local_rank if torch.cuda.is_available() else None, find_unused_parameters=False)
        elif self.gpu_ids and len(self.gpu_ids) > 1 and torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            logger.info(f"Using DataParallel with GPUs: {self.gpu_ids}")

        self._create_dataloaders()
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        opt_groups = [{'params': [p for n,p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
                      {'params': [p for n,p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        self.optimizer = optim.AdamW(opt_groups, lr=self.learning_rate)

        if len(self.train_dataloader) > 0 and self.gradient_accumulation_steps > 0:
            num_optimizer_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.gradient_accumulation_steps)
            num_training_steps = num_optimizer_steps_per_epoch * self.num_epochs
        else:
            num_training_steps = 1 # Avoid division by zero

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.learning_rate, total_steps=num_training_steps, pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0)
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)

        if self.use_wandb and self.is_main_process:
            wandb_config = {
                "batch_size": self.batch_size, "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs, "warmup_steps": self.warmup_steps, "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_grad_norm": self.max_grad_norm, "fp16": self.fp16, "device": str(self.device),
                "distributed_training": self.distributed_training, "world_size": self.world_size, "gpu_ids": self.gpu_ids,
                "eval_every_n_epochs": self.eval_every_n_epochs,
                "is_fine_tuning": self.is_fine_tuning
            }
            model_config_for_wandb = {}
            model_to_get_config_from = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
            if hasattr(model_to_get_config_from, 'config') and hasattr(model_to_get_config_from.config, 'to_dict'):
                 model_config_for_wandb = model_to_get_config_from.config.to_dict()
            wandb_config["model_config"] = model_config_for_wandb
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=wandb_config)

    def _create_dataloaders(self):
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True) if self.distributed_training else None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=self.distributed_training)
        if self.val_dataset:
            val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=False) if self.distributed_training else None
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=False)
        else: self.val_dataloader = None

    def train(self):
        logger.info(f"Starting {'fine-tuning' if self.is_fine_tuning else 'pre-training'}")
        best_val_loss = float('inf')
        global_step = 0
        for epoch in range(self.num_epochs):
            if self.stop_event.is_set():
                logger.info(f"Stop event received. Halting training at epoch {epoch+1}.")
                break
            if self.distributed_training and hasattr(self.train_dataloader.sampler, 'set_epoch'): self.train_dataloader.sampler.set_epoch(epoch)
            self.model.train()
            epoch_loss_accumulator = 0.0
            batches_accumulated = 0

            if not self.train_dataloader:
                logger.warning(f"Skipping epoch {epoch+1} as train_dataloader is empty.")
                continue

            progress_bar = tqdm(total=len(self.train_dataloader), desc=f"Epoch {epoch+1}/{self.num_epochs}", disable=not self.is_main_process)
            for step, batch in enumerate(self.train_dataloader):
                if self.stop_event.is_set():
                    logger.info(f"Stop event received during epoch {epoch+1}, step {step+1}. Halting training.")
                    progress_bar.close()
                    break
                try:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                        if loss is None:
                            logger.warning(f"Loss is None for batch {step} in epoch {epoch+1}. Skipping.")
                            progress_bar.update(1)
                            continue
                        loss = loss / self.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                    epoch_loss_accumulator += loss.item()
                    batches_accumulated +=1

                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        global_step += 1

                        current_avg_loss_for_step = epoch_loss_accumulator * self.gradient_accumulation_steps / batches_accumulated if batches_accumulated > 0 else 0.0
                        progress_bar.set_postfix({"loss": f"{current_avg_loss_for_step:.4f}"})

                        if self.use_wandb and self.is_main_process:
                            log_data = {"train/loss": current_avg_loss_for_step, "train/learning_rate": self.scheduler.get_last_lr()[0], "train/epoch_progress": epoch + (step + 1) / len(self.train_dataloader)}
                            if torch.cuda.is_available():
                                log_data["train/gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
                                log_data["train/gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                            wandb.log(log_data, step=global_step)

                        epoch_loss_accumulator = 0.0
                        batches_accumulated = 0

                        if self.checkpoint_steps > 0 and global_step % self.checkpoint_steps == 0 and self.is_main_process: self.save_checkpoint(f"step-{global_step}")
                    if self.iteration_checkpoint_steps > 0 and (step + 1) % self.iteration_checkpoint_steps == 0 and self.is_main_process: self.save_checkpoint(f"epoch{epoch+1}-iter{step+1}")
                except Exception as e:
                    logger.error(f"Error processing batch {step} in epoch {epoch+1}: {e}", exc_info=True)
                    if self.dynamic_batch_sizing and "CUDA out of memory" in str(e) and self.batch_size > 1:
                        self.batch_size = max(1, self.batch_size // 2)
                        logger.warning(f"OOM: Reducing batch size to {self.batch_size}. Restarting dataloader and epoch.")
                        torch.cuda.empty_cache()
                        self._create_dataloaders()
                        progress_bar.close()
                        break # Break from inner loop to restart epoch with new batch size
                    else: raise e
                progress_bar.update(1)

            if self.stop_event.is_set():
                break # Break from outer epoch loop if stopped

            progress_bar.close()
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed.")
            if self.val_dataloader and self.eval_every_n_epochs > 0 and (epoch + 1) % self.eval_every_n_epochs == 0:
                if self.stop_event.is_set():
                    logger.info(f"Stop event received before evaluation for epoch {epoch+1}. Skipping evaluation.")
                    break
                val_loss = self.evaluate()
                if not np.isinf(val_loss):
                    logger.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
                    if self.use_wandb and self.is_main_process: wandb.log({"eval/val_loss": val_loss, "eval/epoch": epoch + 1}, step=global_step)
                    if val_loss < best_val_loss and self.is_main_process:
                        best_val_loss = val_loss
                        self.save_checkpoint("best_model")
                        logger.info(f"New best model saved (Val Loss: {val_loss:.4f})")
            if self.is_main_process: self.save_checkpoint(f"epoch-{epoch+1}")

        if self.is_main_process and not self.stop_event.is_set():
            self.save_checkpoint("final")
        elif self.is_main_process and self.stop_event.is_set():
            logger.info("Training was stopped. Final checkpoint 'final' will not be saved.")

        logger.info("Training process finished.")
        if self.use_wandb and self.is_main_process: wandb.finish()

    def evaluate(self):
        if not self.val_dataloader:
            return float('inf')
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        progress_bar = tqdm(total=len(self.val_dataloader), desc="Validation", disable=not self.is_main_process)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if self.stop_event.is_set():
                    progress_bar.close()
                    logger.info("Stop event received during evaluation. Aborting.")
                    return float('inf')
                try:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                        if loss is not None:
                            total_val_loss += loss.item()
                            num_val_batches +=1
                        else:
                             logger.warning(f"Validation loss is None for batch {batch_idx}.")
                except Exception as e:
                    logger.error(f"Error during validation batch {batch_idx}: {e}", exc_info=True)
                    pass
                progress_bar.update(1)
        progress_bar.close()
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        self.model.train() # Set back to train mode
        return avg_val_loss

    def save_checkpoint(self, name: str):
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model

        # Save model weights
        model_save_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        try:
            torch.save(model_to_save.state_dict(), model_save_path)
            logger.info(f"Model weights saved to {model_save_path}")
        except Exception as e:
            logger.error(f"Error saving model state_dict: {e}")
            return

        # Save model config (reflecting current state, e.g., resized vocab)
        if hasattr(model_to_save, 'config') and hasattr(model_to_save.config, 'save_pretrained'):
            try:
                model_to_save.config.save_pretrained(checkpoint_dir)
                logger.info(f"Model config saved to {checkpoint_dir}/config.json")
            except Exception as e:
                logger.error(f"Error saving model config: {e}")
        else:
            logger.warning("Model or its config does not have save_pretrained method. Config not saved with checkpoint.")
        
        # Save tokenizer
        if self.is_fine_tuning:
            if hasattr(self.train_dataset, 'is_hf_tokenizer') and self.train_dataset.is_hf_tokenizer and \
               hasattr(self.train_dataset.tokenizer, 'save_pretrained'):
                try:
                    self.train_dataset.tokenizer.save_pretrained(checkpoint_dir)
                    logger.info(f"Hugging Face tokenizer saved to {checkpoint_dir}")
                except Exception as e:
                    logger.error(f"Error saving Hugging Face tokenizer for fine-tuned model: {e}")
            elif self.original_manual_vocab_path_for_ft and os.path.exists(self.original_manual_vocab_path_for_ft):
                 # If fine-tuning was done with a manual vocab, copy that vocab.json
                try:
                    shutil.copy(self.original_manual_vocab_path_for_ft, os.path.join(checkpoint_dir, "vocab.json"))
                    logger.info(f"Manual vocab.json (from {self.original_manual_vocab_path_for_ft}) copied to {checkpoint_dir}")
                except Exception as e:
                    logger.error(f"Error copying manual vocab.json for fine-tuned model: {e}")
            else:
                logger.warning("Fine-tuning tokenizer/vocab source not clearly identified or available for saving with checkpoint.")
        else: # Pre-training (assumes manual vocab was used, from data_cfg.tokenizer_path)
            original_vocab_path = self.train_dataset.data_path # This is wrong. Need the actual vocab path used for pretrain_dataset
            # The pretrain_dataset itself doesn't store the original path of the vocab_dict it got.
            # This information should be passed from train_from_config if we want to copy it.
            # For now, assume pre-training checkpoints won't automatically save vocab.json unless explicitly managed.
            # A better approach: train_from_config should handle copying the vocab.json to the output_dir/initial_checkpoint.
            # Or, if dataset was initialized with a path, it should store it.
            # Let's assume for now `self.original_manual_vocab_path_for_ft` could also be set for pretrain if appropriate
            if self.original_manual_vocab_path_for_ft and os.path.exists(self.original_manual_vocab_path_for_ft):
                try:
                    shutil.copy(self.original_manual_vocab_path_for_ft, os.path.join(checkpoint_dir, "vocab.json"))
                    logger.info(f"Manual vocab.json (from {self.original_manual_vocab_path_for_ft}) copied to {checkpoint_dir} for pre-trained model.")
                except Exception as e:
                    logger.error(f"Error copying manual vocab.json for pre-trained model: {e}")
            else:
                 logger.info("Pre-training: vocab.json not automatically saved with this checkpoint. Ensure it's managed separately if it was a manual vocab.")


def get_available_gpus() -> List[Dict[str, Any]]:
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({"id": i, "name": props.name, "total_memory": props.total_memory / (1024**3), "compute_capability": f"{props.major}.{props.minor}"})
    return gpu_info

def train_from_config(config_path: str, stop_event: Optional[threading.Event] = None):
    try:
        with open(config_path, "r", encoding="utf-8") as f: config_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load/parse config {config_path}: {e}")
        return

    data_cfg = config_data.get("data_config",{})
    model_cfg_from_file = config_data.get("model_config",{}) # Config for a *new* model or *base* for FT
    train_cfg = config_data.get("training_config",{})

    is_fine_tuning_mode = train_cfg.get("task_type", "pretrain") == "finetune"
    
    # --- Tokenizer and Vocab Size Determination ---
    # tokenizer_path_in_config: Path to manual vocab.json OR HF tokenizer name/path
    tokenizer_path_in_config = data_cfg.get("tokenizer_path") 
    use_hf_tokenizer_for_ft = data_cfg.get("use_hf_tokenizer_for_finetune", False)

    actual_hf_tokenizer_object = None
    manual_vocab_dict_loaded = None
    final_vocab_size_for_model_config = 0
    final_pad_id, final_bos_id, final_eos_id, final_unk_id = 0, 1, 2, 3 # ApertisConfig defaults
    
    # Path to the tokenizer/vocab that will be saved with the model checkpoint
    path_to_tokenizer_to_save_with_model_checkpoint: Optional[str] = None


    if is_fine_tuning_mode and use_hf_tokenizer_for_ft:
        if not tokenizer_path_in_config:
            logger.error("Fine-tuning with HF tokenizer requested, but tokenizer_path (HF name/path) is missing in data_config.")
            return
        try:
            from transformers import AutoTokenizer
            actual_hf_tokenizer_object = AutoTokenizer.from_pretrained(tokenizer_path_in_config)
            final_vocab_size_for_model_config = actual_hf_tokenizer_object.vocab_size
            # Use tokenizer's special tokens if available, otherwise ApertisConfig defaults will be used by model.
            if actual_hf_tokenizer_object.pad_token_id is not None: final_pad_id = actual_hf_tokenizer_object.pad_token_id
            if actual_hf_tokenizer_object.bos_token_id is not None: final_bos_id = actual_hf_tokenizer_object.bos_token_id
            if actual_hf_tokenizer_object.eos_token_id is not None: final_eos_id = actual_hf_tokenizer_object.eos_token_id
            if actual_hf_tokenizer_object.unk_token_id is not None: final_unk_id = actual_hf_tokenizer_object.unk_token_id
            path_to_tokenizer_to_save_with_model_checkpoint = tokenizer_path_in_config # This could be a Hub ID or local path
            logger.info(f"Using HF Tokenizer '{tokenizer_path_in_config}' for fine-tuning. Effective vocab_size: {final_vocab_size_for_model_config}")
        except Exception as e:
            logger.error(f"Failed to load HF tokenizer '{tokenizer_path_in_config}' for fine-tuning: {e}. Ensure it's a valid path/name.")
            return
    elif tokenizer_path_in_config: # Pre-training with manual vocab OR Fine-tuning with manual vocab
        try:
            manual_vocab_dict_loaded, final_vocab_size_for_model_config = _load_vocabulary_and_get_size(tokenizer_path_in_config)
            # For manual vocab, use its own special tokens if defined, otherwise ApertisConfig defaults.
            if "<pad>" in manual_vocab_dict_loaded: final_pad_id = manual_vocab_dict_loaded["<pad>"]
            if "<bos>" in manual_vocab_dict_loaded: final_bos_id = manual_vocab_dict_loaded["<bos>"]
            if "<eos>" in manual_vocab_dict_loaded: final_eos_id = manual_vocab_dict_loaded["<eos>"]
            if "<unk>" in manual_vocab_dict_loaded: final_unk_id = manual_vocab_dict_loaded["<unk>"]
            path_to_tokenizer_to_save_with_model_checkpoint = tokenizer_path_in_config
            logger.info(f"Using manual vocab from '{tokenizer_path_in_config}'. Effective vocab_size: {final_vocab_size_for_model_config}")
        except Exception as e:
            logger.error(f"Failed to load manual vocab from '{tokenizer_path_in_config}': {e}")
            return
    else:
        logger.error("tokenizer_path (for manual vocab, or as HF name if use_hf_tokenizer_for_finetune=True) is missing in data_config.")
        return

    # --- Dataset Initialization ---
    train_ds: Union[ApertisPretrainDataset, ApertisFineTuneDataset]
    val_ds: Optional[Union[ApertisPretrainDataset, ApertisFineTuneDataset]] = None
    try:
        if is_fine_tuning_mode:
            dataset_tokenizer_arg = actual_hf_tokenizer_object if actual_hf_tokenizer_object else manual_vocab_dict_loaded
            train_ds = ApertisFineTuneDataset(
                data_path=data_cfg.get("train_data_path"), tokenizer=dataset_tokenizer_arg,
                max_length=data_cfg.get("max_length", 512),
                prompt_template=data_cfg.get("prompt_template", "User: {instruction}\nAssistant: {output}"),
                is_hf_tokenizer=bool(actual_hf_tokenizer_object),
                model_config_vocab_size=final_vocab_size_for_model_config,
                model_config_eos_token_id=final_eos_id, model_config_pad_token_id=final_pad_id,
                model_config_unk_token_id=final_unk_id
            )
            if data_cfg.get("val_data_path"):
                val_ds = ApertisFineTuneDataset(
                    data_path=data_cfg.get("val_data_path"), tokenizer=dataset_tokenizer_arg,
                    max_length=data_cfg.get("max_length", 512),
                    prompt_template=data_cfg.get("prompt_template", "User: {instruction}\nAssistant: {output}"),
                    is_hf_tokenizer=bool(actual_hf_tokenizer_object),
                    model_config_vocab_size=final_vocab_size_for_model_config,
                    model_config_eos_token_id=final_eos_id, model_config_pad_token_id=final_pad_id,
                    model_config_unk_token_id=final_unk_id
                )
        else: # Pre-training (always uses manual_vocab_dict_loaded)
            if manual_vocab_dict_loaded is None or final_vocab_size_for_model_config == 0:
                 logger.error("Manual vocabulary must be provided and valid for pre-training.")
                 return
            train_ds = ApertisPretrainDataset(
                data_path=data_cfg.get("train_data_path"), vocab_dict=manual_vocab_dict_loaded,
                model_config_vocab_size=final_vocab_size_for_model_config,
                max_length=data_cfg.get("max_length",512), multimodal=model_cfg_from_file.get("multimodal",False),
                image_dir=data_cfg.get("image_dir"), image_size=model_cfg_from_file.get("image_size",224),
                pad_token_id_from_config=final_pad_id, unk_token_id_from_config=final_unk_id
            )
            if data_cfg.get("val_data_path"):
                val_ds = ApertisPretrainDataset(
                    data_cfg.get("val_data_path"), manual_vocab_dict_loaded,
                    final_vocab_size_for_model_config,
                    data_cfg.get("max_length",512), model_cfg_from_file.get("multimodal",False),
                    data_cfg.get("image_dir"), model_cfg_from_file.get("image_size",224),
                    pad_token_id_from_config=final_pad_id, unk_token_id_from_config=final_unk_id
                )
    except Exception as e:
        logger.error(f"Error creating datasets: {e}", exc_info=True); return

    # --- Model Initialization ---
    model_for_trainer: ApertisForCausalLM
    try:
        if is_fine_tuning_mode and train_cfg.get("pretrained_model_path_for_finetune"):
            pretrained_model_weights_path = train_cfg.get("pretrained_model_path_for_finetune")
            config_dir_for_base_model = pretrained_model_weights_path
            if os.path.isfile(pretrained_model_weights_path):
                config_dir_for_base_model = os.path.dirname(pretrained_model_weights_path)
            
            logger.info(f"Fine-tuning: Loading base model config from: {config_dir_for_base_model}")
            base_model_config = ApertisConfig.from_pretrained(config_dir_for_base_model)
            original_vocab_size_from_base_model_config = base_model_config.vocab_size
            
            # Update base model config with tokenizer's reality BEFORE creating model instance
            base_model_config.vocab_size = final_vocab_size_for_model_config
            base_model_config.pad_token_id = final_pad_id
            base_model_config.bos_token_id = final_bos_id
            base_model_config.eos_token_id = final_eos_id
            base_model_config.unk_token_id = final_unk_id
            
            # model_cfg_from_file can contain overrides for the base model structure (e.g. multimodal=True for a non-MM base)
            # This is advanced. Usually, for fine-tuning, model_cfg_from_file would be empty or just for minor tweaks.
            merged_config_for_ft_model_dict = base_model_config.to_dict()
            merged_config_for_ft_model_dict.update(model_cfg_from_file) # User overrides on base
            
            # Ensure final vocab/special tokens are from tokenizer, not overridden by model_cfg_from_file
            merged_config_for_ft_model_dict["vocab_size"] = final_vocab_size_for_model_config
            merged_config_for_ft_model_dict["pad_token_id"] = final_pad_id
            merged_config_for_ft_model_dict["bos_token_id"] = final_bos_id
            merged_config_for_ft_model_dict["eos_token_id"] = final_eos_id
            merged_config_for_ft_model_dict["unk_token_id"] = final_unk_id

            final_config_for_ft_model_instance = ApertisConfig.from_dict(merged_config_for_ft_model_dict)
            model_for_trainer = ApertisForCausalLM(final_config_for_ft_model_instance)
            logger.info(f"Fine-tuning: Instantiated ApertisForCausalLM with (potentially merged) config. Effective vocab: {model_for_trainer.config.vocab_size}")

            actual_weights_file = pretrained_model_weights_path
            if os.path.isdir(pretrained_model_weights_path):
                bin_path = os.path.join(pretrained_model_weights_path, "pytorch_model.bin")
                pt_path = os.path.join(pretrained_model_weights_path, "model.pt")
                if os.path.exists(bin_path): actual_weights_file = bin_path
                elif os.path.exists(pt_path): actual_weights_file = pt_path
                else: logger.error(f"No model weights found in dir: {pretrained_model_weights_path}"); return
            
            logger.info(f"Fine-tuning: Loading state_dict from: {actual_weights_file}")
            state_dict_from_checkpoint = torch.load(actual_weights_file, map_location="cpu", weights_only=True)

            # If vocab size changed from base model to current fine-tuning tokenizer, resize embeddings
            if original_vocab_size_from_base_model_config != final_vocab_size_for_model_config:
                logger.info(f"Resizing token embeddings from base model vocab {original_vocab_size_from_base_model_config} to new fine-tuning tokenizer vocab {final_vocab_size_for_model_config}")
                # Temporarily set model's vocab size to original for loading, then resize.
                # This is complex because load_state_dict needs exact matches for embedding layers unless strict=False.
                # A safer approach:
                # 1. Create model with *final* (tokenizer-aligned) vocab size.
                # 2. Manually copy compatible parts of embedding/lm_head from checkpoint.
                # 3. Load the rest of the state_dict with strict=False.

                # Create temporary model with original vocab to extract embeddings correctly
                temp_base_config_for_load = ApertisConfig.from_dict(base_model_config.to_dict()) # Get a clean copy of original base config
                temp_base_config_for_load.vocab_size = original_vocab_size_from_base_model_config # Set its vocab to original
                temp_model_for_load = ApertisForCausalLM(temp_base_config_for_load)
                # Load checkpoint into this temp model (it should match perfectly for embeddings)
                temp_load_result = temp_model_for_load.load_state_dict(state_dict_from_checkpoint, strict=False)
                if temp_load_result.missing_keys or temp_load_result.unexpected_keys:
                     logger.warning(f"Loading into temp model had issues: missing={temp_load_result.missing_keys}, unexpected={temp_load_result.unexpected_keys}")


                # Now, model_for_trainer has the *final* vocab size. Copy matching embedding parts.
                num_tokens_to_copy = min(original_vocab_size_from_base_model_config, final_vocab_size_for_model_config)
                
                # Input embeddings
                model_for_trainer.model.token_embeddings.weight.data[:num_tokens_to_copy, :] = \
                    temp_model_for_load.model.token_embeddings.weight.data[:num_tokens_to_copy, :]
                
                # LM head (if not tied or needs explicit copy)
                if not model_for_trainer.config.tie_word_embeddings:
                    model_for_trainer.lm_head.weight.data[:num_tokens_to_copy, :] = \
                        temp_model_for_load.lm_head.weight.data[:num_tokens_to_copy, :]
                elif model_for_trainer.config.tie_word_embeddings: # If tied, it will pick up from token_embeddings
                    model_for_trainer.lm_head.weight = model_for_trainer.model.token_embeddings.weight


                # Remove embedding/lm_head keys from state_dict before loading into final model instance
                # to avoid size mismatches if we load with strict=True later, or just use strict=False.
                keys_to_delete = []
                if 'model.token_embeddings.weight' in state_dict_from_checkpoint: keys_to_delete.append('model.token_embeddings.weight')
                if 'lm_head.weight' in state_dict_from_checkpoint: keys_to_delete.append('lm_head.weight')
                for key_del in keys_to_delete:
                    if key_del in state_dict_from_checkpoint: del state_dict_from_checkpoint[key_del]
                
                load_result = model_for_trainer.load_state_dict(state_dict_from_checkpoint, strict=False)

            else: # Vocab sizes match, direct load
                load_result = model_for_trainer.load_state_dict(state_dict_from_checkpoint, strict=True) # strict=True if no resize needed
            
            logger.info(f"Base model state_dict loaded for fine-tuning. Load result: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
            if load_result.missing_keys: logger.warning(f"Missing keys during FT base model load: {load_result.missing_keys}")
            if load_result.unexpected_keys: logger.warning(f"Unexpected keys during FT base model load: {load_result.unexpected_keys}")

        else: # Pre-training or fine-tuning from scratch
            logger.info(f"Initializing a new model for {'pre-training' if not is_fine_tuning_mode else 'fine-tuning from scratch'}.")
            # Use create_apertis_model to handle 'target_param_count' and other presets
            target_param_count_from_config = model_cfg_from_file.get("target_param_count", "125M")
            if "model_size" in model_cfg_from_file:
                logger.warning("Found 'model_size' in model_config. It is deprecated. Please use 'target_param_count' (e.g., '125M', '1.5B').")
                # Potentially map old model_size to a target_param_count if desired, or just use default.
                # For now, if model_size is present but target_param_count is not, we'll use the default "125M".
                # If target_param_count IS present, it will take precedence over model_size.
                if not model_cfg_from_file.get("target_param_count"):
                    logger.info(f"Using default target_param_count '{target_param_count_from_config}' due to deprecated 'model_size' and no 'target_param_count' found.")

            model_for_trainer = create_apertis_model(
                target_param_count=target_param_count_from_config,
                vocab_size_override=final_vocab_size_for_model_config,
                attention_type_override=model_cfg_from_file.get("attention_type"),
                multimodal=model_cfg_from_file.get("multimodal", False),
                use_expert_system=model_cfg_from_file.get("use_expert_system", False),
                # Pass other relevant args from model_cfg_from_file if create_apertis_model accepts them
                # e.g., num_experts_target_override, experts_per_token_target_override, ssm parameters, config_overrides
                num_experts_target_override=model_cfg_from_file.get("num_experts"), # ApertisConfig uses num_experts
                experts_per_token_target_override=model_cfg_from_file.get("experts_per_token"),
                use_flash_attention=model_cfg_from_file.get("use_flash_attention", False), # Pass this through
                ssm_d_inner=model_cfg_from_file.get("ssm_d_inner"),
                ssm_d_state=model_cfg_from_file.get("ssm_d_state", 16), # Default from ApertisConfig
                ssm_dt_rank=model_cfg_from_file.get("ssm_dt_rank", "auto"), # Default from ApertisConfig
                ssm_conv_kernel=model_cfg_from_file.get("ssm_conv_kernel", 4), # Default from ApertisConfig
                config_overrides=model_cfg_from_file.get("config_overrides") # Pass general overrides
            )
            # Ensure special token IDs from tokenizer are respected
            model_for_trainer.config.pad_token_id = final_pad_id
            model_for_trainer.config.bos_token_id = final_bos_id
            model_for_trainer.config.eos_token_id = final_eos_id
            model_for_trainer.config.unk_token_id = final_unk_id

            logger.info(f"Initialized NEW ApertisForCausalLM with config: {model_for_trainer.config.to_dict()}")
        
        logger.info(f"Final model object for trainer has config: {model_for_trainer.config.to_dict()}")
    except Exception as e:
        logger.error(f"Error creating model for training: {e}", exc_info=True); return

    actual_stop_event = stop_event if stop_event is not None else threading.Event()

    try:
        trainer = ApertisTrainer(
            model_for_trainer, train_ds, val_ds,
            train_cfg.get("output_dir","output"), train_cfg.get("batch_size",4), train_cfg.get("learning_rate",5e-5),
            train_cfg.get("weight_decay",0.01), train_cfg.get("num_epochs",3), train_cfg.get("warmup_steps",0),
            train_cfg.get("gradient_accumulation_steps",4), train_cfg.get("max_grad_norm",1.0),
            train_cfg.get("use_wandb",False), train_cfg.get("wandb_project","apertis"), train_cfg.get("wandb_run_name"),
            train_cfg.get("fp16",True), train_cfg.get("device"), train_cfg.get("checkpoint_steps",0),
            train_cfg.get("iteration_checkpoint_steps",0), train_cfg.get("gpu_memory_fraction",0.7),
            train_cfg.get("use_gradient_checkpointing",True), train_cfg.get("eval_every_n_epochs",1),
            train_cfg.get("dynamic_batch_sizing",True), train_cfg.get("gpu_ids"),
            train_cfg.get("distributed_training",False), train_cfg.get("local_rank",-1),
            stop_event=actual_stop_event,
            is_fine_tuning=is_fine_tuning_mode,
            original_tokenizer_path_for_ft_hf=tokenizer_path_in_config if is_fine_tuning_mode and actual_hf_tokenizer_object else None,
            original_manual_vocab_path_for_ft=tokenizer_path_in_config if (is_fine_tuning_mode and manual_vocab_dict_loaded) or (not is_fine_tuning_mode and manual_vocab_dict_loaded) else None
        )
    except Exception as e:
        logger.error(f"Error initializing ApertisTrainer: {e}", exc_info=True); return

    try:
        logger.info(f"Starting {'fine-tuning' if is_fine_tuning_mode else 'pre-training'} with config: {config_path}")
        trainer.train()
        if actual_stop_event.is_set():
            logger.info(f"Training for {config_path} was stopped by user request.")
        else:
            logger.info(f"Finished training for {config_path}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        pass # Allow flow to continue if thread was daemonized

class YoloStyleTrainingPipeline: # Kept for compatibility if used elsewhere directly
    def __init__(self, config_path: str, stop_event: Optional[threading.Event] = None):
        self.config_path = config_path
        self.stop_event = stop_event if stop_event is not None else threading.Event()
    def train(self):
        train_from_config(self.config_path, self.stop_event)

def create_sample_config(output_path: str):
    """Creates a sample training configuration file."""
    sample_config = {
        "data_config": {
            "train_data_path": "path/to/train.jsonl",
            "val_data_path": "path/to/val.jsonl", # Optional
            "tokenizer_path": "path/to/vocab.json", # Or HF tokenizer name if use_hf_tokenizer_for_finetune=True
            "use_hf_tokenizer_for_finetune": False, # Set true if tokenizer_path is HF name for fine-tuning
            "max_length": 512,
            "prompt_template": "User: {instruction}\nAssistant: {output}", # For fine-tuning
            "image_dir": None, # For multimodal pre-training
            "image_size": 224 # For multimodal pre-training
        },
        "model_config": { # These are for NEW models (pre-training) OR OVERRIDES for fine-tuning base
            "target_param_count": "125M", # Target parameter count (e.g., "10M", "1.5B"). Ignored if fine-tuning from pretrained_model_path.
            "attention_type": "standard_mha", # Options: "standard_mha", "selective_ssm"
            "use_flash_attention": False, # Set to true to try using FlashAttention for "standard_mha"
            "multimodal": False,
            "use_expert_system": False,
            "num_experts": 8, # Used if use_expert_system is True
            "experts_per_token": 2, # Used if use_expert_system is True
            # "config_overrides": { # Optional: Directly override specific ApertisConfig values AFTER parameter-based calculation
            #    "hidden_size": 768, 
            #    "num_hidden_layers": 12
            # }
            # Other ApertisConfig params can be added here to override defaults for new models
            # e.g., "vocab_size", "max_position_embeddings"
        },
        "training_config": {
            "task_type": "pretrain", # "pretrain" or "finetune"
            "pretrained_model_path_for_finetune": None, # Path to base model dir for fine-tuning
            "output_dir": "output/my_apertis_model",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "num_epochs": 3,
            "warmup_steps": 0, # Calculated automatically if 0 based on total_steps * 0.1
            "gradient_accumulation_steps": 4,
            "max_grad_norm": 1.0,
            "eval_every_n_epochs": 1,
            "checkpoint_steps": 1000, # Save every N global steps (optimizer steps)
            "iteration_checkpoint_steps": 0, # Save every N iterations (batches) within an epoch
            "use_wandb": False,
            "wandb_project": "apertis",
            "wandb_run_name": None, # Auto-generated if None
            "fp16": True,
            "device": None, # Auto-detects CUDA, or set "cpu"
            "gpu_memory_fraction": 0.7,
            "use_gradient_checkpointing": True,
            "dynamic_batch_sizing": True,
            "gpu_ids": None, # e.g., [0, 1] or None for all available/auto
            "distributed_training": False, # Auto-true if len(gpu_ids) > 1
            "local_rank": -1 # For DDP, usually set by launcher
        }
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(sample_config, f, indent=2)
        logger.info(f"Sample configuration created at {output_path}")
    except Exception as e:
        logger.error(f"Failed to create sample config at {output_path}: {e}")