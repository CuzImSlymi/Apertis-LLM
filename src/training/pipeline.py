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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.core import ApertisConfig, ApertisForCausalLM

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
        actual_vocab_size = 0

        if isinstance(vocab_data, dict):
            if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                token_list = vocab_data["tokens"]
                actual_vocab_dict = {token: idx for idx, token in enumerate(token_list)}
                actual_vocab_size = len(actual_vocab_dict)
                for token, idx in actual_vocab_dict.items():
                    if idx >= actual_vocab_size:
                        logger.error(f"Inconsistent token ID {idx} for token '{token}' in list-based vocab. Expected max ID {actual_vocab_size-1}.")
                        raise ValueError("Inconsistent token IDs in list-based vocabulary file.")
            else:
                actual_vocab_dict = vocab_data
                actual_vocab_size = len(actual_vocab_dict)
                if not actual_vocab_dict: 
                    return {}, 0
                max_id_found = -1
                seen_ids = set()
                for token, token_id in actual_vocab_dict.items():
                    if not isinstance(token_id, int) or token_id < 0:
                        logger.error(f"Invalid token ID '{token_id}' for token '{token}' in {tokenizer_path}. IDs must be non-negative integers.")
                        raise ValueError("Invalid token ID found in vocabulary file.")
                    if token_id in seen_ids:
                        logger.error(f"Duplicate token ID {token_id} found in {tokenizer_path}.")
                        raise ValueError("Duplicate token ID found in vocabulary file.")
                    seen_ids.add(token_id)
                    if token_id > max_id_found:
                        max_id_found = token_id
                if max_id_found >= actual_vocab_size:
                    logger.warning(
                        f"Vocabulary in {tokenizer_path} has {actual_vocab_size} unique tokens, "
                        f"but the maximum assigned token ID is {max_id_found}. "
                        f"This indicates non-contiguous or out-of-bounds IDs. "
                        f"The model will be built with vocab_size={actual_vocab_size}, and tokens with IDs >= {actual_vocab_size} in the data will cause issues if not handled by the tokenizer. "
                        f"It is STRONGLY recommended to re-index your vocabulary to have dense IDs from 0 to {actual_vocab_size-1}."
                    )
        else:
            raise ValueError(f"Unsupported vocabulary format in {tokenizer_path}: {type(vocab_data)}")
        
        logger.info(f"Loaded {len(actual_vocab_dict)} unique tokens from {tokenizer_path}. Effective vocabulary size for model will be {actual_vocab_size}.")
        return actual_vocab_dict, actual_vocab_size
        
    except Exception as e:
        logger.error(f"Failed to load or determine size of vocabulary at {tokenizer_path}: {e}")
        raise

class ApertisDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        vocab_dict: Dict[str, int], 
        model_vocab_size: int,      
        max_length: int = 512,
        multimodal: bool = False,
        image_dir: Optional[str] = None,
        image_size: int = 224,
    ):
        self.data_path = data_path
        self.vocab = vocab_dict 
        self.model_vocab_size = model_vocab_size
        self.max_length = max_length
        self.multimodal = multimodal
        self.image_dir = image_dir
        self.image_size = image_size
        self.data = self._load_data()
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
                        data.append(item)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line {i+1} in {self.data_path}: {e}")
            logger.info(f"Loaded {len(data)} examples from {self.data_path}")
        except FileNotFoundError:
             logger.error(f"Data file not found: {self.data_path}")
             raise
        return data

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        unk_token_name = "<unk>"
        default_unk_id = min(1, self.model_vocab_size -1) if self.model_vocab_size > 1 else 0 
        if self.model_vocab_size == 0: default_unk_id = 0
        unk_token_id = self.vocab.get(unk_token_name)
        if unk_token_id is None: 
            unk_token_id = default_unk_id
            if self.model_vocab_size > 0 : 
                logger.debug(f"'{unk_token_name}' not found in provided vocabulary. Using ID {unk_token_id} as fallback for OOV words.")
        elif unk_token_id >= self.model_vocab_size : 
             logger.warning(f"'{unk_token_name}' ID {unk_token_id} is out of bounds for model_vocab_size {self.model_vocab_size}. Using {default_unk_id} instead.")
             unk_token_id = default_unk_id
        for word in text.split():
            token_id = self.vocab.get(word, unk_token_id)
            if token_id >= self.model_vocab_size:
                logger.warning(f"Token '{word}' (ID {token_id} from vocab file) is out of bounds for model_vocab_size ({self.model_vocab_size}). Mapping to UNK ID {unk_token_id}. This indicates an issue with your vocab.json (IDs should be < vocab_size).")
                token_id = unk_token_id
            tokens.append(token_id)
        return tokens

    def _load_image(self, image_path: str) -> torch.Tensor:
        if self.image_dir is None: full_path = image_path
        else: full_path = os.path.join(self.image_dir, image_path)
        try:
            image = Image.open(full_path).convert('RGB')
            return self.image_transform(image)
        except FileNotFoundError:
             logger.error(f"Image file not found: {full_path}")
             blank = torch.zeros(3, self.image_size, self.image_size)
             return blank
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item.get("text", "")
        if not input_text: logger.warning(f"Empty text found in item {idx} of {self.data_path}")
        input_ids = self._tokenize(input_text)
        default_pad_id = 0
        pad_token_id = self.vocab.get("<pad>", default_pad_id)
        if "<pad>" not in self.vocab:
            if self.model_vocab_size > 0:
                 logger.debug(f"'<pad>' not found in vocabulary. Using ID {pad_token_id} for padding.")
        elif pad_token_id >= self.model_vocab_size:
             logger.warning(f"'<pad>' ID {pad_token_id} is out of bounds for model_vocab_size {self.model_vocab_size}. Using {default_pad_id} instead.")
             pad_token_id = default_pad_id
        seq_len = len(input_ids)
        if seq_len > self.max_length: input_ids = input_ids[:self.max_length]
        elif seq_len < self.max_length: input_ids.extend([pad_token_id] * (self.max_length - seq_len))
        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != pad_token_id).long()
        output = {"input_ids": input_ids_tensor, "attention_mask": attention_mask, "labels": input_ids_tensor.clone()}
        if self.multimodal and "image" in item:
             if self.image_dir is None and not os.path.isabs(item["image"]):
                  logger.warning(f"Multimodal True, image_dir None, image path '{item['image']}' not absolute. Item {idx}.")
             else:
                output["pixel_values"] = self._load_image(item["image"])
        return output

class ApertisTrainer:
    def __init__(
        self, model: ApertisForCausalLM, train_dataset: ApertisDataset, val_dataset: Optional[ApertisDataset] = None,
        output_dir: str = "output", batch_size: int = 4, learning_rate: float = 5e-5, weight_decay: float = 0.01,
        num_epochs: int = 3, warmup_steps: int = 0, gradient_accumulation_steps: int = 4, max_grad_norm: float = 1.0,
        use_wandb: bool = False, wandb_project: str = "apertis", wandb_run_name: Optional[str] = None,
        fp16: bool = True, device: Optional[str] = None, checkpoint_steps: int = 1000, iteration_checkpoint_steps: int = 0,
        gpu_memory_fraction: float = 0.7, use_gradient_checkpointing: bool = True, eval_every_n_epochs: int = 1,
        dynamic_batch_sizing: bool = True, gpu_ids: Optional[List[int]] = None, distributed_training: bool = False, local_rank: int = -1,
        stop_event: Optional[threading.Event] = None
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

        os.makedirs(output_dir, exist_ok=True)
        self.world_size = 1
        self.is_main_process = True

        if self.distributed_training:
            if self.local_rank == -1:
                if 'LOCAL_RANK' in os.environ: self.local_rank = int(os.environ['LOCAL_RANK'])
                else: self.local_rank = 0
            if not dist.is_initialized():
                if torch.cuda.is_available(): dist.init_process_group(backend='nccl')
                else: dist.init_process_group(backend='gloo')
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
            if self.gpu_memory_fraction < 1.0:
                try:
                    dev_idx = self.device.index if self.device.type == 'cuda' else 0
                    if dev_idx is not None: 
                        total_mem = torch.cuda.get_device_properties(dev_idx).total_memory
                        reserved_mem = int(total_mem * (1 - self.gpu_memory_fraction))
                        logger.info(f"GPU {dev_idx}: Total Mem: {total_mem/1024**3:.2f}GiB, Reserved: {reserved_mem/1024**3:.2f}GiB, Available: {(total_mem-reserved_mem)/1024**3:.2f}GiB")
                except Exception as e: logger.warning(f"GPU memory info error: {e}")
        logger.info(f"Using device: {self.device}")

        if self.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing")
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
            num_training_steps = 1
            if len(self.train_dataloader) == 0 :
                logger.warning("Train dataloader is empty. Scheduler total_steps set to 1.")

        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.learning_rate, total_steps=num_training_steps, pct_start=0.1, anneal_strategy='cos', div_factor=25.0, final_div_factor=10000.0)
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)

        if self.use_wandb and self.is_main_process:
            wandb_config = {
                "batch_size": self.batch_size, "learning_rate": self.learning_rate, "weight_decay": self.weight_decay,
                "num_epochs": self.num_epochs, "warmup_steps": self.warmup_steps, "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "max_grad_norm": self.max_grad_norm, "fp16": self.fp16, "device": str(self.device),
                "distributed_training": self.distributed_training, "world_size": self.world_size, "gpu_ids": self.gpu_ids,
                "eval_every_n_epochs": self.eval_every_n_epochs
            }
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'to_dict'):
                 wandb_config["model_config"] = self.model.config.to_dict()
            wandb.init(project=self.wandb_project, name=self.wandb_run_name, config=wandb_config)

    def _create_dataloaders(self):
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=True) if self.distributed_training else None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=(train_sampler is None), sampler=train_sampler, num_workers=4, pin_memory=True, drop_last=self.distributed_training)
        if self.val_dataset:
            val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.local_rank, shuffle=False) if self.distributed_training else None
            self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=val_sampler, num_workers=4, pin_memory=True, drop_last=False)
        else: self.val_dataloader = None

    def train(self):
        logger.info("Starting training")
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
                            logger.warning(f"Skipping batch {step} in epoch {epoch+1} due to None loss.")
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
                        logger.warning(f"OOM: Reducing batch size to {self.batch_size}. Restarting epoch.")
                        torch.cuda.empty_cache()
                        self._create_dataloaders()
                        progress_bar.close() 
                        break 
                    else: raise e
                progress_bar.update(1)
            
            if self.stop_event.is_set(): # Check again after inner loop
                break

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
                else: logger.warning(f"Epoch {epoch+1} validation skipped/failed.")
            if self.is_main_process: self.save_checkpoint(f"epoch-{epoch+1}")

        if self.is_main_process and not self.stop_event.is_set():
            self.save_checkpoint("final")
        elif self.is_main_process and self.stop_event.is_set():
            logger.info("Training was stopped. Final checkpoint 'final' will not be saved.")
            
        logger.info("Training process finished.")
        if self.use_wandb and self.is_main_process: wandb.finish()

    def evaluate(self):
        if not self.val_dataloader:
            logger.warning("Val dataset not provided. Skipping eval.")
            return float('inf')
        logger.info("Evaluating model...")
        self.model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        progress_bar = tqdm(total=len(self.val_dataloader), desc="Validation", disable=not self.is_main_process)
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                if self.stop_event.is_set():
                    logger.info(f"Stop event received during evaluation (batch {batch_idx+1}). Halting evaluation.")
                    progress_bar.close()
                    return float('inf') # Indicate evaluation was interrupted
                try:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                        if loss is not None:
                            total_val_loss += loss.item()
                            num_val_batches += 1
                        else: logger.warning("None loss during validation batch.")
                except Exception as e: logger.error(f"Error during validation batch: {e}", exc_info=True)
                progress_bar.update(1)
        progress_bar.close()
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        if num_val_batches > 0: logger.info(f"Validation - Avg Loss: {avg_val_loss:.4f}")
        else: logger.warning("Validation - No batches processed.")
        self.model.train()
        return avg_val_loss

    def save_checkpoint(self, name: str):
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model
        
        model_config = None
        if hasattr(model_to_save, 'config'): model_config = model_to_save.config
        elif hasattr(self.model, 'config'): model_config = self.model.config 
        else: logger.error("Cannot find 'config' on model. Config.json will not be saved.")

        model_save_path = os.path.join(checkpoint_dir, "pytorch_model.bin") 
        try: torch.save(model_to_save.state_dict(), model_save_path)
        except Exception as e: logger.error(f"Failed to save model state_dict to {model_save_path}: {e}"); return

        if model_config and hasattr(model_config, 'to_dict'):
            config_save_path = os.path.join(checkpoint_dir, "config.json")
            try:
                with open(config_save_path, "w", encoding="utf-8") as f: json.dump(model_config.to_dict(), f, indent=2)
            except Exception as e: logger.error(f"Failed to save config.json to {config_save_path}: {e}")
        logger.info(f"Checkpoint saved to {checkpoint_dir}")

def get_available_gpus() -> List[Dict[str, Any]]:
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({"id": i, "name": props.name, "total_memory": props.total_memory / (1024**3), "compute_capability": f"{props.major}.{props.minor}"})
    return gpu_info

def train_from_config(config_path: str, stop_event: Optional[threading.Event] = None):
    try:
        with open(config_path, "r", encoding="utf-8") as f: config = json.load(f)
    except Exception as e: logger.error(f"Failed to load/parse config {config_path}: {e}"); return

    data_cfg, model_cfg, train_cfg = config.get("data_config",{}), config.get("model_config",{}), config.get("training_config",{})
    tokenizer_p = data_cfg.get("tokenizer_path")
    if not tokenizer_p: logger.error("tokenizer_path missing in data_config."); return
    
    try:
        loaded_vocab_dict, actual_vocab_s = _load_vocabulary_and_get_size(tokenizer_p)
    except Exception as e: logger.error(f"Failed to load vocab for size determination: {e}"); return

    try:
        train_ds = ApertisDataset(
            data_path=data_cfg.get("train_data_path"), 
            vocab_dict=loaded_vocab_dict, 
            model_vocab_size=actual_vocab_s, 
            max_length=data_cfg.get("max_length",512), 
            multimodal=model_cfg.get("multimodal",False), 
            image_dir=data_cfg.get("image_dir"), 
            image_size=data_cfg.get("image_size",224)
        )
        val_ds = None
        if data_cfg.get("val_data_path"):
            val_ds = ApertisDataset(
                data_cfg.get("val_data_path"), 
                loaded_vocab_dict, 
                actual_vocab_s, 
                data_cfg.get("max_length",512), 
                model_cfg.get("multimodal",False), 
                data_cfg.get("image_dir"), 
                data_cfg.get("image_size",224)
            )
    except Exception as e: logger.error(f"Error creating datasets: {e}"); return
    
    apertis_config_defaults = ApertisConfig().to_dict()
    model_params = {**apertis_config_defaults, **model_cfg} 
    model_params["vocab_size"] = actual_vocab_s 
    
    try:
        model_conf_obj = ApertisConfig(**model_params)
        model = ApertisForCausalLM(model_conf_obj)
        logger.info(f"Created ApertisForCausalLM with config: {model_conf_obj.to_dict()}")
    except Exception as e: logger.error(f"Error creating model with {model_params}: {e}"); return

    actual_stop_event = stop_event if stop_event is not None else threading.Event()

    try:
        trainer = ApertisTrainer(
            model, train_ds, val_ds,
            train_cfg.get("output_dir","output"), train_cfg.get("batch_size",4), train_cfg.get("learning_rate",5e-5),
            train_cfg.get("weight_decay",0.01), train_cfg.get("num_epochs",3), train_cfg.get("warmup_steps",0),
            train_cfg.get("gradient_accumulation_steps",4), train_cfg.get("max_grad_norm",1.0),
            train_cfg.get("use_wandb",False), train_cfg.get("wandb_project","apertis"), train_cfg.get("wandb_run_name"),
            train_cfg.get("fp16",True), train_cfg.get("device"), train_cfg.get("checkpoint_steps",0),
            train_cfg.get("iteration_checkpoint_steps",0), train_cfg.get("gpu_memory_fraction",0.7),
            train_cfg.get("use_gradient_checkpointing",True), train_cfg.get("eval_every_n_epochs",1),
            train_cfg.get("dynamic_batch_sizing",True), train_cfg.get("gpu_ids"),
            train_cfg.get("distributed_training",False), train_cfg.get("local_rank",-1),
            stop_event=actual_stop_event
        )
    except Exception as e: logger.error(f"Error initializing ApertisTrainer: {e}"); return

    try:
        logger.info(f"Starting training with config: {config_path}")
        trainer.train()
        if actual_stop_event.is_set():
            logger.info(f"Training for {config_path} was stopped by user request.")
        else:
            logger.info(f"Finished training for {config_path}")
    except Exception as e: logger.error(f"Error during training: {e}", exc_info=True)

class YoloStyleTrainingPipeline:
    def __init__(self, config_path: str, stop_event: Optional[threading.Event] = None):
        self.config_path = config_path
        self.stop_event = stop_event if stop_event is not None else threading.Event()
    def train(self):
        train_from_config(self.config_path, self.stop_event)