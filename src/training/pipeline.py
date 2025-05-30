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
                        raise ValueError("Invalid token ID found in vocabulary file.")
                    if token_id in seen_ids:
                        raise ValueError("Duplicate token ID found in vocabulary file.")
                    seen_ids.add(token_id)
                    if token_id > max_id_found:
                        max_id_found = token_id
                if max_id_found >= actual_vocab_size:
                    pass
        else:
            raise ValueError(f"Unsupported vocabulary format in {tokenizer_path}: {type(vocab_data)}")

        return actual_vocab_dict, actual_vocab_size

    except Exception as e:
        raise

class ApertisPretrainDataset(Dataset):
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
                        if "text" not in item:
                            continue
                        data.append(item)
                    except json.JSONDecodeError as e:
                        pass
        except FileNotFoundError:
             raise
        return data

    def _tokenize(self, text: str) -> List[int]:
        tokens = []
        unk_token_name = "<unk>"
        default_unk_id = min(3, self.model_vocab_size -1) if self.model_vocab_size > 3 else 0
        if self.model_vocab_size == 0: default_unk_id = 0
        unk_token_id = self.vocab.get(unk_token_name)
        if unk_token_id is None:
            unk_token_id = default_unk_id
        elif unk_token_id >= self.model_vocab_size :
             unk_token_id = default_unk_id

        current_tokens = []
        if isinstance(text, str):
            current_tokens = text.split()
        elif isinstance(text, list):
            current_tokens = text

        for word_or_token in current_tokens:
            if isinstance(word_or_token, int):
                token_id = word_or_token
            else:
                token_id = self.vocab.get(str(word_or_token), unk_token_id)

            if token_id >= self.model_vocab_size:
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
             blank = torch.zeros(3, self.image_size, self.image_size)
             return blank
        except Exception as e:
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank

    def __len__(self) -> int: return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item.get("text", "")

        input_ids = self._tokenize(input_text)

        pad_token_name = "<pad>"
        default_pad_id = 0
        pad_token_id = self.vocab.get(pad_token_name, default_pad_id)

        if pad_token_name not in self.vocab:
            pass
        elif pad_token_id >= self.model_vocab_size:
             pad_token_id = default_pad_id

        seq_len = len(input_ids)
        if seq_len > self.max_length: input_ids = input_ids[:self.max_length]
        elif seq_len < self.max_length: input_ids.extend([pad_token_id] * (self.max_length - seq_len))

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != pad_token_id).long()

        labels = input_ids_tensor.clone()

        output = {"input_ids": input_ids_tensor, "attention_mask": attention_mask, "labels": labels}

        if self.multimodal and "image" in item:
             if self.image_dir is None and not os.path.isabs(item["image"]):
                  pass
             else:
                output["pixel_values"] = self._load_image(item["image"])
        return output

class ApertisFineTuneDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 512,
        prompt_template: str = "User: {instruction}\nAssistant: {output}",
        hf_tokenizer_name: Optional[str] = None,
        vocab_dict: Optional[Dict[str, int]] = None,
        model_vocab_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.data = self._load_data()

        self.is_hf_tokenizer = False
        if hf_tokenizer_name:
            from transformers import AutoTokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer_name)
                self.is_hf_tokenizer = True
            except Exception as e:
                raise ValueError(f"Could not load HF Tokenizer: {hf_tokenizer_name}. Please check the name/path.") from e

            self.eos_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

            if self.eos_token_id is None:
                common_eos_strings = ["<|endoftext|>", "</s>", "<|im_end|>"]
                found_eos = False
                for eos_str in common_eos_strings:
                    if hasattr(self.tokenizer, "vocab") and eos_str in self.tokenizer.vocab:
                        self.eos_token_id = self.tokenizer.vocab[eos_str]
                        if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token is None:
                            self.tokenizer.eos_token = eos_str
                        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is None:
                            self.tokenizer.eos_token_id = self.eos_token_id
                        found_eos = True
                        break
                if not found_eos:
                    err_msg = (f"HF Tokenizer '{hf_tokenizer_name}' must have a detectable EOS token "
                               f"(eos_token_id attribute or one of {common_eos_strings} in its vocab) "
                               "for generative fine-tuning. Please choose a tokenizer designed for generation.")
                    raise ValueError(err_msg)

            if self.pad_token_id is None:
                self.pad_token_id = self.eos_token_id
                if hasattr(self.tokenizer, 'pad_token') and self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is None:
                    self.tokenizer.pad_token_id = self.eos_token_id

        elif vocab_dict and model_vocab_size is not None and eos_token_id is not None and pad_token_id is not None:
            self.vocab = vocab_dict
            self.model_vocab_size = model_vocab_size
            self.eos_token_id = eos_token_id
            self.pad_token_id = pad_token_id
            self.is_hf_tokenizer = False
        else:
            raise ValueError("For ApertisFineTuneDataset, provide either hf_tokenizer_name or all of (vocab_dict, model_vocab_size, eos_token_id, pad_token_id).")

    def _load_data(self) -> List[Dict]:
        data = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if "instruction" not in item or "output" not in item:
                            continue
                        data.append(item)
                    except json.JSONDecodeError as e:
                        pass
        except FileNotFoundError:
            raise
        return data

    def _manual_tokenize(self, text: str) -> List[int]:
        if self.is_hf_tokenizer:
            raise RuntimeError("_manual_tokenize called with HF tokenizer setup.")

        tokens = []
        unk_token_name = "<unk>"
        default_unk_id = min(3, self.model_vocab_size -1) if self.model_vocab_size > 3 else self.pad_token_id
        unk_token_id = self.vocab.get(unk_token_name, default_unk_id)
        if unk_token_id >= self.model_vocab_size : unk_token_id = default_unk_id

        for word in text.split():
            token_id = self.vocab.get(word, unk_token_id)
            if token_id >= self.model_vocab_size:
                token_id = unk_token_id
            tokens.append(token_id)
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
            instruction_prefix = "User: "
            output_prefix = "Assistant: "
            full_text = f"{instruction_prefix}{instruction}{output_prefix}{output_text}"
            prompt_part_for_len_calc = f"{instruction_prefix}{instruction}{output_prefix}".rstrip()

        if self.is_hf_tokenizer:
            prompt_tokens = self.tokenizer.encode(prompt_part_for_len_calc, add_special_tokens=False)
            full_text_with_eos = full_text + (self.tokenizer.eos_token if self.tokenizer.eos_token else "")
            full_tokenized = self.tokenizer.encode(full_text_with_eos, add_special_tokens=True, truncation=True, max_length=self.max_length)

            len_prompt_tokens = 0
            if self.tokenizer.bos_token_id is not None and \
               full_tokenized and full_tokenized[0] == self.tokenizer.bos_token_id and \
               (not prompt_tokens or prompt_tokens[0] != self.tokenizer.bos_token_id):
                len_prompt_tokens = len(prompt_tokens) + 1
            else:
                len_prompt_tokens = len(prompt_tokens)
        else:
            prompt_tokens = self._manual_tokenize(prompt_part_for_len_calc)
            output_tokens_with_eos = self._manual_tokenize(output_text) + [self.eos_token_id]
            full_tokenized = prompt_tokens + output_tokens_with_eos
            if len(full_tokenized) > self.max_length:
                full_tokenized = full_tokenized[:self.max_length-1] + [self.eos_token_id]
            len_prompt_tokens = len(prompt_tokens)

        input_ids = list(full_tokenized)

        if len(input_ids) < self.max_length:
            input_ids.extend([self.pad_token_id] * (self.max_length - len(input_ids)))

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = (input_ids_tensor != self.pad_token_id).long()

        labels = input_ids_tensor.clone()

        len_prompt_tokens = min(len_prompt_tokens, self.max_length)
        labels[:len_prompt_tokens] = -100
        labels[input_ids_tensor == self.pad_token_id] = -100

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
        stop_event: Optional[threading.Event] = None, is_fine_tuning: bool = False
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
            num_training_steps = 1

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

            if self.stop_event.is_set():
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
                    return float('inf')
                try:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        loss = outputs[0] if isinstance(outputs, tuple) else outputs.loss
                        if loss is not None:
                            total_val_loss += loss.item()
                            num_val_batches +=1
                except Exception as e:
                    pass
                progress_bar.update(1)
        progress_bar.close()
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        self.model.train()
        return avg_val_loss

    def save_checkpoint(self, name: str):
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model

        model_config = None
        if hasattr(model_to_save, 'config'): model_config = model_to_save.config
        elif hasattr(self.model, 'config'): model_config = self.model.config

        model_save_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        try: torch.save(model_to_save.state_dict(), model_save_path)
        except Exception as e:
            return

        if model_config and hasattr(model_config, 'to_dict'):
            config_save_path = os.path.join(checkpoint_dir, "config.json")
            try:
                with open(config_save_path, "w", encoding="utf-8") as f: json.dump(model_config.to_dict(), f, indent=2)
            except Exception as e:
                pass

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
    except Exception as e:
        logger.error(f"Failed to load/parse config {config_path}: {e}")
        return

    data_cfg = config.get("data_config",{})
    model_cfg = config.get("model_config",{})
    train_cfg = config.get("training_config",{})

    is_fine_tuning_mode = train_cfg.get("task_type", "pretrain") == "finetune"
    tokenizer_p = data_cfg.get("tokenizer_path")

    loaded_vocab_dict = None
    actual_vocab_s = 0
    hf_tokenizer_name_for_ft = None
    apertis_cfg_defaults = ApertisConfig()
    eos_token_id_for_ft = model_cfg.get("eos_token_id", apertis_cfg_defaults.eos_token_id)
    pad_token_id_for_ft = model_cfg.get("pad_token_id", apertis_cfg_defaults.pad_token_id)
    temp_hf_tokenizer_for_ft = None

    if is_fine_tuning_mode and data_cfg.get("use_hf_tokenizer_for_finetune", False):
        hf_tokenizer_name_for_ft = tokenizer_p
        try:
            from transformers import AutoTokenizer
            temp_hf_tokenizer_for_ft = AutoTokenizer.from_pretrained(hf_tokenizer_name_for_ft)
            actual_vocab_s = temp_hf_tokenizer_for_ft.vocab_size

            current_eos = temp_hf_tokenizer_for_ft.eos_token_id
            current_pad = temp_hf_tokenizer_for_ft.pad_token_id

            if current_eos is None:
                common_eos_strings = ["<|endoftext|>", "</s>", "<|im_end|>"]
                for eos_str in common_eos_strings:
                    if hasattr(temp_hf_tokenizer_for_ft, "vocab") and eos_str in temp_hf_tokenizer_for_ft.vocab:
                        current_eos = temp_hf_tokenizer_for_ft.vocab[eos_str]
                        break
            eos_token_id_for_ft = current_eos if current_eos is not None else apertis_cfg_defaults.eos_token_id

            pad_token_id_for_ft = current_pad if current_pad is not None else eos_token_id_for_ft

            logger.info(f"Using HF Tokenizer '{hf_tokenizer_name_for_ft}' for fine-tuning. Vocab size: {actual_vocab_s}, EOS: {eos_token_id_for_ft}, PAD: {pad_token_id_for_ft}")
        except Exception as e:
            logger.error(f"Failed to load HF tokenizer '{hf_tokenizer_name_for_ft}' for fine-tuning: {e}. Ensure it's a valid path/name.")
            return
    elif tokenizer_p:
        try:
            loaded_vocab_dict, actual_vocab_s = _load_vocabulary_and_get_size(tokenizer_p)
            if is_fine_tuning_mode and loaded_vocab_dict:
                eos_token_id_for_ft = loaded_vocab_dict.get("<eos>", model_cfg.get("eos_token_id", apertis_cfg_defaults.eos_token_id))
                pad_token_id_for_ft = loaded_vocab_dict.get("<pad>", model_cfg.get("pad_token_id", apertis_cfg_defaults.pad_token_id))
        except Exception as e:
            logger.error(f"Failed to load manual vocab from '{tokenizer_p}': {e}")
            return
    else:
        logger.error("tokenizer_path (for manual vocab or as HF name if use_hf_tokenizer_for_finetune=True) missing in data_config."); return

    train_ds: Union[ApertisPretrainDataset, ApertisFineTuneDataset]
    val_ds: Optional[Union[ApertisPretrainDataset, ApertisFineTuneDataset]] = None

    try:
        if is_fine_tuning_mode:
            train_ds = ApertisFineTuneDataset(
                data_path=data_cfg.get("train_data_path"),
                tokenizer=temp_hf_tokenizer_for_ft if hf_tokenizer_name_for_ft else loaded_vocab_dict,
                max_length=data_cfg.get("max_length", 512),
                prompt_template=data_cfg.get("prompt_template", "User: {instruction}\nAssistant: {output}"),
                hf_tokenizer_name=hf_tokenizer_name_for_ft,
                vocab_dict=loaded_vocab_dict if not hf_tokenizer_name_for_ft else None,
                model_vocab_size=actual_vocab_s if not hf_tokenizer_name_for_ft else None,
                eos_token_id=eos_token_id_for_ft,
                pad_token_id=pad_token_id_for_ft
            )
            if data_cfg.get("val_data_path"):
                val_ds = ApertisFineTuneDataset(
                    data_path=data_cfg.get("val_data_path"),
                    tokenizer=temp_hf_tokenizer_for_ft if hf_tokenizer_name_for_ft else loaded_vocab_dict,
                    max_length=data_cfg.get("max_length", 512),
                    prompt_template=data_cfg.get("prompt_template", "User: {instruction}\nAssistant: {output}"),
                    hf_tokenizer_name=hf_tokenizer_name_for_ft,
                    vocab_dict=loaded_vocab_dict if not hf_tokenizer_name_for_ft else None,
                    model_vocab_size=actual_vocab_s if not hf_tokenizer_name_for_ft else None,
                    eos_token_id=eos_token_id_for_ft,
                    pad_token_id=pad_token_id_for_ft
                )
        else:
            if loaded_vocab_dict is None or actual_vocab_s == 0:
                 logger.error("Manual vocabulary (vocab.json) must be provided and valid for pre-training.")
                 return
            train_ds = ApertisPretrainDataset(
                data_path=data_cfg.get("train_data_path"),
                vocab_dict=loaded_vocab_dict,
                model_vocab_size=actual_vocab_s,
                max_length=data_cfg.get("max_length",512),
                multimodal=model_cfg.get("multimodal",False),
                image_dir=data_cfg.get("image_dir"),
                image_size=data_cfg.get("image_size",224)
            )
            if data_cfg.get("val_data_path"):
                val_ds = ApertisPretrainDataset(
                    data_cfg.get("val_data_path"),
                    loaded_vocab_dict,
                    actual_vocab_s,
                    data_cfg.get("max_length",512),
                    model_cfg.get("multimodal",False),
                    data_cfg.get("image_dir"),
                    data_cfg.get("image_size",224)
                )
    except Exception as e:
        logger.error(f"Error creating datasets: {e}", exc_info=True)
        return

    model: ApertisForCausalLM
    base_model_config_loaded_for_ft = None
    try:
        if is_fine_tuning_mode and train_cfg.get("pretrained_model_path_for_finetune"):
            pretrained_model_weights_path = train_cfg.get("pretrained_model_path_for_finetune")

            base_model_config_dir = ""
            if os.path.isfile(pretrained_model_weights_path):
                base_model_config_dir = os.path.dirname(pretrained_model_weights_path)
            elif os.path.isdir(pretrained_model_weights_path):
                base_model_config_dir = pretrained_model_weights_path
            else:
                logger.error(f"Invalid pretrained_model_path_for_finetune: {pretrained_model_weights_path}")
                return

            logger.info(f"Fine-tuning: Attempting to load base model config from directory: {base_model_config_dir}")
            try:
                base_model_config_loaded_for_ft = ApertisConfig.from_pretrained(base_model_config_dir)
                original_vocab_size_from_checkpoint_config = base_model_config_loaded_for_ft.vocab_size
                logger.info(f"Base model config loaded. Original hidden_size: {base_model_config_loaded_for_ft.hidden_size}, vocab_size: {original_vocab_size_from_checkpoint_config}")

                model = ApertisForCausalLM(base_model_config_loaded_for_ft) # base_model_config_loaded_for_ft.vocab_size might get modified by resize_token_embeddings
                logger.info(f"Instantiated base model with its original config for fine-tuning.")

                if original_vocab_size_from_checkpoint_config != actual_vocab_s:
                    logger.info(f"Resizing token embeddings from base model vocab {original_vocab_size_from_checkpoint_config} to new fine-tuning tokenizer vocab {actual_vocab_s}")
                    model.resize_token_embeddings(actual_vocab_s)

                actual_weights_file_to_load = ""
                if os.path.isdir(pretrained_model_weights_path):
                    bin_path = os.path.join(pretrained_model_weights_path, "pytorch_model.bin")
                    pt_path = os.path.join(pretrained_model_weights_path, "model.pt")
                    if os.path.exists(bin_path): actual_weights_file_to_load = bin_path
                    elif os.path.exists(pt_path): actual_weights_file_to_load = pt_path
                elif os.path.isfile(pretrained_model_weights_path):
                    actual_weights_file_to_load = pretrained_model_weights_path

                if not actual_weights_file_to_load or not os.path.exists(actual_weights_file_to_load):
                     logger.error(f"No model weights file (pytorch_model.bin or model.pt) found based on path: {pretrained_model_weights_path}")
                     return

                logger.info(f"Loading state_dict from: {actual_weights_file_to_load} into model with hidden_size: {model.config.hidden_size} and vocab_size: {model.config.vocab_size}")
                state_dict_from_checkpoint = torch.load(actual_weights_file_to_load, map_location="cpu", weights_only=True)

                checkpoint_original_vocab_size_for_manual_copy = original_vocab_size_from_checkpoint_config
                current_target_vocab_size = actual_vocab_s

                keys_to_delete_from_checkpoint = []

                if 'model.token_embeddings.weight' in state_dict_from_checkpoint:
                    old_embed_weights = state_dict_from_checkpoint['model.token_embeddings.weight']
                    if old_embed_weights.shape[0] == checkpoint_original_vocab_size_for_manual_copy:
                        num_tokens_to_copy = min(checkpoint_original_vocab_size_for_manual_copy, current_target_vocab_size)
                        model.model.token_embeddings.weight.data[:num_tokens_to_copy, :] = old_embed_weights.data[:num_tokens_to_copy, :]
                        logger.info(f"Manually copied {num_tokens_to_copy} token embeddings from checkpoint to resized layer.")
                        keys_to_delete_from_checkpoint.append('model.token_embeddings.weight')
                    else:
                        logger.warning(f"Checkpoint 'model.token_embeddings.weight' shape {old_embed_weights.shape} "
                                       f"doesn't match checkpoint's original config vocab_size {checkpoint_original_vocab_size_for_manual_copy}. Skipping manual copy.")

                if 'lm_head.weight' in state_dict_from_checkpoint:
                    old_lm_head_weights = state_dict_from_checkpoint['lm_head.weight']
                    if old_lm_head_weights.shape[0] == checkpoint_original_vocab_size_for_manual_copy:
                        num_tokens_to_copy = min(checkpoint_original_vocab_size_for_manual_copy, current_target_vocab_size)
                        model.lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head_weights.data[:num_tokens_to_copy, :]
                        logger.info(f"Manually copied {num_tokens_to_copy} lm_head weights from checkpoint to resized layer.")
                        keys_to_delete_from_checkpoint.append('lm_head.weight')
                    else:
                        logger.warning(f"Checkpoint 'lm_head.weight' shape {old_lm_head_weights.shape} "
                                       f"doesn't match checkpoint's original config vocab_size {checkpoint_original_vocab_size_for_manual_copy}. Skipping manual copy.")

                for key_to_del in keys_to_delete_from_checkpoint:
                    if key_to_del in state_dict_from_checkpoint:
                        del state_dict_from_checkpoint[key_to_del]

                load_result = model.load_state_dict(state_dict_from_checkpoint, strict=False)
                logger.info(f"Base model state_dict loaded. Load result: missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)}")
                if load_result.missing_keys: logger.warning(f"Missing keys during load: {load_result.missing_keys}")
                if load_result.unexpected_keys: logger.warning(f"Unexpected keys during load: {load_result.unexpected_keys}")

            except FileNotFoundError as e_cfg_nf:
                logger.error(f"Config file ('config.json') not found in directory '{base_model_config_dir}' for the base model: {e_cfg_nf}. Please ensure a 'config.json' exists alongside the model weights file.")
                return
            except Exception as e_load_base:
                logger.error(f"Error loading or preparing base model using path '{pretrained_model_weights_path}' and config dir '{base_model_config_dir}': {e_load_base}", exc_info=True)
                return
        else:
            logger.info(f"Initializing a new model for {'pre-training' if not is_fine_tuning_mode else 'fine-tuning from scratch'}.")
            final_model_params = ApertisConfig().to_dict()
            final_model_params.update(model_cfg)
            final_model_params["vocab_size"] = actual_vocab_s

            if temp_hf_tokenizer_for_ft:
                if hasattr(temp_hf_tokenizer_for_ft, 'bos_token_id') and temp_hf_tokenizer_for_ft.bos_token_id is not None: final_model_params["bos_token_id"] = temp_hf_tokenizer_for_ft.bos_token_id
                if hasattr(temp_hf_tokenizer_for_ft, 'eos_token_id') and temp_hf_tokenizer_for_ft.eos_token_id is not None: final_model_params["eos_token_id"] = temp_hf_tokenizer_for_ft.eos_token_id
                if hasattr(temp_hf_tokenizer_for_ft, 'pad_token_id') and temp_hf_tokenizer_for_ft.pad_token_id is not None: final_model_params["pad_token_id"] = temp_hf_tokenizer_for_ft.pad_token_id
                if hasattr(temp_hf_tokenizer_for_ft, 'unk_token_id') and temp_hf_tokenizer_for_ft.unk_token_id is not None: final_model_params["unk_token_id"] = temp_hf_tokenizer_for_ft.unk_token_id

            current_model_config_obj = ApertisConfig(**final_model_params)
            model = ApertisForCausalLM(current_model_config_obj)
            logger.info(f"Initialized NEW ApertisForCausalLM with config: {current_model_config_obj.to_dict()}")

        logger.info(f"Final model object for trainer has config: {model.config.to_dict()}")
    except Exception as e:
        logger.error(f"Error creating model for training: {e}", exc_info=True)
        return

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
            stop_event=actual_stop_event,
            is_fine_tuning=is_fine_tuning_mode
        )
    except Exception as e:
        logger.error(f"Error initializing ApertisTrainer: {e}", exc_info=True)
        return

    try:
        logger.info(f"Starting {'fine-tuning' if is_fine_tuning_mode else 'pre-training'} with config: {config_path}")
        trainer.train()
        if actual_stop_event.is_set():
            logger.info(f"Training for {config_path} was stopped by user request.")
        else:
            logger.info(f"Finished training for {config_path}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        pass

class YoloStyleTrainingPipeline:
    def __init__(self, config_path: str, stop_event: Optional[threading.Event] = None):
        self.config_path = config_path
        self.stop_event = stop_event if stop_event is not None else threading.Event()
    def train(self):
        train_from_config(self.config_path, self.stop_event)