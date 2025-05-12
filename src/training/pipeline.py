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

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.core import ApertisConfig, ApertisForCausalLM

# Import distributed training modules
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper function to load vocabulary and get its size
def _load_vocabulary_size(tokenizer_path: str) -> int:
    """Loads vocabulary and returns its actual size (number of entries)."""
    try:
        with open(tokenizer_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)

        if isinstance(vocab_data, dict):
            if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                # Correct for list format
                return len(vocab_data["tokens"])
            else:
                # Standard format: {"token": id, ...}
                # The size is the number of entries in the dictionary
                return len(vocab_data) # <-- FIX: Use length, not max_id+1
        else:
            raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")
    except Exception as e:
        logger.error(f"Failed to load or determine size of vocabulary at {tokenizer_path}: {e}")
        raise # Re-raise the exception to halt training setup

class ApertisDataset(Dataset):
    """Dataset for training Apertis models."""

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        model_vocab_size: int, # Now required
        max_length: int = 512,
        multimodal: bool = False,
        image_dir: Optional[str] = None,
        image_size: int = 224,
    ):
        """
        Initialize the dataset.

        Args:
            data_path: Path to the dataset file (jsonl format)
            tokenizer_path: Path to the tokenizer vocabulary file
            model_vocab_size: The vocabulary size of the model being trained.
            max_length: Maximum sequence length
            multimodal: Whether to include image data
            image_dir: Directory containing images (required if multimodal=True)
            image_size: Size of images (height and width)
        """
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.multimodal = multimodal
        self.image_dir = image_dir
        self.image_size = image_size
        self.model_vocab_size = model_vocab_size # Use the provided size

        # Load vocabulary
        self.vocab = self._load_vocabulary()
        # Verify consistency (optional but good practice)
        actual_loaded_size = len(self.vocab)
        if actual_loaded_size != self.model_vocab_size:
             logger.warning(f"Provided model_vocab_size ({self.model_vocab_size}) does not match the actual number of entries ({actual_loaded_size}) in {tokenizer_path}. Ensure this is intended.")


        # Load data
        self.data = self._load_data()

        # Set up image transformation if multimodal
        if self.multimodal:
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_data(self) -> List[Dict]:
        """Load data from jsonl file."""
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

    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary from json file."""
        try:
            with open(self.tokenizer_path, 'r', encoding='utf-8') as f:
                vocab_data = json.load(f)

            # Handle different vocabulary formats
            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    # Format: {"tokens": ["token1", "token2", ...]}
                    token_list = vocab_data["tokens"]
                    # Convert list to dictionary with indices as values
                    vocab = {token: idx for idx, token in enumerate(token_list)}
                    logger.info(f"Converted list-based vocabulary to dictionary format with {len(vocab)} tokens")
                else:
                    # Standard format: {"token1": 0, "token2": 1, ...}
                    vocab = vocab_data
            else:
                raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")

            logger.info(f"Loaded vocabulary with {len(vocab)} tokens from {self.tokenizer_path}")
            return vocab
        except FileNotFoundError:
             logger.error(f"Vocabulary file not found: {self.tokenizer_path}")
             raise
        except json.JSONDecodeError as e:
             logger.error(f"Error decoding JSON from vocabulary file {self.tokenizer_path}: {e}")
             raise


    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization by splitting on spaces and mapping to vocabulary."""
        tokens = []

        # Special token IDs from the actual loaded vocab
        unk_token_id = self.vocab.get("<unk>", 3) # Default to 3 if <unk> isn't present

        for word in text.split():
            token_id = self.vocab.get(word, unk_token_id)
            # Ensure token ID is within the *model's* declared vocabulary bounds
            # This check is mainly useful if the vocab file contains IDs >= model_vocab_size
            if token_id >= self.model_vocab_size:
                logger.warning(f"Token '{word}' mapped to ID {token_id} which is >= model_vocab_size ({self.model_vocab_size}). Mapping to UNK ({unk_token_id}). Check vocabulary consistency.")
                token_id = unk_token_id
            tokens.append(token_id)

        return tokens

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        # If image_dir is None, treat image_path as absolute or relative to CWD
        if self.image_dir is None:
             full_path = image_path
        else:
             full_path = os.path.join(self.image_dir, image_path)
        try:
            image = Image.open(full_path).convert('RGB')
            return self.image_transform(image)
        except FileNotFoundError:
             logger.error(f"Image file not found: {full_path}")
             # Return a blank image as fallback
             blank = torch.zeros(3, self.image_size, self.image_size)
             return blank
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            # Return a blank image as fallback
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        item = self.data[idx]

        # Tokenize text
        input_text = item.get("text", "") # Handle cases where 'text' might be missing
        if not input_text:
             logger.warning(f"Empty text found in item {idx} of {self.data_path}")

        input_ids = self._tokenize(input_text)

        # Get pad token ID from vocab, default to 0
        pad_token_id = self.vocab.get("<pad>", 0)

        # Truncate or pad to max_length
        seq_len = len(input_ids)
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
        elif seq_len < self.max_length:
            input_ids = input_ids + [pad_token_id] * (self.max_length - seq_len)

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != pad_token_id).long() # Use long for consistency

        # Prepare output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # For causal language modeling
        }

        # Add image if multimodal
        if self.multimodal and "image" in item:
             if self.image_dir is None and not os.path.isabs(item["image"]):
                  logger.warning(f"Multimodal is True but image_dir is not set and image path '{item['image']}' is not absolute. Cannot reliably load image for item {idx}.")
             else:
                image_path = item["image"]
                pixel_values = self._load_image(image_path)
                output["pixel_values"] = pixel_values

        return output


class ApertisTrainer:
    """High-performance trainer implementation for Apertis models."""

    def __init__(
        self,
        model: ApertisForCausalLM,
        train_dataset: ApertisDataset,
        val_dataset: Optional[ApertisDataset] = None,
        output_dir: str = "output",
        batch_size: int = 4,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        wandb_project: str = "apertis",
        wandb_run_name: Optional[str] = None,
        fp16: bool = True,
        device: Optional[str] = None,
        checkpoint_steps: int = 1000,
        iteration_checkpoint_steps: int = 0,
        gpu_memory_fraction: float = 0.7,
        use_gradient_checkpointing: bool = True,
        eval_every_n_epochs: int = 1, # Validate after every N epochs (0 to disable)
        dynamic_batch_sizing: bool = True,
        gpu_ids: Optional[List[int]] = None,
        distributed_training: bool = False,
        local_rank: int = -1,
    ):
        """
        Initialize the trainer.

        Args:
            model: The model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            output_dir: Directory to save outputs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            num_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for learning rate scheduler
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            use_wandb: Whether to use Weights & Biases for logging
            wandb_project: Weights & Biases project name
            wandb_run_name: Weights & Biases run name
            fp16: Whether to use mixed precision training
            device: Device to use for training
            checkpoint_steps: Save checkpoint every N global steps (0 to disable)
            iteration_checkpoint_steps: Save checkpoint every N iterations within an epoch (0 to disable)
            gpu_memory_fraction: Fraction of GPU memory to use
            use_gradient_checkpointing: Whether to use gradient checkpointing
            eval_every_n_epochs: Validate model every N epochs (0 to disable epoch-based validation)
            dynamic_batch_sizing: Whether to dynamically adjust batch size
            gpu_ids: List of GPU IDs to use for training (e.g., [0, 1, 2])
            distributed_training: Whether to use distributed training across multiple GPUs
            local_rank: Local rank for distributed training
        """
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

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set up distributed training if requested
        self.world_size = 1
        self.is_main_process = True

        if self.distributed_training:
            if self.local_rank == -1:
                # Auto-detect local_rank if not provided
                if 'LOCAL_RANK' in os.environ:
                    self.local_rank = int(os.environ['LOCAL_RANK'])
                else:
                    self.local_rank = 0

            # Initialize the distributed process group
            if not dist.is_initialized():
                if torch.cuda.is_available():
                    dist.init_process_group(backend='nccl')
                else:
                    dist.init_process_group(backend='gloo')

            self.world_size = dist.get_world_size()
            self.is_main_process = self.local_rank == 0

            logger.info(f"Distributed training enabled. World size: {self.world_size}, Local rank: {self.local_rank}")

        # Set device based on configuration
        if self.distributed_training:
            # In distributed mode, use the local rank to determine device
            self.device = torch.device(f"cuda:{self.local_rank}" if torch.cuda.is_available() else "cpu")
        elif self.gpu_ids and len(self.gpu_ids) > 0 and torch.cuda.is_available():
            # Use the first specified GPU if multiple are provided but distributed training is not enabled
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        elif device is not None:
            # Use the specified device
            self.device = torch.device(device)
        else:
            # Default to first available CUDA device or CPU
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Configure PyTorch CUDA memory allocation
        if torch.cuda.is_available():
            # Set environment variable to avoid memory fragmentation
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

            # Clear CUDA cache before starting
            torch.cuda.empty_cache()

            # Limit GPU memory usage if specified
            if self.gpu_memory_fraction < 1.0:
                try:
                    # Get total GPU memory for the current device
                    device_idx = self.device.index if self.device.type == 'cuda' else 0
                    if device_idx is not None and torch.cuda.is_available():
                        total_memory = torch.cuda.get_device_properties(device_idx).total_memory
                        # Calculate reserved memory
                        reserved_memory = int(total_memory * (1 - self.gpu_memory_fraction))

                        # Log memory information
                        logger.info(f"Total GPU memory (device {device_idx}): {total_memory / 1024**3:.2f} GiB")
                        logger.info(f"Reserving {reserved_memory / 1024**3:.2f} GiB for system")
                        logger.info(f"Available for training: {(total_memory - reserved_memory) / 1024**3:.2f} GiB")
                except Exception as e:
                    logger.warning(f"Failed to get GPU memory info: {e}")

        logger.info(f"Using device: {self.device}")

        # Enable gradient checkpointing if requested (reduces memory usage)
        if self.use_gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            logger.info("Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable()

        # Move model to device
        self.model.to(self.device)

        # Set up distributed model if using distributed training
        if self.distributed_training:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=False # Set to True if you encounter issues with unused parameters
            )
        # Set up DataParallel if using multiple GPUs without distributed training
        elif self.gpu_ids and len(self.gpu_ids) > 1 and torch.cuda.is_available():
            self.model = nn.DataParallel(self.model, device_ids=self.gpu_ids)
            logger.info(f"Using DataParallel with GPUs: {self.gpu_ids}")

        # Create data loaders with appropriate batch size
        self._create_dataloaders()

        # Set up optimizer with weight decay
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': self.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Set up learning rate scheduler
        num_training_steps = len(self.train_dataloader) // self.gradient_accumulation_steps * self.num_epochs
        # Ensure num_training_steps is at least 1
        num_training_steps = max(1, num_training_steps)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=num_training_steps,
            pct_start=0.1, # Adjust pct_start if needed
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        # Set up mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)

        # Initialize Weights & Biases if requested
        if self.use_wandb and self.is_main_process:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "num_epochs": self.num_epochs,
                    "warmup_steps": self.warmup_steps,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                    "max_grad_norm": self.max_grad_norm,
                    "fp16": self.fp16,
                    "device": str(self.device),
                    "distributed_training": self.distributed_training,
                    "world_size": self.world_size,
                    "gpu_ids": self.gpu_ids,
                    "eval_every_n_epochs": self.eval_every_n_epochs,
                    "model_config": self.model.config.to_dict() # Log model config
                }
            )

    def _create_dataloaders(self):
        """Create data loaders for training and validation."""
        # Set up samplers for distributed training
        train_sampler = None
        val_sampler = None

        if self.distributed_training:
            train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True
            )
            if self.val_dataset is not None:
                val_sampler = DistributedSampler(
                    self.val_dataset,
                    num_replicas=self.world_size,
                    rank=self.local_rank,
                    shuffle=False
                )

        # Create data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=4, # Adjust based on system capabilities
            pin_memory=True,
            drop_last=self.distributed_training # Drop last incomplete batch in distributed mode
        )

        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size, # Consider a larger eval batch size if memory allows
                shuffle=False,
                sampler=val_sampler,
                num_workers=4, # Adjust based on system capabilities
                pin_memory=True,
                drop_last=False
            )
        else:
            self.val_dataloader = None

    def train(self):
        """Train the model."""
        logger.info("Starting training")

        # Track best validation loss
        best_val_loss = float('inf')
        global_step = 0

        # Training loop
        for epoch in range(self.num_epochs):
            # Set epoch for distributed sampler
            if self.distributed_training and hasattr(self.train_dataloader, 'sampler') and hasattr(self.train_dataloader.sampler, 'set_epoch'):
                self.train_dataloader.sampler.set_epoch(epoch)

            # Training
            self.model.train()
            epoch_loss = 0.0
            batches_processed_this_epoch = 0

            # Progress bar for training
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.is_main_process
            )

            for step, batch in enumerate(self.train_dataloader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        # Fix for tuple output format - extract loss from first element of tuple
                        if isinstance(outputs, tuple):
                            loss = outputs[0]  # First element of the tuple is the loss
                        else:
                            loss = outputs.loss # Assuming 'loss' attribute if not tuple

                        if loss is None:
                             logger.warning(f"Skipping batch {step} in epoch {epoch+1} due to None loss.")
                             progress_bar.update(1) # Still update progress bar
                             continue

                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps

                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()

                    # Accumulate loss for epoch average (using the scaled loss)
                    epoch_loss += loss.item() # Accumulate the scaled loss
                    batches_processed_this_epoch += 1

                    # Update weights if gradient accumulation steps reached
                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                        # Clip gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                        # Update weights
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step() # Step scheduler after optimizer step
                        self.optimizer.zero_grad(set_to_none=True) # More efficient zeroing

                        # Increment global step
                        global_step += 1

                        # Calculate the actual loss for logging (unscale accumulated loss)
                        current_train_loss = (epoch_loss / batches_processed_this_epoch) * self.gradient_accumulation_steps

                        # Log to Weights & Biases
                        if self.use_wandb and self.is_main_process:
                            log_data = {
                                "train/loss": current_train_loss,
                                "train/learning_rate": self.scheduler.get_last_lr()[0],
                                "train/epoch": epoch + (step + 1) / len(self.train_dataloader),
                                "train/global_step": global_step,
                            }
                            # Log GPU memory usage if available
                            if torch.cuda.is_available():
                                log_data["train/gpu_mem_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2 # MB
                                log_data["train/gpu_mem_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2 # MB
                            wandb.log(log_data, step=global_step) # Log against global_step

                        # Save checkpoint if needed (based on global steps)
                        if self.checkpoint_steps > 0 and global_step % self.checkpoint_steps == 0 and self.is_main_process:
                            self.save_checkpoint(f"step-{global_step}")

                        # Reset epoch loss tracking after optimizer step
                        epoch_loss = 0.0
                        batches_processed_this_epoch = 0

                        # Update progress bar postfix with the actual (unscaled) loss
                        progress_bar.set_postfix({"loss": f"{current_train_loss:.4f}"})


                    # Save checkpoint if needed (based on iterations within epoch)
                    if self.iteration_checkpoint_steps > 0 and (step + 1) % self.iteration_checkpoint_steps == 0 and self.is_main_process:
                        self.save_checkpoint(f"epoch{epoch+1}-iter{step+1}")

                    # Update progress bar
                    progress_bar.update(1)
                    # Update postfix less frequently if needed, or keep it updated per batch with scaled loss
                    # progress_bar.set_postfix({"loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}"})


                except Exception as e:
                    logger.error(f"Error processing batch {step} in epoch {epoch+1}: {e}", exc_info=True)

                    # Try to recover with dynamic batch sizing
                    if self.dynamic_batch_sizing and "CUDA out of memory" in str(e) and self.batch_size > 1:
                        # Reduce batch size
                        self.batch_size = max(1, self.batch_size // 2)
                        logger.warning(f"CUDA OOM detected. Reducing batch size to {self.batch_size}")

                        # Clear CUDA cache
                        torch.cuda.empty_cache()

                        # Recreate data loaders with new batch size
                        self._create_dataloaders()

                        # Skip to next epoch - restart progress bar for this epoch
                        logger.warning("Restarting current epoch due to OOM and batch size reduction.")
                        progress_bar.close() # Close old bar
                        # Re-initialize epoch loop state if necessary (e.g., reset epoch_loss)
                        epoch_loss = 0.0
                        batches_processed_this_epoch = 0 # Reset counter
                        # Create new progress bar for the remainder of the epoch
                        progress_bar = tqdm(
                            total=len(self.train_dataloader),
                            desc=f"Epoch {epoch+1}/{self.num_epochs} (restarted)",
                            initial=step+1, # Start from where we left off
                            disable=not self.is_main_process
                        )
                        continue # Continue with the next batch in the current epoch
                    else:
                        # If not OOM or cannot reduce batch size further, raise the error
                        logger.error("Unrecoverable error during training step.")
                        raise e

            # Close progress bar at the end of epoch
            progress_bar.close()

            # Calculate final epoch average loss (handle potential leftover loss)
            final_epoch_loss = float('nan')
            total_steps_processed = step + 1 # total batches iterated through
            if total_steps_processed > 0:
                 # Average the loss over all batches processed in the epoch
                 # Note: This might be slightly less accurate if OOM happened mid-accumulation cycle
                 # A more precise way would involve tracking total loss before scaling, but this is simpler.
                 # If the last step wasn't an optimizer step, average the remaining accumulated loss
                 if batches_processed_this_epoch > 0:
                     final_epoch_loss = (epoch_loss / batches_processed_this_epoch) * self.gradient_accumulation_steps
                 else:
                      # Use the last logged loss if available
                      try: final_epoch_loss = current_train_loss
                      except NameError: pass # Keep NaN if no loss was ever computed

                 if not np.isnan(final_epoch_loss):
                      logger.info(f"Epoch {epoch+1}/{self.num_epochs} completed - Approx. Average Train Loss: {final_epoch_loss:.4f}")
                 else:
                      logger.warning(f"Epoch {epoch+1}/{self.num_epochs} completed - Could not calculate average loss.")
            else:
                 logger.warning(f"Epoch {epoch+1}/{self.num_epochs} completed - No batches processed.")


            # Evaluate at the end of each Nth epoch
            if self.val_dataloader is not None and self.eval_every_n_epochs > 0 and (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss = self.evaluate()
                if not np.isinf(val_loss): # Check if validation was successful
                     logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Validation Loss: {val_loss:.4f}")

                     # Log validation loss
                     if self.use_wandb and self.is_main_process:
                         wandb.log({
                             "eval/val_loss": val_loss,
                             "eval/epoch": epoch + 1,
                         }, step=global_step) # Log against global_step

                     # Save best model based on validation loss
                     if val_loss < best_val_loss and self.is_main_process:
                         best_val_loss = val_loss
                         self.save_checkpoint(f"best_model")
                         logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
                else:
                     logger.warning(f"Epoch {epoch+1}/{self.num_epochs} - Validation skipped or failed.")


            # Save checkpoint at the end of each epoch
            if self.is_main_process:
                self.save_checkpoint(f"epoch-{epoch+1}")

        # Save final model
        if self.is_main_process:
            self.save_checkpoint("final")

        logger.info("Training completed")

        # Close Weights & Biases
        if self.use_wandb and self.is_main_process:
            wandb.finish()

    def evaluate(self):
        """Evaluate the model on the validation set."""
        if self.val_dataloader is None:
             logger.warning("Validation dataset not provided. Skipping evaluation.")
             return float('inf') # Return infinity if no validation set

        logger.info("Evaluating model...")

        # Set model to evaluation mode
        self.model.eval()

        # Track validation loss
        total_val_loss = 0.0
        num_val_batches = 0

        # Progress bar for validation
        progress_bar = tqdm(
            total=len(self.val_dataloader),
            desc="Validation",
            disable=not self.is_main_process
        )

        # Evaluation loop
        with torch.no_grad():
            for batch in self.val_dataloader:
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

                    # Forward pass with mixed precision context (though grads aren't needed)
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        # Fix for tuple output format - extract loss from first element of tuple
                        if isinstance(outputs, tuple):
                            loss = outputs[0]
                        else:
                            loss = outputs.loss

                        if loss is not None:
                             total_val_loss += loss.item()
                             num_val_batches += 1
                        else:
                             logger.warning("Encountered None loss during validation.")

                except Exception as e:
                    logger.error(f"Error during validation batch: {e}", exc_info=True)

                # Update progress bar
                progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

        # Calculate average loss
        if num_val_batches > 0:
            avg_val_loss = total_val_loss / num_val_batches
            logger.info(f"Validation finished - Average Loss: {avg_val_loss:.4f}")
        else:
             logger.warning("Validation finished - No batches were successfully processed.")
             avg_val_loss = float('inf')

        # Switch back to training mode before exiting evaluate
        self.model.train()

        return avg_val_loss

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get model to save (unwrap DDP or DataParallel if needed)
        model_to_save = self.model.module if isinstance(self.model, (nn.DataParallel, DDP)) else self.model

        # Ensure model_to_save has a config attribute
        if not hasattr(model_to_save, 'config'):
             logger.error("Model object does not have a 'config' attribute. Cannot save config.json.")
             # Fallback: try to get config from the trainer's model if possible
             if hasattr(self.model, 'config'):
                  model_config = self.model.config
             else:
                  logger.error("Could not find config on trainer's model either. Skipping config save.")
                  model_config = None
        else:
             model_config = model_to_save.config

        # Save model state dict
        model_save_path = os.path.join(checkpoint_dir, "model.pt")
        try:
            torch.save(model_to_save.state_dict(), model_save_path)
        except Exception as e:
             logger.error(f"Failed to save model state dict to {model_save_path}: {e}")
             return # Don't proceed if model saving fails

        # Save configuration if available
        if model_config is not None and hasattr(model_config, 'to_dict'):
            config_save_path = os.path.join(checkpoint_dir, "config.json")
            try:
                with open(config_save_path, "w") as f:
                    json.dump(model_config.to_dict(), f, indent=2)
            except Exception as e:
                 logger.error(f"Failed to save config.json to {config_save_path}: {e}")

        logger.info(f"Checkpoint saved to {checkpoint_dir}")


def get_available_gpus() -> List[Dict[str, Any]]:
    """Get information about available GPUs."""
    gpu_info = []

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "id": i,
                "name": props.name,
                "total_memory": props.total_memory / (1024**3),  # Convert to GB
                "compute_capability": f"{props.major}.{props.minor}"
            })

    return gpu_info


def train_from_config(config_path: str):
    """Train a model from a configuration file."""
    # Load configuration
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except FileNotFoundError:
         logger.error(f"Configuration file not found: {config_path}")
         return
    except json.JSONDecodeError as e:
         logger.error(f"Error decoding JSON from configuration file {config_path}: {e}")
         return

    # Extract configuration sections
    data_config = config.get("data_config", {})
    model_config = config.get("model_config", {})
    training_config = config.get("training_config", {})

    # --- Determine Vocabulary Size ---
    tokenizer_path = data_config.get("tokenizer_path")
    if not tokenizer_path:
        logger.error("`tokenizer_path` not specified in data_config.")
        return
    try:
        # Load the vocabulary solely to get its size for model creation
        actual_vocab_size = _load_vocabulary_size(tokenizer_path)
        logger.info(f"Determined vocabulary size from {tokenizer_path}: {actual_vocab_size}")
    except Exception as e:
        logger.error(f"Failed to load vocabulary to determine size: {e}")
        return

    # --- Create Datasets ---
    # Pass the determined vocab size to the dataset for consistency checks
    try:
        train_dataset = ApertisDataset(
            data_path=data_config.get("train_data_path"),
            tokenizer_path=tokenizer_path,
            model_vocab_size=actual_vocab_size, # Pass the correct size
            max_length=data_config.get("max_length", 512),
            multimodal=model_config.get("multimodal", False), # Use model_config value
            image_dir=data_config.get("image_dir"),
            image_size=data_config.get("image_size", 224),
        )

        val_dataset = None
        val_data_path = data_config.get("val_data_path")
        if val_data_path:
            val_dataset = ApertisDataset(
                data_path=val_data_path,
                tokenizer_path=tokenizer_path,
                model_vocab_size=actual_vocab_size, # Pass the correct size
                max_length=data_config.get("max_length", 512),
                multimodal=model_config.get("multimodal", False), # Use model_config value
                image_dir=data_config.get("image_dir"),
                image_size=data_config.get("image_size", 224),
            )
    except Exception as e:
         logger.error(f"Error creating datasets: {e}")
         return

    # --- Create Model ---
    # Use the actual_vocab_size determined from the file
    model_config_dict = {
        "vocab_size": actual_vocab_size, # Use the actual size
        "hidden_size": model_config.get("hidden_size", 768),
        "num_hidden_layers": model_config.get("num_hidden_layers", 12),
        "num_attention_heads": model_config.get("num_attention_heads", 12),
        "intermediate_size": model_config.get("intermediate_size", 3072),
        "use_expert_system": model_config.get("use_expert_system", False),
        "multimodal": model_config.get("multimodal", False),
        # Add other relevant ApertisConfig parameters from model_config if needed
        "image_size": model_config.get("image_size", 224),
        "vision_embed_dim": model_config.get("vision_embed_dim", 768),
        "vision_patch_size": model_config.get("vision_patch_size", 16),
        "vision_layers": model_config.get("vision_layers", 12),
        "vision_heads": model_config.get("vision_heads", 12),
    }

    try:
        model_config_obj = ApertisConfig(**model_config_dict)
        model = ApertisForCausalLM(model_config_obj)
        logger.info(f"Created ApertisForCausalLM model with config: {model_config_obj.to_dict()}")
    except Exception as e:
         logger.error(f"Error creating model with config {model_config_dict}: {e}")
         return

    # Get GPU configuration
    gpu_ids = training_config.get("gpu_ids", None)
    distributed_training = training_config.get("distributed_training", False)
    local_rank = training_config.get("local_rank", -1)

    # --- Create Trainer ---
    try:
        trainer = ApertisTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=training_config.get("output_dir", "output"),
            batch_size=training_config.get("batch_size", 4),
            learning_rate=training_config.get("learning_rate", 5e-5),
            weight_decay=training_config.get("weight_decay", 0.01),
            num_epochs=training_config.get("num_epochs", 3),
            warmup_steps=training_config.get("warmup_steps", 0),
            gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4),
            max_grad_norm=training_config.get("max_grad_norm", 1.0),
            use_wandb=training_config.get("use_wandb", False),
            wandb_project=training_config.get("wandb_project", "apertis"),
            wandb_run_name=training_config.get("wandb_run_name"),
            fp16=training_config.get("fp16", True),
            device=training_config.get("device"), # Let trainer determine default if None
            checkpoint_steps=training_config.get("checkpoint_steps", 0), # Default 0 unless specified
            iteration_checkpoint_steps=training_config.get("iteration_checkpoint_steps", 0), # Default 0 unless specified
            gpu_memory_fraction=training_config.get("gpu_memory_fraction", 0.7),
            use_gradient_checkpointing=training_config.get("use_gradient_checkpointing", True),
            eval_every_n_epochs=training_config.get("eval_every_n_epochs", 1),
            dynamic_batch_sizing=training_config.get("dynamic_batch_sizing", True),
            gpu_ids=gpu_ids,
            distributed_training=distributed_training,
            local_rank=local_rank,
        )
    except Exception as e:
         logger.error(f"Error initializing ApertisTrainer: {e}")
         return

    # --- Train Model ---
    try:
        logger.info(f"Starting training using config: {config_path}")
        trainer.train()
        logger.info(f"Finished training for config: {config_path}")
    except Exception as e:
         logger.error(f"Error during training: {e}", exc_info=True)


class YoloStyleTrainingPipeline:
    """YOLO-style training pipeline for Apertis models."""

    def __init__(
        self,
        config_path: str,
    ):
        """
        Initialize the training pipeline.

        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path

    def train(self):
        """Train the model."""
        train_from_config(self.config_path)