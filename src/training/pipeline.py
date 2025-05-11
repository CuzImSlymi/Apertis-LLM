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

class ApertisDataset(Dataset):
    """Dataset for training Apertis models."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_length: int = 512,
        multimodal: bool = False,
        image_dir: Optional[str] = None,
        image_size: int = 224,
        model_vocab_size: int = 32000,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file (jsonl format)
            tokenizer_path: Path to the tokenizer vocabulary file
            max_length: Maximum sequence length
            multimodal: Whether to include image data
            image_dir: Directory containing images (required if multimodal=True)
            image_size: Size of images (height and width)
            model_vocab_size: The vocabulary size of the model (to ensure token IDs are in bounds)
        """
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.max_length = max_length
        self.multimodal = multimodal
        self.image_dir = image_dir
        self.image_size = image_size
        self.model_vocab_size = model_vocab_size
        
        # Load data
        self.data = self._load_data()
        
        # Load vocabulary
        self.vocab = self._load_vocabulary()
        
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
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append(item)
        logger.info(f"Loaded {len(data)} examples from {self.data_path}")
        return data
    
    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary from json file."""
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
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization by splitting on spaces and mapping to vocabulary."""
        tokens = []
        vocab_size = 0
        
        # Find the maximum vocabulary index to determine actual vocab size
        for _, idx in self.vocab.items():
            if isinstance(idx, int) and idx > vocab_size:
                vocab_size = idx
        
        # Add 1 to account for 0-indexing
        vocab_size += 1
        
        # Get model's expected vocab size (default to a large number if not available)
        model_vocab_size = getattr(self, "model_vocab_size", 200000)
        
        # Use the smaller of the two to ensure we don't exceed model's vocabulary size
        effective_vocab_size = min(vocab_size, model_vocab_size)
        
        # Special token IDs
        unk_token_id = self.vocab.get("<unk>", 3)
        
        for word in text.split():
            if word in self.vocab:
                token_id = self.vocab[word]
                # Ensure token ID is within vocabulary bounds
                if token_id >= effective_vocab_size:
                    token_id = unk_token_id
                tokens.append(token_id)
            else:
                tokens.append(unk_token_id)  # Default to <unk> token
        
        return tokens
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        try:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
            return self.image_transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a blank image as fallback
            blank = torch.zeros(3, self.image_size, self.image_size)
            return blank
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        item = self.data[idx]
        
        # Tokenize text
        input_text = item["text"]
        input_ids = self._tokenize(input_text)
        
        # Truncate or pad to max_length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        else:
            input_ids = input_ids + [self.vocab.get("<pad>", 0)] * (self.max_length - len(input_ids))
        
        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.vocab.get("<pad>", 0)).float()
        
        # Prepare output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # For causal language modeling
        }
        
        # Add image if multimodal
        if self.multimodal and "image" in item:
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
            checkpoint_steps: Save checkpoint every N global steps
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
                find_unused_parameters=False
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
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=num_training_steps,
            pct_start=0.1,
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
            num_workers=4,
            pin_memory=True,
        )
        
        if self.val_dataset is not None:
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True,
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
            
            # Progress bar for training
            progress_bar = tqdm(
                total=len(self.train_dataloader),
                desc=f"Epoch {epoch+1}/{self.num_epochs}",
                disable=not self.is_main_process
            )
            
            for step, batch in enumerate(self.train_dataloader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass with mixed precision
                    with torch.amp.autocast('cuda', enabled=self.fp16):
                        outputs = self.model(**batch)
                        # Fix for tuple output format - extract loss from first element of tuple
                        if isinstance(outputs, tuple):
                            loss = outputs[0]  # First element of the tuple is the loss
                        else:
                            loss = outputs.loss
                        
                        # Scale loss for gradient accumulation
                        loss = loss / self.gradient_accumulation_steps
                    
                    # Backward pass with gradient scaling
                    self.scaler.scale(loss).backward()
                    
                    # Update weights if gradient accumulation steps reached
                    if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(self.train_dataloader):
                        # Clip gradients
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                        # Update weights
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.scheduler.step()
                        self.optimizer.zero_grad()
                        
                        # Increment global step
                        global_step += 1
                        
                        # Log to Weights & Biases
                        if self.use_wandb and self.is_main_process:
                            wandb.log({
                                "train_loss": loss.item() * self.gradient_accumulation_steps,
                                "learning_rate": self.scheduler.get_last_lr()[0],
                                "epoch": epoch + step / len(self.train_dataloader),
                                "global_step": global_step,
                            })
                        
                        # Save checkpoint if needed (based on global steps)
                        if self.checkpoint_steps > 0 and global_step % self.checkpoint_steps == 0 and self.is_main_process:
                            self.save_checkpoint(f"step-{global_step}")
                    
                    # Save checkpoint if needed (based on iterations within epoch)
                    if self.iteration_checkpoint_steps > 0 and (step + 1) % self.iteration_checkpoint_steps == 0 and self.is_main_process:
                        self.save_checkpoint(f"epoch{epoch+1}-iter{step+1}")
                    
                    # Update progress bar
                    progress_bar.update(1)
                    progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})
                    
                    # Accumulate loss for epoch average
                    epoch_loss += loss.item() * self.gradient_accumulation_steps
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    
                    # Try to recover with dynamic batch sizing
                    if self.dynamic_batch_sizing and "CUDA out of memory" in str(e) and self.batch_size > 1:
                        # Reduce batch size
                        self.batch_size = max(1, self.batch_size // 2)
                        logger.warning(f"CUDA OOM detected. Reducing batch size to {self.batch_size}")
                        
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
                        
                        # Recreate data loaders with new batch size
                        self._create_dataloaders()
                        
                        # Skip to next epoch
                        break
            
            # Close progress bar
            progress_bar.close()
            
            # Calculate epoch average loss
            epoch_loss /= len(self.train_dataloader)
            logger.info(f"Epoch {epoch+1}/{self.num_epochs} - Average loss: {epoch_loss:.4f}")
            
            # Evaluate at the end of each Nth epoch
            if self.val_dataloader is not None and self.eval_every_n_epochs > 0 and (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss = self.evaluate()
                
                # Log validation loss
                if self.use_wandb and self.is_main_process:
                    wandb.log({
                        "val_loss": val_loss,
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    })
                
                # Save best model
                if val_loss < best_val_loss and self.is_main_process:
                    best_val_loss = val_loss
                    self.save_checkpoint(f"best_model")
                    logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
            
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
        logger.info("Evaluating model")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Track validation loss
        val_loss = 0.0
        
        # Progress bar for validation
        progress_bar = tqdm(
            total=len(self.val_dataloader),
            desc="Validation",
            disable=not self.is_main_process
        )
        
        # Evaluation loop
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                # Fix for tuple output format - extract loss from first element of tuple
                if isinstance(outputs, tuple):
                    loss = outputs[0]  # First element of the tuple is the loss
                else:
                    loss = outputs.loss
                
                # Accumulate loss
                val_loss += loss.item()
                
                # Update progress bar
                progress_bar.update(1)
        
        # Close progress bar
        progress_bar.close()
        
        # Calculate average loss
        val_loss /= len(self.val_dataloader)
        logger.info(f"Validation loss: {val_loss:.4f}")
        
        # Switch back to training mode (important if evaluate is called mid-training loop, though not strictly necessary here as it's at epoch end)
        self.model.train()
        
        return val_loss
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Get model to save (unwrap DDP or DataParallel if needed)
        if isinstance(self.model, (nn.DataParallel, DDP)):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        
        # Save model state dict
        torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save configuration
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(model_to_save.config.to_dict(), f, indent=2)
        
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
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract configuration sections
    data_config = config.get("data_config", {})
    model_config = config.get("model_config", {})
    training_config = config.get("training_config", {})
    
    # Create datasets
    train_dataset = ApertisDataset(
        data_path=data_config.get("train_data_path"),
        tokenizer_path=data_config.get("tokenizer_path"),
        max_length=data_config.get("max_length", 512),
        multimodal=data_config.get("multimodal", False),
        image_dir=data_config.get("image_dir"),
        image_size=data_config.get("image_size", 224),
    )
    
    val_dataset = None
    if "val_data_path" in data_config and data_config["val_data_path"]:
        val_dataset = ApertisDataset(
            data_path=data_config.get("val_data_path"),
            tokenizer_path=data_config.get("tokenizer_path"),
            max_length=data_config.get("max_length", 512),
            multimodal=data_config.get("multimodal", False),
            image_dir=data_config.get("image_dir"),
            image_size=data_config.get("image_size", 224),
        )
    
    # Create model
    model_config_obj = ApertisConfig(
        vocab_size=model_config.get("vocab_size", 32000),
        hidden_size=model_config.get("hidden_size", 768),
        num_hidden_layers=model_config.get("num_hidden_layers", 12),
        num_attention_heads=model_config.get("num_attention_heads", 12),
        intermediate_size=model_config.get("intermediate_size", 3072),
        use_expert_system=model_config.get("use_expert_system", False),
        multimodal=model_config.get("multimodal", False),
    )
    model = ApertisForCausalLM(model_config_obj)
    
    # Get GPU configuration
    gpu_ids = training_config.get("gpu_ids", None)
    distributed_training = training_config.get("distributed_training", False)
    local_rank = training_config.get("local_rank", -1)
    
    # Create trainer
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
        device=training_config.get("device"),
        checkpoint_steps=training_config.get("checkpoint_steps", 1000),
        iteration_checkpoint_steps=training_config.get("iteration_checkpoint_steps", 0),
        gpu_memory_fraction=training_config.get("gpu_memory_fraction", 0.7),
        use_gradient_checkpointing=training_config.get("use_gradient_checkpointing", True),
        eval_every_n_epochs=training_config.get("eval_every_n_epochs", 1),
        dynamic_batch_sizing=training_config.get("dynamic_batch_sizing", True),
        gpu_ids=gpu_ids,
        distributed_training=distributed_training,
        local_rank=local_rank,
    )
    
    # Train model
    trainer.train()


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