# /home/ubuntu/ApertisAI_Project/Apertis AI_/src/training/pipeline.py
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
import threading # Added for stop_event
import gradio as gr # Added for progress_callback type hint

import sys
import os

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.core import ApertisConfig, ApertisForCausalLM

# Import distributed training modules
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        model_vocab_size: int = 32000, # Added model_vocab_size
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
        self.model_vocab_size = model_vocab_size # Store model vocab size

        # Load data
        self.data = self._load_data()

        # Load vocabulary
        self.vocab = self._load_vocabulary()

        # Set up image transformation if multimodal
        if self.multimodal:
            if not image_dir or not os.path.isdir(image_dir):
                 raise ValueError("Image directory must be provided and valid for multimodal dataset.")
            self.image_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def _load_data(self) -> List[Dict]:
        """Load data from jsonl file."""
        data = []
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        # Basic validation: check for \'text\' key
                        if "text" not in item:
                            logger.warning(f"Skipping line {i+1} in {self.data_path}: missing \'text\' key.")
                            continue
                        if self.multimodal and "image" not in item:
                            logger.warning(f"Skipping line {i+1} in {self.data_path}: missing \'image\' key for multimodal data.")
                            continue
                        data.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping line {i+1} in {self.data_path}: invalid JSON.")
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        logger.info(f"Loaded {len(data)} examples from {self.data_path}")
        return data

    def _load_vocabulary(self) -> Dict[str, int]:
        """Load vocabulary from json file."""
        try:
            with open(self.tokenizer_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
        except FileNotFoundError:
             logger.error(f"Vocabulary file not found: {self.tokenizer_path}")
             raise
        except json.JSONDecodeError:
             logger.error(f"Failed to decode JSON from vocabulary file: {self.tokenizer_path}")
             raise

        # Handle different vocabulary formats
        if isinstance(vocab_data, dict):
            if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                vocab = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                logger.info(f"Converted list-based vocabulary to dictionary format with {len(vocab)} tokens")
            elif all(isinstance(k, str) and isinstance(v, int) for k, v in vocab_data.items()):
                vocab = vocab_data
            elif "model" in vocab_data and "vocab" in vocab_data["model"] and isinstance(vocab_data["model"]["vocab"], dict):
                 logger.info("Extracting vocab dict from \'model.vocab\' structure in JSON")
                 vocab = {k: int(v) for k, v in vocab_data["model"]["vocab"].items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
            else:
                # Attempt to handle simple {token: id} dict even if values aren\'t strictly int
                try:
                    vocab = {str(k): int(v) for k, v in vocab_data.items()}
                    logger.info(f"Loaded vocabulary dictionary with {len(vocab)} tokens (converted values to int)." )
                except (ValueError, TypeError):
                     raise ValueError("Unsupported vocabulary dictionary format in JSON file.")
        else:
            raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")

        # Add standard special tokens if missing (using high indices to avoid collision)
        # This is a simple approach; a proper tokenizer build process is better.
        next_id = max(vocab.values()) + 1 if vocab else 0
        added_tokens = []
        for token in ["<pad>", "<unk>", "<bos>", "<eos>"]:
             if token not in vocab:
                 vocab[token] = next_id
                 added_tokens.append(token)
                 next_id += 1
        if added_tokens:
             logger.warning(f"Added missing special tokens to vocab: {added_tokens}. Consider rebuilding vocab properly.")

        logger.info(f"Loaded vocabulary with {len(vocab)} tokens from {self.tokenizer_path}")
        return vocab

    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization by splitting on spaces and mapping to vocabulary."""
        tokens = []
        unk_token_id = self.vocab.get("<unk>") # Get UNK id from vocab
        if unk_token_id is None:
            # Fallback if <unk> is somehow missing (should be added by pipeline init)
            logger.error("'<unk>' token not found in loaded vocabulary! Falling back to PAD ID.")
            unk_token_id = self.vocab.get("<pad>", 0) # Use PAD ID (default 0) as fallback
        pad_token_id = self.vocab.get("<pad>", 0) # Default PAD id

        # Use the model\'s expected vocab size for bounds checking
        effective_vocab_size = len(self.vocab)

        for word in text.split():
            token_id = self.vocab.get(word, unk_token_id)
            # Ensure token ID is within the *model's* vocabulary bounds
            if token_id >= self.model_vocab_size:
                logger.warning(f"Token \t\"{word}\" (ID: {token_id}) is outside model vocab size ({self.model_vocab_size}). Mapping to UNK ({unk_token_id}).")
                token_id = unk_token_id
            tokens.append(token_id)

        return tokens

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image."""
        full_path = os.path.join(self.image_dir, image_path)
        try:
            image = Image.open(full_path).convert("RGB")
            return self.image_transform(image)
        except FileNotFoundError:
            logger.error(f"Image file not found: {full_path}")
            # Return a dummy tensor or raise error? Raising error is safer.
            raise
        except Exception as e:
            logger.error(f"Error loading image {full_path}: {e}")
            raise IOError(f"Failed to load image {image_path}: {e}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example from the dataset."""
        item = self.data[idx]
        pad_token_id = self.vocab.get("<pad>", 0)

        # Tokenize text
        input_text = item["text"]
        input_ids = self._tokenize(input_text)

        # Truncate or pad to max_length
        seq_len = len(input_ids)
        if seq_len > self.max_length:
            input_ids = input_ids[:self.max_length]
            seq_len = self.max_length
        else:
            padding = [pad_token_id] * (self.max_length - seq_len)
            input_ids = input_ids + padding

        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long) # Use long for consistency
        attention_mask[:seq_len] = 1

        # Prepare output dictionary
        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # For causal language modeling
        }
        # For causal LM, labels should be shifted, and padding ignored
        # Shift labels: label for token i is token i+1
        output["labels"][:-1] = input_ids[1:].clone()
        # Set label for the last token and padding tokens to -100 to be ignored by loss function
        output["labels"][seq_len-1:] = -100

        # Add image if multimodal
        if self.multimodal and "image" in item:
            try:
                image_path = item["image"]
                pixel_values = self._load_image(image_path)
                output["pixel_values"] = pixel_values
            except Exception as e:
                 # If image loading fails, we might skip this item or return a flag
                 # For simplicity, let\'s log and potentially raise to skip the batch
                 logger.error(f"Skipping item {idx} due to image loading error: {e}")
                 # To make collate_fn skip this, we can return None or raise an error
                 # Returning None might be problematic for DataLoader. Let\'s raise.
                 raise RuntimeError(f"Failed to load image for item {idx}") from e

        return output


class YoloStyleTrainingPipeline: # Renamed from ApertisTrainer for clarity
    """High-performance training pipeline for Apertis models."""

    def __init__(
        self,
        config: ApertisConfig, # Pass config object directly
        train_data_path: str,
        vocab_path: str,
        output_dir: str = "output",
        val_data_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        batch_size: int = 4,
        learning_rate: float = 1e-5, # Reduced default LR
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        wandb_project: str = "apertis",
        wandb_run_name: Optional[str] = None,
        fp16: bool = True,
        device: Optional[str] = None, # Keep device option for non-distributed
        checkpoint_steps: int = 1000,
        iteration_checkpoint_steps: int = 0,
        gpu_memory_fraction: float = 0.7, # Keep for potential single GPU limit
        use_gradient_checkpointing: bool = True,
        eval_steps: int = 500,
        dynamic_batch_sizing: bool = True,
        gpu_ids: Optional[List[int]] = None,
        use_distributed: bool = False,
        local_rank: int = -1, # Keep for distributed setup
    ):
        """
        Initialize the training pipeline.
        Args:
            config: ApertisConfig object for the model.
            train_data_path: Path to training data (jsonl).
            vocab_path: Path to vocabulary file (json).
            output_dir: Directory to save outputs.
            val_data_path: Path to validation data (jsonl, optional).
            image_dir: Directory containing images (required if config.multimodal=True).
            batch_size: Batch size per GPU.
            learning_rate: Peak learning rate.
            weight_decay: Weight decay for optimizer.
            num_epochs: Number of training epochs.
            warmup_steps: Number of warmup steps for learning rate scheduler.
            gradient_accumulation_steps: Accumulate gradients over N steps.
            max_grad_norm: Maximum gradient norm for clipping.
            use_wandb: Use Weights & Biases for logging.
            wandb_project: W&B project name.
            wandb_run_name: W&B run name.
            fp16: Use mixed precision training (torch.amp).
            device: Explicit device selection (e.g., \'cpu\', \'cuda:0\'), overrides distributed/gpu_ids if set.
            checkpoint_steps: Save checkpoint every N global steps (0 to disable).
            iteration_checkpoint_steps: Save checkpoint every N iterations within an epoch (0 to disable).
            gpu_memory_fraction: (Not directly used for limiting, informational).
            use_gradient_checkpointing: Enable gradient checkpointing in the model.
            eval_steps: Evaluate model every N global steps (0 to disable).
            dynamic_batch_sizing: Attempt to reduce batch size on OOM (experimental).
            gpu_ids: List of GPU IDs for DataParallel (if use_distributed=False).
            use_distributed: Use DistributedDataParallel.
            local_rank: Local rank for distributed training (usually set by launcher).
        """
        self.config = config
        self.train_data_path = train_data_path
        self.vocab_path = vocab_path
        self.output_dir = output_dir
        self.val_data_path = val_data_path
        self.image_dir = image_dir
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
        self.fp16 = fp16 and torch.cuda.is_available() # Only enable if CUDA is available
        self.checkpoint_steps = checkpoint_steps
        self.iteration_checkpoint_steps = iteration_checkpoint_steps
        self.gpu_memory_fraction = gpu_memory_fraction
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.eval_steps = eval_steps
        self.dynamic_batch_sizing = dynamic_batch_sizing # Note: OOM handling is basic
        self.gpu_ids = gpu_ids
        self.use_distributed = use_distributed and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.local_rank = local_rank

        # --- Distributed Setup ---
        self.world_size = 1
        self.global_rank = 0
        self.is_main_process = True

        if self.use_distributed:
            if self.local_rank == -1:
                # Try to get rank from environment variables if not passed explicitly
                self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.global_rank = int(os.environ.get("RANK", 0))
                self.world_size = int(os.environ.get("WORLD_SIZE", 1))
            else:
                # Assume setup is handled externally if local_rank is provided
                self.global_rank = self.local_rank # Simple case if not using complex setup
                # self.world_size needs to be set correctly based on the launch config
                # For simplicity, assume world_size matches number of GPUs if gpu_ids provided
                self.world_size = len(gpu_ids) if gpu_ids else torch.cuda.device_count()

            if not dist.is_initialized():
                # Setup the process group if not already done
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                logger.info(f"Initializing distributed process group (backend: {backend}, rank: {self.global_rank}, world_size: {self.world_size})")
                try:
                    dist.init_process_group(backend=backend, rank=self.global_rank, world_size=self.world_size)
                except Exception as e:
                    logger.error(f"Failed to initialize distributed process group: {e}", exc_info=True)
                    raise RuntimeError("Distributed training setup failed.") from e

            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
            self.is_main_process = self.global_rank == 0
            logger.info(f"Distributed training enabled. Rank: {self.global_rank}, Local Rank: {self.local_rank}, World Size: {self.world_size}, Device: {self.device}")
        elif device: # Explicit device takes precedence over gpu_ids if not distributed
            self.device = torch.device(device)
            self.is_main_process = True
            logger.info(f"Using explicit device: {self.device}")
        elif torch.cuda.is_available():
            if gpu_ids and len(gpu_ids) == 1:
                self.device = torch.device(f"cuda:{gpu_ids[0]}")
            else:
                self.device = torch.device("cuda") # Default to cuda:0 if no specific single GPU ID
            self.is_main_process = True
            logger.info(f"Using single GPU device: {self.device}")
        else:
            self.device = torch.device("cpu")
            self.is_main_process = True
            logger.info("Using CPU device.")

        # --- Initialize W&B (only on main process) ---
        if self.use_wandb and self.is_main_process:
            try:
                wandb.init(
                    project=self.wandb_project,
                    name=self.wandb_run_name,
                    config=self.config.to_dict() # Log model config
                )
                logger.info("Weights & Biases initialized.")
            except Exception as e:
                logger.error(f"Failed to initialize W&B: {e}")
                self.use_wandb = False # Disable W&B if init fails

        # --- Create Output Directory (only on main process) ---
        if self.is_main_process:
            os.makedirs(self.output_dir, exist_ok=True)
            # Save config to output directory
            config_save_path = os.path.join(self.output_dir, "config.json")
            try:
                 self.config.save_pretrained(self.output_dir)
                 logger.info(f"Model configuration saved to {config_save_path}")
            except Exception as e:
                 logger.error(f"Failed to save config to {config_save_path}: {e}")

        # --- Load Vocabulary (using dataset's logic to include special tokens) ---
        logger.info("Loading vocabulary and ensuring special tokens...")
        try:
            # --- Start: Logic similar to ApertisDataset._load_vocabulary ---
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)

            if isinstance(vocab_data, dict):
                if "tokens" in vocab_data and isinstance(vocab_data["tokens"], list):
                    vocab = {token: idx for idx, token in enumerate(vocab_data["tokens"])}
                    logger.info(f"Converted list-based vocabulary to dictionary format with {len(vocab)} tokens")
                elif all(isinstance(k, str) and isinstance(v, int) for k, v in vocab_data.items()):
                    vocab = vocab_data
                elif "model" in vocab_data and "vocab" in vocab_data["model"] and isinstance(vocab_data["model"]["vocab"], dict):
                     logger.info("Extracting vocab dict from 'model.vocab' structure in JSON")
                     vocab = {k: int(v) for k, v in vocab_data["model"]["vocab"].items() if isinstance(v, (int, float, str)) and str(v).isdigit()}
                else:
                    try:
                        vocab = {str(k): int(v) for k, v in vocab_data.items()}
                        logger.info(f"Loaded vocabulary dictionary with {len(vocab)} tokens (converted values to int).")
                    except (ValueError, TypeError):
                         raise ValueError("Unsupported vocabulary dictionary format in JSON file.")
            else:
                raise ValueError(f"Unsupported vocabulary format: {type(vocab_data)}")

            # Add standard special tokens if missing
            next_id = max(vocab.values()) + 1 if vocab else 0
            added_tokens = []
            for token in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                 if token not in vocab:
                     vocab[token] = next_id
                     added_tokens.append(token)
                     next_id += 1
            if added_tokens:
                 logger.warning(f"Added missing special tokens to vocab during pipeline init: {added_tokens}. Consider rebuilding vocab properly.")
            # --- End: Logic similar to ApertisDataset._load_vocabulary ---

            # Determine the effective vocabulary size based on the highest token ID
            if not vocab:
                 raise ValueError("Vocabulary is empty after loading and processing.")
            max_token_id = max(vocab.values())
            actual_vocab_size = max_token_id + 1 # Use max ID + 1 for embedding size
            logger.info(f"Loaded and processed vocabulary with {len(vocab)} unique tokens from {self.vocab_path}. Max token ID: {max_token_id}. Effective vocab size for model: {actual_vocab_size}")

            # Update config with actual vocab size *before* model initialization
            if self.config.vocab_size != actual_vocab_size:
                 logger.warning(f"Updating config.vocab_size from {self.config.vocab_size} to actual size {actual_vocab_size}")
                 self.config.vocab_size = actual_vocab_size
            else:
                 logger.info(f"Config vocab_size ({self.config.vocab_size}) matches actual vocab size ({actual_vocab_size}).")

        except Exception as e:
             logger.error(f"Failed to load or process vocabulary from {self.vocab_path}: {e}", exc_info=True)
             raise RuntimeError("Vocabulary loading failed.") from e

        # --- Initialize Model ---
        logger.info("Initializing model with updated config...")
        self.model = ApertisForCausalLM(self.config)
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled.")
        self.model.to(self.device)
        logger.info(f"Model initialized with config: {self.config.to_dict()}")
        logger.info(f"Model placed on device: {self.device}")

        # --- Wrap Model for Distributed/DataParallel ---
        if self.use_distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
            logger.info(f"Model wrapped with DistributedDataParallel on rank {self.global_rank}.")
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1 and gpu_ids and len(gpu_ids) > 1:
            logger.warning("Using DataParallel. DistributedDataParallel (DDP) is generally recommended for multi-GPU training.")
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids)
            logger.info(f"Model wrapped with DataParallel on GPUs: {gpu_ids}")

        # --- Prepare Datasets and DataLoaders ---
        logger.info("Preparing datasets...")
        # Pass the model\'s actual vocab size to the dataset
        train_dataset = ApertisDataset(
            data_path=self.train_data_path,
            tokenizer_path=self.vocab_path,
            max_length=self.config.max_position_embeddings,
            multimodal=self.config.multimodal,
            image_dir=self.image_dir,
            image_size=getattr(self.config, "image_size", 224),
            model_vocab_size=self.config.vocab_size # Pass model vocab size
        )
        val_dataset = None
        if self.val_data_path:
            val_dataset = ApertisDataset(
                data_path=self.val_data_path,
                tokenizer_path=self.vocab_path,
                max_length=self.config.max_position_embeddings,
                multimodal=self.config.multimodal,
                image_dir=self.image_dir,
                image_size=getattr(self.config, "image_size", 224),
                model_vocab_size=self.config.vocab_size # Pass model vocab size
            )

        # Sampler for distributed training
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.global_rank) if self.use_distributed else None
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.global_rank, shuffle=False) if self.use_distributed and val_dataset else None

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None), # Shuffle only if not using distributed sampler
            num_workers=4, # Adjust based on system capabilities
            pin_memory=True,
        )
        self.val_loader = None
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size * 2, # Often use larger batch size for validation
                sampler=val_sampler,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

        # --- Optimizer and Scheduler ---
        logger.info("Setting up optimizer and scheduler...")
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        self.optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)

        # Total training steps
        self.num_training_steps = len(self.train_loader) // self.gradient_accumulation_steps * self.num_epochs

        # Scheduler (simple linear decay with warmup)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / self.warmup_steps) if step < self.warmup_steps else max(0.0, 1.0 - (step - self.warmup_steps) / (self.num_training_steps - self.warmup_steps))
        )

        # --- Mixed Precision Scaling ---
        self.scaler = torch.amp.GradScaler(enabled=self.fp16)
        if self.fp16:
            logger.info("Mixed precision training (FP16) enabled.")

        # --- Training State ---
        self.global_step = 0
        self.current_epoch = 0

        logger.info("Training pipeline initialized.")

    def _save_checkpoint(self, epoch: int, step: int, is_iteration_checkpoint: bool = False):
        """Save model checkpoint (only on main process)."""
        if not self.is_main_process:
            return

        if is_iteration_checkpoint:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}-iter-{step}")
        else:
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-step-{self.global_step}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Get the underlying model state dict if wrapped (DDP/DataParallel)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model

        # Save model weights
        model_save_path = os.path.join(checkpoint_dir, "model.pt")
        torch.save(model_to_save.state_dict(), model_save_path)

        # Save config (already saved at init, but good practice to save with checkpoint)
        config_save_path = os.path.join(checkpoint_dir, "config.json")
        try:
             model_to_save.config.save_pretrained(checkpoint_dir)
        except Exception as e:
             logger.error(f"Failed to save config during checkpointing: {e}")

        # Save optimizer and scheduler state (optional but recommended for resuming)
        optimizer_save_path = os.path.join(checkpoint_dir, "optimizer.pt")
        torch.save(self.optimizer.state_dict(), optimizer_save_path)
        scheduler_save_path = os.path.join(checkpoint_dir, "scheduler.pt")
        torch.save(self.scheduler.state_dict(), scheduler_save_path)

        # Save training state (optional)
        state_save_path = os.path.join(checkpoint_dir, "trainer_state.json")
        with open(state_save_path, "w") as f:
            json.dump({"epoch": epoch, "global_step": self.global_step}, f)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def _evaluate(self, progress: Optional[gr.Progress] = None):
        """Evaluate the model on the validation set."""
        if not self.val_loader:
            return None

        self.model.eval() # Set model to evaluation mode
        total_eval_loss = 0
        num_eval_steps = 0

        if progress and self.is_main_process:
             progress(0, desc="Starting evaluation...")

        eval_iterator = tqdm(self.val_loader, desc="Evaluating", disable=not self.is_main_process)
        with torch.no_grad():
            for batch in eval_iterator:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

                # Forward pass
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                # Handle potential reduction in distributed setting
                if self.use_distributed and loss is not None:
                    # Average loss across all processes
                    dist.all_reduce(loss, op=dist.ReduceOp.AVG)

                if loss is not None:
                    total_eval_loss += loss.item()
                num_eval_steps += 1

                if progress and self.is_main_process:
                     progress(num_eval_steps / len(self.val_loader), desc=f"Evaluating... Loss: {loss.item():.4f}")

        avg_eval_loss = total_eval_loss / num_eval_steps if num_eval_steps > 0 else 0
        perplexity = torch.exp(torch.tensor(avg_eval_loss)).item() if avg_eval_loss > 0 else float("inf")

        if self.is_main_process:
            logger.info(f"Evaluation finished. Average Loss: {avg_eval_loss:.4f}, Perplexity: {perplexity:.4f}")
            if self.use_wandb:
                wandb.log({"eval/loss": avg_eval_loss, "eval/perplexity": perplexity, "global_step": self.global_step})
            if progress:
                 progress(1, desc=f"Evaluation Complete. Loss: {avg_eval_loss:.4f}, PPL: {perplexity:.4f}")

        self.model.train() # Set model back to training mode
        return avg_eval_loss

    def train(self, stop_event: Optional[threading.Event] = None, progress_callback: Optional[gr.Progress] = None):
        """
        Run the training loop.

        Args:
            stop_event: A threading.Event object to signal training stop.
            progress_callback: A Gradio Progress object to update UI.
        """
        logger.info("Starting training...")
        self.model.train() # Ensure model is in training mode
        total_batches = len(self.train_loader)
        total_steps_per_epoch = total_batches // self.gradient_accumulation_steps

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            if self.is_main_process:
                logger.info(f"Starting Epoch {epoch + 1}/{self.num_epochs}")
            if self.use_distributed and self.train_loader.sampler:
                self.train_loader.sampler.set_epoch(epoch) # Important for shuffling in DDP

            epoch_iterator = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}", disable=not self.is_main_process)
            train_loss = 0.0
            steps_in_epoch = 0

            for step, batch in enumerate(epoch_iterator):
                # Check for stop signal
                if stop_event and stop_event.is_set():
                    logger.info(f"Stop signal received. Stopping training at Epoch {epoch+1}, Step {step}.")
                    if self.is_main_process and self.use_wandb:
                         wandb.finish()
                    return # Exit the training loop

                # Move batch to device
                try:
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                except Exception as e:
                     logger.error(f"Error moving batch to device at Epoch {epoch+1}, Step {step}: {e}")
                     continue # Skip this batch

                # Forward pass with mixed precision
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if loss is None:
                     logger.warning(f"Loss is None at Epoch {epoch+1}, Step {step}. Skipping backward pass.")
                     continue

                # Normalize loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

                # Backward pass with scaler
                self.scaler.scale(loss).backward()

                train_loss += loss.item() * self.gradient_accumulation_steps # Accumulate un-normalized loss for logging

                # Optimizer step (every gradient_accumulation_steps)
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                    # Optimizer step
                    self.scaler.step(self.optimizer)

                    # Update scaler
                    self.scaler.update()

                    # Zero gradients
                    self.optimizer.zero_grad()

                    # Scheduler step
                    self.scheduler.step()

                    self.global_step += 1
                    steps_in_epoch += 1

                    # Logging (only on main process)
                    if self.is_main_process:
                        current_lr = self.scheduler.get_last_lr()[0]
                        avg_loss_so_far = train_loss / (steps_in_epoch * self.gradient_accumulation_steps)
                        log_msg = f"Epoch: {epoch+1}, Step: {steps_in_epoch}/{total_steps_per_epoch}, Global Step: {self.global_step}, LR: {current_lr:.6f}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}, Avg Loss: {avg_loss_so_far:.4f}"
                        epoch_iterator.set_description(log_msg)
                        # logger.info(log_msg) # Optional: log every step

                        if self.use_wandb:
                            wandb.log({
                                "train/loss": loss.item() * self.gradient_accumulation_steps,
                                "train/learning_rate": current_lr,
                                "global_step": self.global_step,
                                "epoch": epoch + (step / total_batches) # Log fractional epoch
                            })

                        # Update Gradio progress
                        if progress_callback is not None and isinstance(progress_callback, gr.Progress): # Check type
                            progress_val = (epoch * total_steps_per_epoch + steps_in_epoch) / (self.num_epochs * total_steps_per_epoch)
                            progress_desc = f"Epoch {epoch+1}/{self.num_epochs}, Step {steps_in_epoch}/{total_steps_per_epoch}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}"
                            try:
                                progress_callback(progress_val, desc=progress_desc)
                            except IndexError as ie:
                                 logger.error(f"Caught IndexError updating Gradio progress (likely empty iterables): {ie}. Progress bar may not update.")
                            except Exception as pe:
                                 logger.error(f"Failed to update Gradio progress: {pe}")
                        elif progress_callback is not None and callable(progress_callback): # Fallback if it's just a callable wrapper
                            progress_val = (epoch * total_steps_per_epoch + steps_in_epoch) / (self.num_epochs * total_steps_per_epoch)
                            progress_desc = f"Epoch {epoch+1}/{self.num_epochs}, Step {steps_in_epoch}/{total_steps_per_epoch}, Loss: {loss.item() * self.gradient_accumulation_steps:.4f}"
                            try:
                                progress_callback(progress_val, desc=progress_desc)
                            except IndexError as ie:
                                 logger.error(f"Caught IndexError updating Gradio progress (likely empty iterables): {ie}. Progress bar may not update.")
                            except Exception as pe:
                                 logger.error(f"Failed to update Gradio progress: {pe}")

                    # Evaluation step
                    if self.eval_steps > 0 and self.global_step % self.eval_steps == 0:
                        self._evaluate(progress=progress_callback)
                        self.model.train() # Ensure model is back in train mode after eval

                    # Global step checkpointing
                    if self.checkpoint_steps > 0 and self.global_step % self.checkpoint_steps == 0:
                        self._save_checkpoint(epoch=epoch + 1, step=self.global_step)

                # Iteration checkpointing (within epoch)
                if self.iteration_checkpoint_steps > 0 and (step + 1) % self.iteration_checkpoint_steps == 0:
                    self._save_checkpoint(epoch=epoch + 1, step=step + 1, is_iteration_checkpoint=True)

            # End of epoch evaluation
            if self.eval_steps == 0: # Evaluate at end of epoch if not evaluating periodically
                 self._evaluate(progress=progress_callback)
                 self.model.train()

            # End of epoch checkpoint (optional, could save last checkpoint)
            # self._save_checkpoint(epoch=epoch + 1, step=self.global_step)

        logger.info("Training finished.")
        if self.is_main_process and self.use_wandb:
            wandb.finish()

        # Final save (only on main process)
        if self.is_main_process:
             final_save_dir = os.path.join(self.output_dir, "final_model")
             os.makedirs(final_save_dir, exist_ok=True)
             model_to_save = self.model.module if hasattr(self.model, "module") else self.model
             torch.save(model_to_save.state_dict(), os.path.join(final_save_dir, "model.pt"))
             model_to_save.config.save_pretrained(final_save_dir)
             logger.info(f"Final model saved to {final_save_dir}")

# --- Utility Functions ---
def get_available_gpus() -> List[Dict[str, Any]]:
    """Returns a list of available CUDA GPUs with their properties."""
    gpus = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "id": i,
                    "name": props.name,
                    "total_memory": props.total_memory / (1024**3), # Convert bytes to GB
                })
            except Exception as e:
                logger.error(f"Could not get properties for GPU {i}: {e}")
    return gpus

