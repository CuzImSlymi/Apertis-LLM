import os
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
            vocab = json.load(f)
        logger.info(f"Loaded vocabulary with {len(vocab)} tokens from {self.tokenizer_path}")
        return vocab
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization by splitting on spaces and mapping to vocabulary."""
        # This is a placeholder for a real tokenizer
        # In a production system, you would use a proper tokenizer
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
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        num_epochs: int = 3,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        use_wandb: bool = False,
        wandb_project: str = "apertis",
        wandb_run_name: Optional[str] = None,
        fp16: bool = False,
        device: Optional[str] = None,
        checkpoint_steps: int = 1000,
        gpu_memory_fraction: float = 0.9,
    ):
        """
        Initialize the trainer.
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
        self.gpu_memory_fraction = gpu_memory_fraction
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Limit GPU memory usage if using CUDA
        if torch.cuda.is_available() and self.gpu_memory_fraction < 1.0:
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = int(total_memory * self.gpu_memory_fraction)
                # This is a workaround to limit memory usage
                # Create a temporary tensor to reserve memory
                tmp_tensor = torch.empty(reserved_memory, dtype=torch.int8, device='cuda')
                del tmp_tensor
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to limit GPU memory: {e}")
        
        logger.info(f"Using device: {self.device}")
        
        # Move model to device
        self.model.to(self.device)
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Reduced from 4 to 0 to avoid hanging issues
            pin_memory=True,
        )
        
        if val_dataset is not None:
            self.val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Reduced from 4 to 0 to avoid hanging issues
                pin_memory=True,
            )
        else:
            self.val_dataloader = None
        
        # Set up optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Set up learning rate scheduler
        total_steps = len(self.train_dataloader) * num_epochs // gradient_accumulation_steps
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if warmup_steps > 0 else 0.1,
        )
        
        # Set up mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if fp16 else None
        
        # Initialize Weights & Biases
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model_type": model.config.model_type,
                    "hidden_size": model.config.hidden_size,
                    "num_hidden_layers": model.config.num_hidden_layers,
                    "num_attention_heads": model.config.num_attention_heads,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "num_epochs": num_epochs,
                    "warmup_steps": warmup_steps,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "fp16": fp16,
                },
            )
    
    def train(self) -> Dict[str, float]:
        """Train the model and return metrics."""
        logger.info("Starting training")
        
        # Track best validation loss
        best_val_loss = float("inf")
        
        # Training loop
        global_step = 0
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_steps = 0
            
            # Add timeout handling and better error reporting
            try:
                progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}")
                for step, batch in enumerate(progress_bar):
                    # Log first batch processing to help with debugging
                    if epoch == 0 and step == 0:
                        logger.info(f"Processing first batch (size: {batch['input_ids'].shape})")
                        # Log max token ID for debugging
                        max_id = torch.max(batch['input_ids']).item()
                        logger.info(f"Maximum token ID in batch: {max_id}, Model vocab size: {self.model.config.vocab_size}")
                    
                    try:
                        # Validate input IDs are within vocabulary bounds
                        if torch.max(batch['input_ids']) >= self.model.config.vocab_size:
                            logger.warning(f"Batch contains token IDs exceeding vocabulary size. Max ID: {torch.max(batch['input_ids']).item()}, Vocab size: {self.model.config.vocab_size}")
                            # Clip token IDs to be within vocabulary bounds
                            batch['input_ids'] = torch.clamp(batch['input_ids'], max=self.model.config.vocab_size-1)
                            # Also clip labels if present
                            if 'labels' in batch:
                                batch['labels'] = torch.clamp(batch['labels'], max=self.model.config.vocab_size-1)
                        
                        # Move batch to device with error handling
                        try:
                            batch = {k: v.to(self.device) for k, v in batch.items()}
                        except RuntimeError as e:
                            if "CUDA out of memory" in str(e):
                                logger.error("CUDA out of memory error. Trying to recover by clearing cache and reducing batch size.")
                                torch.cuda.empty_cache()
                                # Skip this batch and continue with next one
                                continue
                            else:
                                raise
                        
                        # Optimize GPU memory by clearing cache every few steps
                        if step % 5 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Forward pass with mixed precision if enabled
                        if self.fp16:
                            with torch.cuda.amp.autocast():
                                outputs = self.model(**batch)
                                loss = outputs[0]
                                loss = loss / self.gradient_accumulation_steps
                            
                            # Backward pass with gradient scaling
                            self.scaler.scale(loss).backward()
                            
                            if (step + 1) % self.gradient_accumulation_steps == 0:
                                # Gradient clipping
                                self.scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                                
                                # Update weights
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                                self.scheduler.step()
                                self.optimizer.zero_grad()
                                global_step += 1
                                
                                # Synchronize CUDA to prevent hanging
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                        else:
                            # Standard forward pass
                            outputs = self.model(**batch)
                            loss = outputs[0]
                            loss = loss / self.gradient_accumulation_steps
                            
                            # Backward pass
                            loss.backward()
                            
                            if (step + 1) % self.gradient_accumulation_steps == 0:
                                # Gradient clipping
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                                
                                # Update weights
                                self.optimizer.step()
                                self.scheduler.step()
                                self.optimizer.zero_grad()
                                global_step += 1
                                
                                # Synchronize CUDA to prevent hanging
                                if torch.cuda.is_available():
                                    torch.cuda.synchronize()
                        
                        # Update progress bar
                        train_loss += loss.item() * self.gradient_accumulation_steps
                        train_steps += 1
                        
                        # Force progress bar update
                        progress_bar.set_postfix({"loss": train_loss / train_steps})
                        progress_bar.update()
                        
                        # Log progress to file
                        if step % 10 == 0:
                            with open(os.path.join(self.output_dir, "training_log.txt"), "a") as f:
                                f.write(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}\n")
                        
                        # Log to Weights & Biases
                        if self.use_wandb:
                            wandb.log({
                                "train_loss": loss.item() * self.gradient_accumulation_steps,
                                "learning_rate": self.scheduler.get_last_lr()[0],
                                "epoch": epoch + step / len(self.train_dataloader),
                                "global_step": global_step,
                            })
                        
                        # Save checkpoint
                        if global_step % self.checkpoint_steps == 0:
                            self._save_checkpoint(global_step)
                            
                    except Exception as e:
                        logger.error(f"Error processing batch at step {step}: {str(e)}")
                        # Continue with next batch instead of crashing
                        continue
            except Exception as e:
                logger.error(f"Error in training loop: {str(e)}")
                # Save emergency checkpoint
                self._save_checkpoint("emergency")
            
            # Calculate average training loss
            avg_train_loss = train_loss / train_steps
            logger.info(f"Average training loss: {avg_train_loss:.4f}")
            
            # Validation
            if self.val_dataloader is not None:
                val_metrics = self.evaluate()
                val_loss = val_metrics["val_loss"]
                
                logger.info(f"Validation loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save_checkpoint("best")
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
            
            # Save checkpoint at the end of each epoch
            self._save_checkpoint(f"epoch-{epoch+1}")
        
        logger.info("Training complete")
        
        # Save final model
        self._save_checkpoint("final")
        
        # Return metrics
        metrics = {
            "train_loss": avg_train_loss,
        }
        
        if self.val_dataloader is not None:
            metrics.update(val_metrics)
        
        return metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the validation dataset."""
        logger.info("Evaluating model")
        
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_dataloader, desc="Validation")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs[0]
                
                # Update metrics
                val_loss += loss.item()
                val_steps += 1
                progress_bar.set_postfix({"loss": val_loss / val_steps})
        
        # Calculate average validation loss
        avg_val_loss = val_loss / val_steps
        
        # Log to Weights & Biases
        if self.use_wandb:
            wandb.log({"val_loss": avg_val_loss})
        
        return {"val_loss": avg_val_loss}
    
    def _save_checkpoint(self, checkpoint_id: Union[int, str]) -> None:
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{checkpoint_id}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
        
        # Save configuration
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(self.model.config.to_dict(), f, indent=2)
        
        logger.info(f"Saved checkpoint to {checkpoint_dir}")


class YoloStyleTrainingPipeline:
    """YOLO-style training pipeline for Apertis models."""
    
    def __init__(
        self,
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        training_config: Dict[str, Any],
    ):
        """
        Initialize the training pipeline.
        
        Args:
            data_config: Configuration for data loading
            model_config: Configuration for model architecture
            training_config: Configuration for training process
        """
        self.data_config = data_config
        self.model_config = model_config
        self.training_config = training_config
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(training_config["output_dir"], "training.log")),
                logging.StreamHandler(),
            ],
        )
        
        # Create output directory
        os.makedirs(training_config["output_dir"], exist_ok=True)
        
        # Save configurations
        self._save_configs()
    
    def _save_configs(self) -> None:
        """Save configurations to output directory."""
        configs = {
            "data_config": self.data_config,
            "model_config": self.model_config,
            "training_config": self.training_config,
        }
        
        config_path = os.path.join(self.training_config["output_dir"], "config.json")
        with open(config_path, "w") as f:
            json.dump(configs, f, indent=2)
        
        logger.info(f"Saved configurations to {config_path}")
    
    def prepare_datasets(self) -> Tuple[ApertisDataset, Optional[ApertisDataset]]:
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets")
        
        # Get model's vocabulary size
        vocab_size = self.model_config.get("vocab_size", 32000)
        
        # Create training dataset
        train_dataset = ApertisDataset(
            data_path=self.data_config["train_data_path"],
            tokenizer_path=self.data_config["tokenizer_path"],
            max_length=self.data_config.get("max_length", 512),
            multimodal=self.data_config.get("multimodal", False),
            image_dir=self.data_config.get("image_dir"),
            image_size=self.data_config.get("image_size", 224),
            model_vocab_size=vocab_size,  # Pass model's vocab size to dataset
        )
        
        # Create validation dataset if specified
        val_dataset = None
        if "val_data_path" in self.data_config:
            val_dataset = ApertisDataset(
                data_path=self.data_config["val_data_path"],
                tokenizer_path=self.data_config["tokenizer_path"],
                max_length=self.data_config.get("max_length", 512),
                multimodal=self.data_config.get("multimodal", False),
                image_dir=self.data_config.get("image_dir"),
                image_size=self.data_config.get("image_size", 224),
                model_vocab_size=vocab_size,  # Pass model's vocab size to dataset
            )
        
        return train_dataset, val_dataset
    
    def create_model(self) -> ApertisForCausalLM:
        """Create Apertis model based on configuration."""
        logger.info("Creating model")
        
        # Determine vocabulary size based on tokenizer
        tokenizer_path = self.data_config["tokenizer_path"]
        try:
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                vocab = json.load(f)
            
            # Find the maximum token ID to determine actual vocab size
            max_token_id = 0
            for _, idx in vocab.items():
                if isinstance(idx, int) and idx > max_token_id:
                    max_token_id = idx
            
            # Add 1 to account for 0-indexing and add padding for future tokens
            actual_vocab_size = max_token_id + 1
            # Add 10% padding for safety, but at least 100 tokens
            vocab_size = min(actual_vocab_size + max(100, actual_vocab_size // 10), 250000)
            
            logger.info(f"Detected vocabulary size: {actual_vocab_size}, using padded size: {vocab_size}")
            
            # Update model config with the detected vocabulary size
            self.model_config["vocab_size"] = vocab_size
        except Exception as e:
            logger.warning(f"Error detecting vocabulary size: {e}")
            logger.warning("Using default vocabulary size from model config")
        
        # Create configuration
        config = ApertisConfig(
            vocab_size=self.model_config.get("vocab_size", 32000),
            hidden_size=self.model_config.get("hidden_size", 768),
            num_hidden_layers=self.model_config.get("num_hidden_layers", 12),
            num_attention_heads=self.model_config.get("num_attention_heads", 12),
            intermediate_size=self.model_config.get("intermediate_size", 3072),
            hidden_act=self.model_config.get("hidden_act", "gelu"),
            hidden_dropout_prob=self.model_config.get("hidden_dropout_prob", 0.1),
            attention_probs_dropout_prob=self.model_config.get("attention_probs_dropout_prob", 0.1),
            max_position_embeddings=self.model_config.get("max_position_embeddings", 2048),
            position_embedding_type=self.model_config.get("position_embedding_type", "rotary"),
            attention_type=self.model_config.get("attention_type", "selective_linear"),
            use_expert_system=self.model_config.get("use_expert_system", False),
            num_experts=self.model_config.get("num_experts", 8),
            experts_per_token=self.model_config.get("experts_per_token", 2),
            multimodal=self.model_config.get("multimodal", False),
            image_size=self.model_config.get("image_size", 224),
            vision_embed_dim=self.model_config.get("vision_embed_dim", 768),
            vision_patch_size=self.model_config.get("vision_patch_size", 16),
            vision_layers=self.model_config.get("vision_layers", 12),
            vision_heads=self.model_config.get("vision_heads", 12),
        )
        
        # Create model
        model = ApertisForCausalLM(config)
        
        # Load pretrained weights if specified
        if "pretrained_model_path" in self.model_config:
            pretrained_path = self.model_config["pretrained_model_path"]
            logger.info(f"Loading pretrained weights from {pretrained_path}")
            model.load_state_dict(torch.load(pretrained_path))
        
        return model
    
    def train(self) -> Dict[str, float]:
        """Run the training pipeline."""
        logger.info("Starting training pipeline")
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        # Create model
        model = self.create_model()
        
        # Create trainer
        trainer = ApertisTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=self.training_config["output_dir"],
            batch_size=self.training_config.get("batch_size", 8),
            learning_rate=self.training_config.get("learning_rate", 5e-5),
            weight_decay=self.training_config.get("weight_decay", 0.01),
            num_epochs=self.training_config.get("num_epochs", 3),
            warmup_steps=self.training_config.get("warmup_steps", 0),
            gradient_accumulation_steps=self.training_config.get("gradient_accumulation_steps", 1),
            max_grad_norm=self.training_config.get("max_grad_norm", 1.0),
            use_wandb=self.training_config.get("use_wandb", False),
            wandb_project=self.training_config.get("wandb_project", "apertis"),
            wandb_run_name=self.training_config.get("wandb_run_name"),
            fp16=self.training_config.get("fp16", False),
            device=self.training_config.get("device"),
            checkpoint_steps=self.training_config.get("checkpoint_steps", 1000),
        )
        
        # Train model
        metrics = trainer.train()
        
        # Save metrics
        metrics_path = os.path.join(self.training_config["output_dir"], "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        
        return metrics


def train_from_config(config_path: str) -> Dict[str, float]:
    """Train a model from a configuration file."""
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create training pipeline
    pipeline = YoloStyleTrainingPipeline(
        data_config=config["data_config"],
        model_config=config["model_config"],
        training_config=config["training_config"],
    )
    
    # Train model
    metrics = pipeline.train()
    
    return metrics


def create_sample_config(output_path: str) -> None:
    """Create a sample configuration file."""
    config = {
        "data_config": {
            "train_data_path": "data/train.jsonl",
            "val_data_path": "data/val.jsonl",
            "tokenizer_path": "data/vocab.json",
            "max_length": 512,
            "multimodal": False,
            "image_dir": "data/images",
            "image_size": 224,
        },
        "model_config": {
            "vocab_size": 32000,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 2048,
            "position_embedding_type": "rotary",
            "attention_type": "selective_linear",
            "use_expert_system": False,
            "num_experts": 8,
            "experts_per_token": 2,
            "multimodal": False,
            "image_size": 224,
            "vision_embed_dim": 768,
            "vision_patch_size": 16,
            "vision_layers": 12,
            "vision_heads": 12,
        },
        "training_config": {
            "output_dir": "output",
            "batch_size": 8,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
            "num_epochs": 3,
            "warmup_steps": 0,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
            "use_wandb": False,
            "wandb_project": "apertis",
            "wandb_run_name": "apertis-training",
            "fp16": False,
            "device": None,
            "checkpoint_steps": 1000,
        },
    }
    
    # Save configuration
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created sample configuration at {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apertis Training Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--create-config", type=str, help="Create a sample configuration file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config(args.create_config)
    elif args.config:
        train_from_config(args.config)
    else:
        parser.print_help()
