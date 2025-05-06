#!/usr/bin/env python3
"""
Apertis CLI - Command Line Interface for Apertis LLM
"""

import os
import sys
import argparse
import json
import torch
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_module_imports():
    """Ensure that the src module can be imported."""
    # Add the parent directory to sys.path if needed
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    
    try:
        import src
        logger.info(f"Successfully imported src module")
    except ImportError:
        # If import fails, try adding the grandparent directory
        grandparent_dir = parent_dir.parent
        if str(grandparent_dir) not in sys.path:
            sys.path.insert(0, str(grandparent_dir))
        
        try:
            import src
            logger.info(f"Successfully imported src module")
        except ImportError:
            logger.error("Failed to import src module. Make sure it's installed or in the Python path.")
            sys.exit(1)

def chat_command(args):
    """Handle the chat command."""
    from src.inference.interface import ApertisInterface
    
    interface = ApertisInterface(
        model_path=args.model_path,
        vocab_file=args.vocab_file,
        multimodal=args.multimodal,
        device=args.device,
        web=args.web,
        port=args.port,
        share=args.share,
    )
    
    if args.web:
        # Web interface is launched in the ApertisInterface constructor
        pass
    else:
        # Start interactive CLI
        print("Apertis CLI Chat Interface")
        print("Type 'exit' to quit, 'reset' to reset chat history")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "reset":
                interface.reset_chat()
                print("Chat history reset")
                continue
            
            response = interface.chat(
                message=user_input,
                image_path=args.image,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
            print(f"\nApertis: {response}")

def train_command(args):
    """Handle the train command."""
    from src.training.pipeline import train_from_config
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    
    # Train model
    logger.info(f"Starting training with config: {args.config}")
    metrics = train_from_config(args.config)
    
    # Print metrics
    print("\nTraining completed!")
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

def create_model_command(args):
    """Handle the create-model command."""
    from src.model.core import create_apertis_model
    
    # Create model
    logger.info(f"Creating {args.size} model (multimodal: {args.multimodal})")
    model = create_apertis_model(
        model_size=args.size,
        multimodal=args.multimodal,
        use_flash_attention=args.flash_attention,
        use_expert_system=args.expert_system,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(args.output_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model.config.to_dict(), f, indent=2)
    
    # Create a minimal vocabulary if it doesn't exist
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        minimal_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        with open(vocab_path, "w") as f:
            json.dump(minimal_vocab, f, indent=2)
    
    logger.info(f"Model created successfully at {args.output_dir}")
    print(f"Model created successfully!")
    print(f"- Model saved to: {model_path}")
    print(f"- Config saved to: {config_path}")
    print(f"- Minimal vocabulary saved to: {vocab_path}")

def create_config_command(args):
    """Handle the create-config command."""
    from src.training.pipeline import create_sample_config
    
    # Create sample configuration
    create_sample_config(args.output)
    
    logger.info(f"Sample configuration created at {args.output}")
    print(f"Sample training configuration created at: {args.output}")
    print("Edit this file to customize your training settings.")

def main():
    """Main entry point for the CLI."""
    # Ensure src module can be imported
    ensure_module_imports()
    
    # Create parser
    parser = argparse.ArgumentParser(
        description="Apertis CLI - Command Line Interface for Apertis LLM",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with an Apertis model")
    chat_parser.add_argument("--model-path", type=str, help="Path to the model file or directory")
    chat_parser.add_argument("--vocab-file", type=str, help="Path to the vocabulary file")
    chat_parser.add_argument("--multimodal", action="store_true", help="Enable multimodal capabilities")
    chat_parser.add_argument("--image", type=str, help="Path to an image file for multimodal input")
    chat_parser.add_argument("--device", type=str, help="Device to use for inference (cuda or cpu)")
    chat_parser.add_argument("--web", action="store_true", help="Launch web interface")
    chat_parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    chat_parser.add_argument("--share", action="store_true", help="Create a public link")
    chat_parser.add_argument("--max-length", type=int, default=100, help="Maximum response length")
    chat_parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    chat_parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling parameter")
    chat_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling parameter")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train an Apertis model")
    train_parser.add_argument("--config", type=str, required=True, help="Path to training configuration file")
    
    # Create model command
    create_model_parser = subparsers.add_parser("create-model", help="Create a new Apertis model")
    create_model_parser.add_argument("--size", type=str, default="base", choices=["small", "base", "large"], help="Model size")
    create_model_parser.add_argument("--multimodal", action="store_true", help="Enable multimodal capabilities")
    create_model_parser.add_argument("--flash-attention", action="store_true", help="Use flash attention")
    create_model_parser.add_argument("--expert-system", action="store_true", help="Use adaptive expert system")
    create_model_parser.add_argument("--output-dir", type=str, default="models/new_model", help="Output directory")
    
    # Create config command
    create_config_parser = subparsers.add_parser("create-config", help="Create a sample training configuration")
    create_config_parser.add_argument("--output", type=str, default="config.json", help="Output file path")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "chat":
        chat_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "create-model":
        create_model_command(args)
    elif args.command == "create-config":
        create_config_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
