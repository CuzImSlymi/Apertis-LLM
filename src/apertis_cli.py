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
    from src.model.core import create_apertis_model, estimate_model_parameters # Import estimate_model_parameters
    
    # Create model
    logger.info(f"Creating model with target parameters: {args.target_params} (multimodal: {args.multimodal})")

    default_vocab_size_cli = args.vocab_size if args.vocab_size is not None else 32000

    config_overrides = {}
    if args.expert_system:
        config_overrides["num_experts"] = args.num_experts
        config_overrides["experts_per_token"] = args.experts_per_token
        config_overrides["load_balancing_loss_coef"] = args.load_balancing_loss_coef
        config_overrides["expert_capacity_factor"] = args.expert_capacity_factor
        config_overrides["noisy_routing_alpha"] = args.noisy_routing_alpha
        config_overrides["expert_dropout_prob"] = args.expert_dropout_prob
        config_overrides["router_z_loss_coef"] = args.router_z_loss_coef
        config_overrides["use_noisy_top_k_routing"] = args.use_noisy_top_k_routing
        config_overrides["use_expert_capacity_limit"] = args.use_expert_capacity_limit
        config_overrides["use_expert_dropout"] = args.use_expert_dropout
        config_overrides["use_router_z_loss"] = args.use_router_z_loss
        config_overrides["use_load_balancing_loss"] = args.use_load_balancing_loss
        # Ensure experts_per_token is not greater than num_experts if both are provided
        if "num_experts" in config_overrides and "experts_per_token" in config_overrides:
            if config_overrides["experts_per_token"] > config_overrides["num_experts"]:
                logger.warning(f"Experts per token ({config_overrides['experts_per_token']}) > num_experts ({config_overrides['num_experts']}). Clamping experts_per_token.")
                config_overrides["experts_per_token"] = config_overrides["num_experts"]


    model = create_apertis_model(
        target_param_count=args.target_params,
        vocab_size_override=default_vocab_size_cli,
        multimodal=args.multimodal,
        use_flash_attention=args.flash_attention,
        use_expert_system=args.expert_system,
        config_overrides=config_overrides
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

    actual_params = estimate_model_parameters(model.config)
    config_details = model.config.to_dict()

    print(f"Model created successfully!")
    print(f"- Target Parameters: {args.target_params}")
    print(f"- Estimated Actual Parameters: {actual_params:,} (~{actual_params/1e6:.2f}M)")
    print(f"- Model saved to: {model_path}")
    print(f"- Config saved to: {config_path}")
    print(f"  - Hidden Size: {config_details.get('hidden_size')}")
    print(f"  - Num Layers: {config_details.get('num_hidden_layers')}")
    print(f"  - Num Heads: {config_details.get('num_attention_heads')}")
    print(f"  - Intermediate Size: {config_details.get('intermediate_size')}")
    print(f"  - Vocab Size: {config_details.get('vocab_size')}")
    if config_details.get('use_expert_system'):
        print(f"  - Experts: {config_details.get('num_experts')}, Per Token: {config_details.get('experts_per_token')}")
    print(f"- Minimal vocabulary (for vocab size {model.config.vocab_size}) saved to: {vocab_path}")


def create_config_command(args):
    """Handle the create-config command."""
    from src.training.pipeline import create_sample_config
    
    # Create sample configuration
    create_sample_config(args.output)
    
    logger.info(f"Sample configuration created at {args.output}")
    print(f"Sample training configuration created at: {args.output}")
    print("Edit this file to customize your training settings.")

def data_pipeline_command(args):
    """Handle the data-pipeline command."""
    from src.data_pipeline.main import run_pipeline
    from src.data_pipeline.config import DataPipelineConfig
    
    if not os.path.exists(args.config):
        logger.error(f"Data pipeline configuration file not found: {args.config}")
        sys.exit(1)
        
    logger.info(f"Loading data pipeline configuration from {args.config}")
    config = DataPipelineConfig.from_yaml(args.config)
    run_pipeline(config)

def create_pipeline_config_command(args):
    """Handle the create-pipeline-config command."""
    from src.data_pipeline.config import create_sample_pipeline_config
    
    create_sample_pipeline_config(args.output)
    logger.info(f"Sample data pipeline configuration created at {args.output}")
    print(f"Sample data pipeline configuration created at: {args.output}")
    print("Edit this file to configure your data processing stages.")

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
    subparsers = parser.add_subparsers(dest="command", help="Command to execute", required=True)
    
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
    create_model_parser = subparsers.add_parser("create-model", help="Create a new Apertis model based on target parameter count")
    create_model_parser.add_argument("--target-params", type=str, default="125M", help="Target parameter count (e.g., 10M, 125M, 1.5B, 7B, 70B)")
    create_model_parser.add_argument("--vocab-size", type=int, help="Override default vocabulary size for the new model")
    create_model_parser.add_argument("--multimodal", action="store_true", help="Enable multimodal capabilities")
    create_model_parser.add_argument("--flash-attention", action="store_true", help="Use flash attention (if supported by attention type)")
    create_model_parser.add_argument("--output-dir", type=str, default="models/new_param_model", help="Output directory for model files")

    # MoE specific arguments for create-model
    moe_group = create_model_parser.add_argument_group('MoE Configuration (used if --expert-system is specified)')
    moe_group.add_argument("--expert-system", action="store_true", help="Use adaptive expert system (MoE)")
    moe_group.add_argument("--num-experts", type=int, default=8, help="Number of experts")
    moe_group.add_argument("--experts-per-token", type=int, default=2, help="Number of experts to route each token to (top-k)")
    moe_group.add_argument("--load-balancing-loss-coef", type=float, default=0.01, help="Coefficient for MoE load balancing loss")
    moe_group.add_argument("--expert-capacity-factor", type=float, default=1.25, help="Capacity factor for experts")
    moe_group.add_argument("--noisy-routing-alpha", type=float, default=0.1, help="Alpha for noisy routing")
    moe_group.add_argument("--expert-dropout-prob", type=float, default=0.1, help="Dropout probability for entire experts")
    moe_group.add_argument("--router-z-loss-coef", type=float, default=0.001, help="Coefficient for router z-loss")
    moe_group.add_argument("--use-noisy-top-k-routing", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable noisy top-k routing (True/False)")
    moe_group.add_argument("--use-expert-capacity-limit", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable expert capacity limits (True/False)")
    moe_group.add_argument("--use-expert-dropout", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable expert dropout (True/False)")
    moe_group.add_argument("--use-router-z-loss", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable router z-loss (True/False)")
    moe_group.add_argument("--use-load-balancing-loss", type=lambda x: (str(x).lower() == 'true'), default=True, help="Enable load balancing loss (True/False)")

    # Create config command
    create_config_parser = subparsers.add_parser("create-config", help="Create a sample training configuration")
    create_config_parser.add_argument("--output", type=str, default="config.json", help="Output file path")

    # Data pipeline command
    pipeline_parser = subparsers.add_parser("data-pipeline", help="Run the distributed data processing pipeline")
    pipeline_parser.add_argument("--config", type=str, required=True, help="Path to the data pipeline YAML configuration file")

    # Create pipeline config command
    create_pipeline_config_parser = subparsers.add_parser("create-pipeline-config", help="Create a sample data pipeline configuration file")
    create_pipeline_config_parser.add_argument("--output", type=str, default="pipeline_config.yaml", help="Output file path for the pipeline config")

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
    elif args.command == "data-pipeline":
        data_pipeline_command(args)
    elif args.command == "create-pipeline-config":
        create_pipeline_config_command(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()