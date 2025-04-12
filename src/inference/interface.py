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

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import Apertis modules
from src.model.core import ApertisConfig, ApertisForCausalLM, create_apertis_model
from src.training.pipeline import YoloStyleTrainingPipeline, ApertisDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ApertisInterface:
    """Google AI Studio-like interface for Apertis."""
    
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
        """
        Initialize the interface.
        
        Args:
            model_path: Path to the model file
            vocab_file: Path to the vocabulary file
            multimodal: Whether to use multimodal capabilities
            device: Device to use for inference
            web: Whether to launch the web interface
            port: Port for the web interface
            share: Whether to create a public link
        """
        self.model_path = model_path
        self.vocab_file = vocab_file
        self.multimodal = multimodal
        self.web = web
        self.port = port
        self.share = share
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model and vocabulary if provided
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        
        if model_path is not None and vocab_file is not None:
            self.load_model(model_path)
            self.load_vocabulary(vocab_file)
        
        # Initialize chat history
        self.chat_history = []
        
        # Launch web interface if requested
        if web:
            self.launch_web_interface()
    
    def load_model(self, model_path: str) -> None:
        """Load model from file."""
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Check if path is a directory (saved model) or a file (state dict)
            if os.path.isdir(model_path):
                # Load configuration
                config_path = os.path.join(model_path, "config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config_dict = json.load(f)
                    
                    config = ApertisConfig.from_dict(config_dict)
                    self.model = ApertisForCausalLM(config)
                    
                    # Load state dict
                    state_dict_path = os.path.join(model_path, "model.pt")
                    if os.path.exists(state_dict_path):
                        self.model.load_state_dict(torch.load(state_dict_path, map_location=self.device))
                    else:
                        logger.warning(f"Model state dict not found at {state_dict_path}")
                else:
                    logger.warning(f"Model configuration not found at {config_path}")
                    # Create a default model
                    self.model = create_apertis_model(model_size="small", multimodal=self.multimodal)
            else:
                # Assume it's a state dict file
                # Create a default model
                self.model = create_apertis_model(model_size="small", multimodal=self.multimodal)
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Create a default model as fallback
            logger.info("Creating default model as fallback")
            self.model = create_apertis_model(model_size="small", multimodal=self.multimodal)
            self.model.to(self.device)
            self.model.eval()
    
    def load_vocabulary(self, vocab_file: str) -> None:
        """Load vocabulary from file."""
        try:
            logger.info(f"Loading vocabulary from {vocab_file}")
            
            with open(vocab_file, "r") as f:
                self.vocab = json.load(f)
            
            # Create reverse vocabulary for decoding
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            
            logger.info(f"Vocabulary loaded with {len(self.vocab)} tokens")
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            # Create a minimal vocabulary as fallback
            logger.info("Creating minimal vocabulary as fallback")
            self.vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using the loaded vocabulary."""
        if self.vocab is None:
            logger.warning("Vocabulary not loaded, using minimal tokenization")
            return [3] * len(text.split())  # All tokens as <unk>
        
        # Simple tokenization by splitting on spaces
        # In a production system, you would use a proper tokenizer
        tokens = []
        for word in text.split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.vocab.get("<unk>", 3))
        
        return tokens
    
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        if self.reverse_vocab is None:
            logger.warning("Vocabulary not loaded, cannot detokenize")
            return "[Unable to detokenize without vocabulary]"
        
        # Convert token IDs to words
        words = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                word = self.reverse_vocab[token_id]
                # Skip special tokens
                if word not in ["<pad>", "<bos>", "<eos>", "<unk>"]:
                    words.append(word)
            else:
                words.append("<unk>")
        
        return " ".join(words)
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess image for model input."""
        try:
            from torchvision import transforms
            
            # Define image transformations
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            # Load and transform image
            image = Image.open(image_path).convert("RGB")
            image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
            
            return image_tensor
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return a blank image as fallback
            return torch.zeros(1, 3, 224, 224)
    
    def generate_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response to the given prompt."""
        if self.model is None:
            return "Model not loaded. Please load a model first."
        
        try:
            # Tokenize prompt
            input_ids = self.tokenize(prompt)
            
            # Add <bos> token if not present
            if len(input_ids) == 0 or input_ids[0] != self.vocab.get("<bos>", 1):
                input_ids = [self.vocab.get("<bos>", 1)] + input_ids
            
            # Convert to tensor
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            
            # Create attention mask (all 1s for input tokens)
            attention_mask = torch.ones_like(input_ids)
            
            # Prepare image if provided and model is multimodal
            pixel_values = None
            if image_path is not None and self.multimodal:
                pixel_values = self.preprocess_image(image_path).to(self.device)
            
            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    max_length=max_length,
                    do_sample=temperature > 0,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
            
            # Detokenize response (excluding input tokens)
            response_ids = output_ids[0, len(input_ids[0]):].tolist()
            response = self.detokenize(response_ids)
            
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def chat(
        self,
        message: str,
        image_path: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> str:
        """Chat with the model and maintain conversation history."""
        # Add user message to history
        self.chat_history.append({"role": "user", "content": message})
        
        # Create prompt from chat history
        prompt = ""
        for entry in self.chat_history:
            if entry["role"] == "user":
                prompt += f"User: {entry['content']}\n"
            else:
                prompt += f"Assistant: {entry['content']}\n"
        
        prompt += "Assistant: "
        
        # Generate response
        response = self.generate_response(
            prompt=prompt,
            image_path=image_path,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        
        # Add assistant response to history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def reset_chat(self) -> None:
        """Reset chat history."""
        self.chat_history = []
    
    def launch_web_interface(self) -> None:
        """Launch a Google AI Studio-like web interface."""
        logger.info("Launching web interface")
        
        # Define interface components
        with gr.Blocks(title="Apertis AI Studio", theme=gr.themes.Soft()) as interface:
            # Header
            with gr.Row():
                gr.Markdown("# Apertis AI Studio")
            
            # Tabs for different functionalities
            with gr.Tabs():
                # Chat tab
                with gr.TabItem("Chat"):
                    with gr.Row():
                        with gr.Column(scale=4):
                            chatbot = gr.Chatbot(height=500)
                            with gr.Row():
                                message = gr.Textbox(
                                    placeholder="Type your message here...",
                                    show_label=False,
                                )
                                submit_btn = gr.Button("Send")
                            
                            with gr.Row():
                                clear_btn = gr.Button("Clear Chat")
                        
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="filepath",
                                label="Upload Image (optional)",
                            )
                            with gr.Accordion("Generation Settings", open=False):
                                max_length = gr.Slider(
                                    minimum=10,
                                    maximum=500,
                                    value=100,
                                    step=10,
                                    label="Max Length",
                                )
                                temperature = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.5,
                                    value=0.7,
                                    step=0.1,
                                    label="Temperature",
                                )
                                top_k = gr.Slider(
                                    minimum=1,
                                    maximum=100,
                                    value=50,
                                    step=1,
                                    label="Top K",
                                )
                                top_p = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.9,
                                    step=0.1,
                                    label="Top P",
                                )
                
                # Training tab
                with gr.TabItem("Training"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Model Configuration")
                            model_size = gr.Dropdown(
                                choices=["small", "base", "large"],
                                value="base",
                                label="Model Size",
                            )
                            multimodal_checkbox = gr.Checkbox(
                                value=self.multimodal,
                                label="Multimodal (Text + Image)",
                            )
                            use_expert_system = gr.Checkbox(
                                value=False,
                                label="Use Adaptive Expert System",
                            )
                            
                            gr.Markdown("## Training Data")
                            train_data = gr.File(
                                label="Training Data (JSONL)",
                                file_types=[".jsonl"],
                            )
                            val_data = gr.File(
                                label="Validation Data (JSONL, optional)",
                                file_types=[".jsonl"],
                            )
                            vocab_data = gr.File(
                                label="Vocabulary File (JSON)",
                                file_types=[".json"],
                            )
                            image_dir = gr.Textbox(
                                label="Image Directory (for multimodal training)",
                                placeholder="/path/to/images",
                                visible=self.multimodal,
                            )
                            
                            # Update image_dir visibility based on multimodal checkbox
                            multimodal_checkbox.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[multimodal_checkbox],
                                outputs=[image_dir],
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Training Parameters")
                            batch_size = gr.Slider(
                                minimum=1,
                                maximum=64,
                                value=8,
                                step=1,
                                label="Batch Size",
                            )
                            learning_rate = gr.Slider(
                                minimum=1e-6,
                                maximum=1e-3,
                                value=5e-5,
                                step=1e-6,
                                label="Learning Rate",
                            )
                            num_epochs = gr.Slider(
                                minimum=1,
                                maximum=100,
                                value=3,
                                step=1,
                                label="Number of Epochs",
                            )
                            output_dir = gr.Textbox(
                                label="Output Directory",
                                value="output",
                            )
                            use_wandb = gr.Checkbox(
                                value=False,
                                label="Use Weights & Biases for Logging",
                            )
                            wandb_project = gr.Textbox(
                                label="Weights & Biases Project",
                                value="apertis",
                                visible=False,
                            )
                            
                            # Update wandb_project visibility based on use_wandb checkbox
                            use_wandb.change(
                                fn=lambda x: gr.update(visible=x),
                                inputs=[use_wandb],
                                outputs=[wandb_project],
                            )
                            
                            train_btn = gr.Button("Start Training")
                            training_output = gr.Textbox(
                                label="Training Output",
                                interactive=False,
                                lines=10,
                            )
                
                # Model tab
                with gr.TabItem("Models"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("## Load Model")
                            model_path_input = gr.Textbox(
                                label="Model Path",
                                value=self.model_path or "",
                                placeholder="/path/to/model.pt or /path/to/model_dir",
                            )
                            vocab_path_input = gr.Textbox(
                                label="Vocabulary Path",
                                value=self.vocab_file or "",
                                placeholder="/path/to/vocab.json",
                            )
                            load_model_btn = gr.Button("Load Model")
                            model_info = gr.Textbox(
                                label="Model Information",
                                interactive=False,
                                lines=5,
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## Create New Model")
                            new_model_size = gr.Dropdown(
                                choices=["small", "base", "large"],
                                value="base",
                                label="Model Size",
                            )
                            new_model_multimodal = gr.Checkbox(
                                value=False,
                                label="Multimodal (Text + Image)",
                            )
                            new_model_output = gr.Textbox(
                                label="Output Path",
                                value="models/new_model",
                                placeholder="/path/to/save/model",
                            )
                            create_model_btn = gr.Button("Create Model")
                            create_model_info = gr.Textbox(
                                label="Creation Status",
                                interactive=False,
                                lines=3,
                            )
            
            # Define event handlers
            
            # Chat functionality
            def chat_response(message, image, max_len, temp, k, p, history):
                if not message.strip():
                    return history, ""
                
                # Add user message to history
                history = history + [(message, None)]
                
                # Generate response
                response = self.chat(
                    message=message,
                    image_path=image,
                    max_length=max_len,
                    temperature=temp,
                    top_k=k,
                    top_p=p,
                )
                
                # Update history
                history[-1] = (history[-1][0], response)
                
                return history, ""
            
            submit_btn.click(
                chat_response,
                inputs=[message, image_input, max_length, temperature, top_k, top_p, chatbot],
                outputs=[chatbot, message],
            )
            
            message.submit(
                chat_response,
                inputs=[message, image_input, max_length, temperature, top_k, top_p, chatbot],
                outputs=[chatbot, message],
            )
            
            clear_btn.click(
                lambda: ([], ""),
                outputs=[chatbot, message],
            )
            
            # Model loading
            def load_model_handler(model_path, vocab_path):
                try:
                    self.load_model(model_path)
                    self.load_vocabulary(vocab_path)
                    
                    # Get model information
                    if self.model is not None:
                        config = self.model.config
                        info = f"Model loaded successfully!\n"
                        info += f"Model type: {config.model_type}\n"
                        info += f"Hidden size: {config.hidden_size}\n"
                        info += f"Layers: {config.num_hidden_layers}\n"
                        info += f"Attention heads: {config.num_attention_heads}\n"
                        info += f"Vocabulary size: {len(self.vocab) if self.vocab else 'Unknown'}"
                        return info
                    else:
                        return "Failed to load model."
                except Exception as e:
                    return f"Error loading model: {str(e)}"
            
            load_model_btn.click(
                load_model_handler,
                inputs=[model_path_input, vocab_path_input],
                outputs=[model_info],
            )
            
            # Model creation
            def create_model_handler(model_size, multimodal, output_path):
                try:
                    # Create model
                    model = create_apertis_model(
                        model_size=model_size,
                        multimodal=multimodal,
                    )
                    
                    # Create output directory
                    os.makedirs(output_path, exist_ok=True)
                    
                    # Save model
                    torch.save(model.state_dict(), os.path.join(output_path, "model.pt"))
                    
                    # Save configuration
                    with open(os.path.join(output_path, "config.json"), "w") as f:
                        json.dump(model.config.to_dict(), f, indent=2)
                    
                    # Create a minimal vocabulary if it doesn't exist
                    vocab_path = os.path.join(output_path, "vocab.json")
                    if not os.path.exists(vocab_path):
                        minimal_vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
                        with open(vocab_path, "w") as f:
                            json.dump(minimal_vocab, f, indent=2)
                    
                    return f"Model created successfully at {output_path}"
                except Exception as e:
                    return f"Error creating model: {str(e)}"
            
            create_model_btn.click(
                create_model_handler,
                inputs=[new_model_size, new_model_multimodal, new_model_output],
                outputs=[create_model_info],
            )
            
            # Training functionality
            def start_training(
                model_size, multimodal, expert_system,
                train_file, val_file, vocab_file, img_dir,
                batch, lr, epochs, out_dir, use_wb, wb_project,
            ):
                try:
                    # Create temporary directory for training files
                    temp_dir = tempfile.mkdtemp()
                    
                    # Save uploaded files
                    train_path = os.path.join(temp_dir, "train.jsonl")
                    with open(train_path, "wb") as f:
                        f.write(train_file)
                    
                    vocab_path = os.path.join(temp_dir, "vocab.json")
                    with open(vocab_path, "wb") as f:
                        f.write(vocab_file)
                    
                    val_path = None
                    if val_file is not None:
                        val_path = os.path.join(temp_dir, "val.jsonl")
                        with open(val_path, "wb") as f:
                            f.write(val_file)
                    
                    # Create configuration
                    config = {
                        "data_config": {
                            "train_data_path": train_path,
                            "tokenizer_path": vocab_path,
                            "max_length": 512,
                            "multimodal": multimodal,
                        },
                        "model_config": {
                            "hidden_size": {"small": 512, "base": 768, "large": 1024}[model_size],
                            "num_hidden_layers": {"small": 8, "base": 12, "large": 24}[model_size],
                            "num_attention_heads": {"small": 8, "base": 12, "large": 16}[model_size],
                            "intermediate_size": {"small": 2048, "base": 3072, "large": 4096}[model_size],
                            "use_expert_system": expert_system,
                            "multimodal": multimodal,
                        },
                        "training_config": {
                            "output_dir": out_dir,
                            "batch_size": batch,
                            "learning_rate": lr,
                            "num_epochs": epochs,
                            "use_wandb": use_wb,
                            "wandb_project": wb_project,
                        },
                    }
                    
                    # Add validation data if provided
                    if val_path is not None:
                        config["data_config"]["val_data_path"] = val_path
                    
                    # Add image directory if provided
                    if multimodal and img_dir:
                        config["data_config"]["image_dir"] = img_dir
                    
                    # Save configuration
                    config_path = os.path.join(temp_dir, "config.json")
                    with open(config_path, "w") as f:
                        json.dump(config, f, indent=2)
                    
                    # Start training in a separate thread
                    import threading
                    
                    def train_thread():
                        try:
                            from src.training.pipeline import train_from_config
                            
                            # Train model
                            metrics = train_from_config(config_path)
                            
                            # Clean up
                            shutil.rmtree(temp_dir)
                            
                            return f"Training completed!\nMetrics: {json.dumps(metrics, indent=2)}"
                        except Exception as e:
                            return f"Error during training: {str(e)}"
                    
                    threading.Thread(target=train_thread).start()
                    
                    return f"Training started with configuration:\n{json.dumps(config, indent=2)}\n\nCheck the output directory for results."
                except Exception as e:
                    return f"Error starting training: {str(e)}"
            
            train_btn.click(
                start_training,
                inputs=[
                    model_size, multimodal_checkbox, use_expert_system,
                    train_data, val_data, vocab_data, image_dir,
                    batch_size, learning_rate, num_epochs,
                    output_dir, use_wandb, wandb_project,
                ],
                outputs=[training_output],
            )
        
        # Launch interface
        interface.launch(server_name="0.0.0.0", server_port=self.port, share=self.share)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Apertis Interface")
    parser.add_argument("--model-path", type=str, help="Path to the model file")
    parser.add_argument("--vocab-file", type=str, help="Path to the vocabulary file")
    parser.add_argument("--multimodal", action="store_true", help="Enable multimodal capabilities")
    parser.add_argument("--device", type=str, help="Device to use for inference")
    parser.add_argument("--web", action="store_true", help="Launch web interface")
    parser.add_argument("--port", type=int, default=7860, help="Port for web interface")
    parser.add_argument("--share", action="store_true", help="Create a public link")
    
    args = parser.parse_args()
    
    # Create interface
    interface = ApertisInterface(
        model_path=args.model_path,
        vocab_file=args.vocab_file,
        multimodal=args.multimodal,
        device=args.device,
        web=args.web,
        port=args.port,
        share=args.share,
    )
    
    # If not launching web interface, start interactive CLI
    if not args.web:
        print("Apertis CLI Interface")
        print("Type 'exit' to quit, 'reset' to reset chat history")
        
        while True:
            user_input = input("\nYou: ")
            
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "reset":
                interface.reset_chat()
                print("Chat history reset")
                continue
            
            response = interface.chat(user_input)
            print(f"\nApertis: {response}")


if __name__ == "__main__":
    main()
