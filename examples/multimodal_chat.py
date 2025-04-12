#!/usr/bin/env python3
"""
Multimodal Chat Example

This example demonstrates how to use Apertis for multimodal chat with both text and images.
It shows how to load images and get the model to describe or answer questions about them.
"""

import os
import sys
import torch
from PIL import Image

# Add the parent directory to the path so we can import the src modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.inference.interface import ApertisInference

def main():
    # Check if model exists, if not create a tiny test model
    model_path = "models/test_model.pt"
    vocab_file = "data/vocab.json"
    
    if not os.path.exists(model_path):
        print("Test model not found. Creating a small test model...")
        os.makedirs("models", exist_ok=True)
        
        from src.model.core import create_apertis_model
        model = create_apertis_model(model_size='small', multimodal=True)
        torch.save(model.state_dict(), model_path)
        print(f"Test model created at {model_path}")
    
    if not os.path.exists(vocab_file):
        print("Vocabulary file not found. Creating a sample vocabulary...")
        os.makedirs("data", exist_ok=True)
        
        import json
        vocab = {
            "<pad>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "<unk>": 3,
            "the": 4,
            "a": 5,
            "an": 6,
            "is": 7,
            "was": 8,
            "are": 9,
            "were": 10
        }
        
        with open(vocab_file, 'w') as f:
            json.dump(vocab, f, indent=2)
        print(f"Sample vocabulary created at {vocab_file}")
    
    # Initialize the inference engine
    print("Initializing Apertis multimodal inference engine...")
    inference = ApertisInference(
        model_path=model_path,
        vocab_file=vocab_file,
        model_size="small",
        multimodal=True
    )
    
    print("\n" + "="*50)
    print("Welcome to Apertis Multimodal Chat Example!")
    print("Type 'exit' to quit the chat.")
    print("Type 'image <path>' to load an image.")
    print("="*50 + "\n")
    
    # Simple chat loop
    chat_history = [
        {"role": "system", "content": "You are Apertis, a helpful and friendly AI assistant that can understand both text and images."}
    ]
    
    current_image = None
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check for exit command
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        # Check for image loading command
        if user_input.lower().startswith("image "):
            image_path = user_input[6:].strip()
            
            try:
                current_image = Image.open(image_path).convert("RGB")
                print(f"Image loaded: {image_path}")
                continue
            except Exception as e:
                print(f"Error loading image: {e}")
                continue
        
        # Add user message to history
        chat_history.append({"role": "user", "content": user_input})
        
        # Generate response
        print("Apertis: ", end="", flush=True)
        
        try:
            # Stream response
            response_text = ""
            for text in inference.chat(
                messages=chat_history,
                image=current_image,
                stream=True
            ):
                # Print new content
                print(text[len(response_text):], end="", flush=True)
                response_text = text
            
            print()  # New line after response
            
            # Add assistant response to history
            chat_history.append({"role": "assistant", "content": response_text})
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
