import os
import sys
import platform
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("Error: Apertis requires Python 3.8 or higher.")
        print(f"Your Python version: {platform.python_version()}")
        return False
    return True

def check_dependencies():
    try:
        import torch
        import gradio
        import numpy
        import tqdm
        import PIL
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    print("Installing dependencies...")
    pip_cmd = "pip"
    if platform.system() == "Windows":
        pip_cmd = "pip"
    else:
        try:
            subprocess.run(["pip3", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pip_cmd = "pip3"
        except (subprocess.SubprocessError, FileNotFoundError):
            pip_cmd = "pip"
    try:
        subprocess.run([pip_cmd, "install", "-r", os.path.join(os.path.dirname(__file__), "..", "requirements.txt")], check=True)
        print("Dependencies installed successfully!")
        return True
    except subprocess.SubprocessError as e:
        print(f"Error installing dependencies: {e}")
        return False

def setup_environment():
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))
    os.makedirs(os.path.join(current_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(current_dir, "output"), exist_ok=True)
    return True

def launch_web_interface():
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from src.inference.interface import ApertisInterface
    print("Launching Apertis web interface...")
    interface = ApertisInterface(web=False, port=7860, share=False)
    webbrowser.open("http://localhost:7860")
    import gradio as gr
    interface_blocks = gr.Blocks(title="Apertis AI Studio", theme=gr.themes.Soft())
    interface.launch_web_interface()
    return True

def main():
    print("=" * 60)
    print("  Welcome to Apertis - Advanced LLM Architecture")
    print("=" * 60)
    if not check_python_version():
        input("Press Enter to exit...")
        return
    if not setup_environment():
        input("Press Enter to exit...")
        return
    if not check_dependencies():
        print("Some dependencies are missing. Would you like to install them now?")
        choice = input("Install dependencies? [Y/n]: ").strip().lower()
        if choice in ["", "y", "yes"]:
            if not install_dependencies():
                input("Press Enter to exit...")
                return
        else:
            print("Cannot continue without dependencies.")
            input("Press Enter to exit...")
            return
    try:
        launch_web_interface()
        print("\nApertis web interface is running at http://localhost:7860")
        print("Press Ctrl+C to stop the server.")
        import gradio as gr
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Apertis...")
        import gradio as gr
        gr.close_all()
    except Exception as e:
        print(f"\nError: {e}")
        import gradio as gr
        gr.close_all()
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()