import os
import setuptools

# Simple setup without reading external files
setuptools.setup(
    name="apertis",
    version="0.1.0",
    author="CuzImSlymi",
    author_email="slymiservice@gmail.com",
    description="Apertis: A Novel Linear-Time Multimodal LLM Architecture",
    long_description="A user-friendly multimodal LLM framework with linear-time processing capabilities.",
    long_description_content_type="text/markdown",
    url="https://github.com/CuzImSlymi/Apertis-LLM",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pillow>=8.3.0",
        "gradio>=3.0.0",
        "wandb>=0.12.0",
        "torchvision>=0.11.0",
    ],
    entry_points={
        "console_scripts": [
            "apertis=src.apertis_cli:main",
        ],
    }
)
