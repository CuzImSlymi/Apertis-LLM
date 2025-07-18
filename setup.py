import os
import setuptools

# Simple setup without reading external files
setuptools.setup(
    name="apertis",
    version="0.2.0",
    author="CuzImSlymi",
    author_email="slymiservice@gmail.com",
    description="Apertis: A Novel Linear-Time Multimodal LLM Architecture with a Scalable Data Pipeline",
    long_description="A user-friendly multimodal LLM framework with linear-time processing capabilities and a SOTA-level distributed data processing pipeline.",
    long_description_content_type="text/markdown",
    url="https://github.com/CuzImSlymi/Apertis-LLM",
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["apertis_cli"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.0.0",
        "tokenizers>=0.10.0",
        "flash-attn==2.5.8; platform_system != 'Windows'",
        "numpy>=1.20.0",
        "tqdm>=4.62.0",
        "pillow>=8.3.0",
        "setuptools>=58.0.0",
        "wheel>=0.37.0",
        "pyyaml>=6.0",
        "gradio>=3.0.0",
        "wandb>=0.12.0",
        "pyspark>=3.4.0",
        "pyarrow>=14.0.0",
        "beautifulsoup4>=4.9.3",
        "datasketch>=1.5.9",
        "fasttext-wheel>=0.9.2",
        "warcio>=1.7.4",
    ],
    entry_points={
        "console_scripts": [
            "apertis=apertis_cli:main",
        ],
    }
)