#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Read version
__version__ = "0.1.0"

setup(
    name="nemo2-qwen2.5b-japanese-finetune",
    version=__version__,
    author="NeMo Japanese FT Team",
    author_email="nemo-japanese-ft@example.com",
    description="Japanese Fine-tuning for Qwen2.5 Models using NVIDIA NeMo 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.20.0",
            "mypy>=0.991",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=8.0.0",
            "notebook>=6.0.0",
        ],
        "gpu": [
            "apex>=0.1",
            "flash-attn>=2.0.0",
            "triton>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nemo-jp-convert-data=scripts.data_processing.convert_japanese_data:main",
            "nemo-jp-convert-model=scripts.model_conversion.hf_to_nemo:main",
            "nemo-jp-train-peft=scripts.training.run_peft_training:main",
            "nemo-jp-train-sft=scripts.training.run_sft_training:main",
            "nemo-jp-train-continual=scripts.training.run_continual_pretraining:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune/issues",
        "Source": "https://github.com/your-username/nemo2-qwen2.5b-japanese-finetune",
        "Documentation": "https://nemo2-qwen2.5b-japanese-finetune.readthedocs.io/",
    },
    keywords=[
        "nemo",
        "qwen",
        "japanese",
        "fine-tuning",
        "peft",
        "lora",
        "sft",
        "nlp",
        "machine learning",
        "deep learning",
        "language models",
        "transformers",
    ],
    include_package_data=True,
    zip_safe=False,
) 