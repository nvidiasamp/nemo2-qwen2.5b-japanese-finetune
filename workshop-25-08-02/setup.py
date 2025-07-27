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
    name="japanese-continual-learning-nemo",
    version=__version__,
    author="Your Name",
    author_email="your.email@example.com",
    description="Japanese Continual Learning with NeMo 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/japanese-continual-learning-nemo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
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
    },
    entry_points={
        "console_scripts": [
            "nemo-jp-train=src.training.continual_learning:main",
            "nemo-jp-finetune=src.training.peft_finetuning:main",
            "nemo-jp-evaluate=src.evaluation.evaluate_model:main",
            "nemo-jp-inference=src.inference.generate_text:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/japanese-continual-learning-nemo/issues",
        "Source": "https://github.com/your-username/japanese-continual-learning-nemo",
        "Documentation": "https://japanese-continual-learning-nemo.readthedocs.io/",
    },
    keywords=[
        "nemo",
        "japanese",
        "continual learning",
        "peft",
        "lora",
        "nlp",
        "machine learning",
        "deep learning",
        "language models",
    ],
    include_package_data=True,
    zip_safe=False,
) 