"""
Pytest configuration and fixtures for NeMo Japanese Fine-tuning tests.
"""

import pytest
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nemo_japanese_ft.models import QwenModelConfig
from nemo_japanese_ft.utils import setup_logging

# Setup logging for tests
logger = setup_logging(__name__, level="DEBUG")


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return QwenModelConfig(
        model_size="0.5b",
        seq_length=512,  # Smaller for tests
        micro_batch_size=1,
        global_batch_size=2,
        learning_rate=1e-4,
        max_steps=10,
        val_check_interval=5,
    )


@pytest.fixture
def sample_japanese_article():
    """Sample Japanese article data for testing."""
    return {
        "text": "日本は東アジアに位置する島国である。首都は東京。人口は約1億2千万人。日本の文化は独特で、古い伝統と現代技術が共存している。桜の季節は特に美しく、多くの観光客が訪れる。日本料理は世界的に有名で、寿司、ラーメン、天ぷらなどが知られている。",
        "meta": {
            "id": "test_001",
            "title": "日本",
            "url": "https://example.com/japan"
        }
    }


@pytest.fixture
def sample_qa_data():
    """Sample Q&A data for testing."""
    return [
        {
            "input": "日本とは何ですか？",
            "output": "日本は東アジアに位置する島国である。首都は東京。"
        },
        {
            "input": "日本の人口はどのくらいですか？",
            "output": "日本の人口は約1億2千万人です。"
        },
        {
            "input": "日本料理について教えてください。",
            "output": "日本料理は世界的に有名で、寿司、ラーメン、天ぷらなどが知られています。"
        }
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_data_dir(temp_dir):
    """Temporary directory with sample data files."""
    data_dir = Path(temp_dir) / "data"
    data_dir.mkdir()
    
    # Create sample training data
    train_data = [
        {"input": "質問1", "output": "回答1"},
        {"input": "質問2", "output": "回答2"},
    ]
    
    # Create sample validation data
    val_data = [
        {"input": "検証質問1", "output": "検証回答1"},
    ]
    
    # Write training data
    with open(data_dir / "training.jsonl", "w", encoding="utf-8") as f:
        for item in train_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    # Write validation data
    with open(data_dir / "validation.jsonl", "w", encoding="utf-8") as f:
        for item in val_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")
    
    return str(data_dir)


@pytest.fixture
def mock_nemo_file(temp_dir):
    """Mock .nemo file for testing."""
    nemo_path = Path(temp_dir) / "test_model.nemo"
    
    # Create a dummy file (in real scenarios this would be a proper NeMo model)
    with open(nemo_path, "wb") as f:
        f.write(b"mock nemo model data")
    
    return str(nemo_path)


@pytest.fixture(autouse=True)
def clean_loggers():
    """Clean up loggers after each test to avoid duplicate handlers."""
    yield
    # Clear any handlers that might have been added during tests
    import logging
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        if logger_name.startswith("nemo_japanese_ft"):
            logger = logging.getLogger(logger_name)
            logger.handlers.clear()


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests that don't require external dependencies"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests that may require NeMo/GPU"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU access"
    ) 