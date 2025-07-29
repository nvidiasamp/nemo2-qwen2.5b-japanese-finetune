"""
Unit tests for model modules.
"""

import pytest
from nemo_japanese_ft.models import QwenModelConfig, QwenModelManager, ModelUtils


@pytest.mark.unit
class TestQwenModelConfig:
    """Test Qwen model configuration."""
    
    def test_init_default(self):
        """Test default initialization."""
        config = QwenModelConfig()
        assert config.model_size == "0.5b"
        assert config.seq_length == 2048
        assert config.micro_batch_size == 2
        assert config.global_batch_size == 16
        assert config.learning_rate == 3e-4
    
    def test_init_custom(self):
        """Test custom initialization."""
        config = QwenModelConfig(
            model_size="7b",
            seq_length=4096,
            micro_batch_size=1,
            global_batch_size=8,
            learning_rate=1e-4
        )
        assert config.model_size == "7b"
        assert config.seq_length == 4096
        assert config.micro_batch_size == 1
        assert config.global_batch_size == 8
        assert config.learning_rate == 1e-4
    
    def test_validation_invalid_size(self):
        """Test validation of invalid model size."""
        with pytest.raises(ValueError, match="Invalid model size"):
            QwenModelConfig(model_size="invalid")
    
    def test_validation_batch_size(self):
        """Test validation of batch size compatibility."""
        with pytest.raises(ValueError, match="Global batch size.*must be divisible"):
            QwenModelConfig(micro_batch_size=3, global_batch_size=16)
    
    def test_get_hf_model_name(self):
        """Test HuggingFace model name generation."""
        config = QwenModelConfig(model_size="0.5b")
        assert config.get_hf_model_name() == "Qwen/Qwen2.5-0.5B"
        
        config = QwenModelConfig(model_size="7b")
        assert config.get_hf_model_name() == "Qwen/Qwen2.5-7B"
    
    def test_valid_sizes(self):
        """Test all valid model sizes."""
        valid_sizes = ["0.5b", "1.5b", "7b", "14b", "32b", "72b"]
        
        for size in valid_sizes:
            config = QwenModelConfig(model_size=size)
            assert config.model_size == size
            # Should not raise any exceptions
            hf_name = config.get_hf_model_name()
            assert isinstance(hf_name, str)
            assert "Qwen/Qwen2.5" in hf_name


@pytest.mark.unit
class TestQwenModelManager:
    """Test Qwen model manager."""
    
    def test_init(self, sample_model_config):
        """Test manager initialization."""
        manager = QwenModelManager(sample_model_config)
        assert manager.config == sample_model_config
        assert manager._model is None
        assert manager._model_config is None
    
    def test_get_model_info(self, sample_model_config):
        """Test model info generation."""
        manager = QwenModelManager(sample_model_config)
        info = manager.get_model_info()
        
        assert info["model_size"] == sample_model_config.model_size
        assert info["seq_length"] == sample_model_config.seq_length
        assert info["learning_rate"] == sample_model_config.learning_rate
        assert "hf_model_name" in info


@pytest.mark.unit
class TestModelUtils:
    """Test model utilities."""
    
    def test_get_model_size_from_path(self):
        """Test model size extraction from path."""
        # Test various path formats
        test_cases = [
            ("qwen2.5-0.5b.nemo", "0.5b"),
            ("model_7b_checkpoint.nemo", "7b"),
            ("Qwen2.5-1.5B-model.nemo", "1.5b"),
            ("checkpoint_500m.nemo", "0.5b"),
            ("random_name.nemo", None),
        ]
        
        for path, expected in test_cases:
            result = ModelUtils.get_model_size_from_path(path)
            assert result == expected
    
    def test_validate_nemo_file_nonexistent(self):
        """Test validation of non-existent file."""
        assert not ModelUtils.validate_nemo_file("/nonexistent/file.nemo")
    
    def test_validate_nemo_file_wrong_extension(self, temp_dir):
        """Test validation of file with wrong extension."""
        from pathlib import Path
        
        wrong_file = Path(temp_dir) / "not_nemo.txt"
        wrong_file.write_text("dummy content")
        
        assert not ModelUtils.validate_nemo_file(str(wrong_file))
    
    def test_validate_nemo_file_empty(self, temp_dir):
        """Test validation of empty file."""
        from pathlib import Path
        
        empty_file = Path(temp_dir) / "empty.nemo"
        empty_file.touch()  # Create empty file
        
        assert not ModelUtils.validate_nemo_file(str(empty_file))
    
    def test_validate_nemo_file_valid(self, mock_nemo_file):
        """Test validation of valid mock file."""
        assert ModelUtils.validate_nemo_file(mock_nemo_file) 