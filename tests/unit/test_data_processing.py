"""
Unit tests for data processing modules.
"""

import pytest
import tempfile
import json
from pathlib import Path

from nemo_japanese_ft.data import JapaneseWikipediaConverter, QADataProcessor


@pytest.mark.unit
class TestJapaneseWikipediaConverter:
    """Test Japanese Wikipedia data converter."""
    
    def test_init(self):
        """Test converter initialization."""
        converter = JapaneseWikipediaConverter()
        assert converter.max_output_length == 1500
        assert converter.min_output_length == 50
        assert converter.max_input_length == 200
        assert converter.chunk_size == 800
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        converter = JapaneseWikipediaConverter()
        
        # Test excessive newlines
        text = "Line 1\n\n\n\nLine 2"
        cleaned = converter.clean_text(text)
        assert cleaned == "Line 1\n\nLine 2"
        
        # Test excessive spaces
        text = "Word1     Word2"
        cleaned = converter.clean_text(text)
        assert cleaned == "Word1 Word2"
    
    def test_create_summary(self):
        """Test summary creation."""
        converter = JapaneseWikipediaConverter()
        
        # Test normal summary
        text = "これは最初の段落です。これは詳細な説明です。\n\n二番目の段落です。"
        summary = converter.create_summary(text, max_length=50)
        assert len(summary) <= 50
        assert "これは最初の段落です。" in summary
    
    def test_split_text_into_chunks(self):
        """Test text chunking."""
        converter = JapaneseWikipediaConverter(chunk_size=100, min_output_length=20)
        
        text = "短い文です。" * 20  # Create text longer than chunk size
        chunks = converter.split_text_into_chunks(text)
        
        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 100
            assert len(chunk) >= 20
    
    def test_create_qa_pairs_from_article(self, sample_japanese_article):
        """Test Q&A pair generation from article."""
        converter = JapaneseWikipediaConverter()
        
        qa_pairs = converter.create_qa_pairs_from_article(sample_japanese_article)
        
        assert len(qa_pairs) > 0
        
        # Check format
        for qa in qa_pairs:
            assert "input" in qa
            assert "output" in qa
            assert isinstance(qa["input"], str)
            assert isinstance(qa["output"], str)
            assert len(qa["input"]) > 0
            assert len(qa["output"]) > 0
        
        # Check for expected question patterns
        questions = [qa["input"] for qa in qa_pairs]
        title = sample_japanese_article["meta"]["title"]
        assert any(f"{title}とは何ですか？" in q for q in questions)
    
    def test_save_qa_pairs(self, temp_dir):
        """Test saving Q&A pairs to file."""
        converter = JapaneseWikipediaConverter()
        output_file = Path(temp_dir) / "test_output.jsonl"
        
        qa_pairs = [
            {"input": "質問1", "output": "回答1"},
            {"input": "質問2", "output": "回答2"},
        ]
        
        converter.save_qa_pairs(qa_pairs, str(output_file))
        
        # Verify file was created
        assert output_file.exists()
        
        # Verify content
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Parse and verify JSON
        for i, line in enumerate(lines):
            data = json.loads(line.strip())
            assert data == qa_pairs[i]


@pytest.mark.unit 
class TestQADataProcessor:
    """Test Q&A data processor."""
    
    def test_validate_data_format(self, temp_data_dir):
        """Test data format validation."""
        processor = QADataProcessor(dataset_root=temp_data_dir)
        
        # Test valid file
        train_file = Path(temp_data_dir) / "training.jsonl"
        assert processor.validate_data_format(str(train_file))
    
    def test_validate_data_format_invalid(self, temp_dir):
        """Test data format validation with invalid data."""
        processor = QADataProcessor(dataset_root=temp_dir)
        
        # Create invalid data file
        invalid_file = Path(temp_dir) / "invalid.jsonl"
        with open(invalid_file, "w", encoding="utf-8") as f:
            f.write('{"invalid": "format"}\n')  # Missing required fields
        
        assert not processor.validate_data_format(str(invalid_file))
    
    def test_get_data_stats(self, temp_data_dir):
        """Test data statistics generation."""
        processor = QADataProcessor(dataset_root=temp_data_dir)
        
        train_file = Path(temp_data_dir) / "training.jsonl"
        stats = processor.get_data_stats(str(train_file))
        
        assert stats["total_samples"] == 2  # From conftest.py fixture
        assert stats["avg_input_length"] > 0
        assert stats["avg_output_length"] > 0
        assert stats["max_input_length"] >= stats["min_input_length"]
        assert stats["max_output_length"] >= stats["min_output_length"] 