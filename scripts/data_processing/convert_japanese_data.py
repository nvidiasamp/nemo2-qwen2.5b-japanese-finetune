#!/usr/bin/env python3
"""
Convert Japanese Wikipedia data to NeMo SFT format.

This script converts Japanese Wikipedia JSONL files to question-answer format
suitable for NeMo fine-tuning.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nemo_japanese_ft.data import JapaneseWikipediaConverter
from nemo_japanese_ft.utils import setup_logging

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert Japanese Wikipedia data to NeMo SFT format"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing Japanese Wikipedia JSONL files"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str,
        required=True,
        help="Directory to save converted training data"
    )
    
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=1500,
        help="Maximum output length for answers"
    )
    
    parser.add_argument(
        "--min-output-length",
        type=int,
        default=50,
        help="Minimum output length for answers"
    )
    
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=200,
        help="Maximum input length for questions"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=800,
        help="Chunk size for splitting long articles"
    )
    
    parser.add_argument(
        "--train-pattern",
        type=str,
        default="train_",
        help="Pattern to identify training files"
    )
    
    parser.add_argument(
        "--validation-pattern",
        type=str,
        default="validation_",
        help="Pattern to identify validation files"
    )
    
    args = parser.parse_args()
    
    logger.info("=== Japanese Data Conversion ===")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Max output length: {args.max_output_length}")
    logger.info(f"Min output length: {args.min_output_length}")
    logger.info(f"Max input length: {args.max_input_length}")
    logger.info(f"Chunk size: {args.chunk_size}")
    
    # Create converter
    converter = JapaneseWikipediaConverter(
        max_output_length=args.max_output_length,
        min_output_length=args.min_output_length,
        max_input_length=args.max_input_length,
        chunk_size=args.chunk_size,
    )
    
    try:
        # Convert data
        converter.convert(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            train_pattern=args.train_pattern,
            validation_pattern=args.validation_pattern,
        )
        
        logger.info("Data conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Data conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 