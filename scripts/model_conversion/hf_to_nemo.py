#!/usr/bin/env python3
"""
Convert HuggingFace models to NeMo format.

This script converts HuggingFace Qwen models to NeMo .nemo format
for use in NeMo training pipelines.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from nemo_japanese_ft.models import import_model_from_hf, ModelUtils
from nemo_japanese_ft.utils import setup_logging

logger = setup_logging(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to NeMo format"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B",
        help="HuggingFace model name to convert"
    )
    
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the converted .nemo file"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the converted model file"
    )
    
    args = parser.parse_args()
    
    logger.info("=== HuggingFace to NeMo Conversion ===")
    logger.info(f"Source model: {args.model_name}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Overwrite: {args.overwrite}")
    
    try:
        # Convert model
        logger.info("Starting model conversion...")
        output_path = import_model_from_hf(
            model_name=args.model_name,
            output_path=args.output_path,
            overwrite=args.overwrite
        )
        
        logger.info(f"Model conversion completed: {output_path}")
        
        # Validate converted model if requested
        if args.validate:
            logger.info("Validating converted model...")
            if ModelUtils.validate_nemo_file(output_path):
                logger.info("Model validation passed!")
            else:
                logger.error("Model validation failed!")
                sys.exit(1)
        
        logger.info("Conversion process completed successfully!")
        
    except Exception as e:
        logger.error(f"Model conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 