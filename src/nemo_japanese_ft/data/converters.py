#!/usr/bin/env python3
"""
Data converters for transforming various data formats to NeMo-compatible formats.
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..utils import setup_logging

logger = setup_logging(__name__)


class DataConverter(ABC):
    """Abstract base class for data converters."""
    
    @abstractmethod
    def convert(self, input_path: str, output_path: str, **kwargs) -> None:
        """Convert data from input format to output format."""
        pass


class JapaneseWikipediaConverter(DataConverter):
    """
    Converter for Japanese Wikipedia data to NeMo SFT format.
    
    Input format: {"text": "article content", "meta": {"id": "ID", "title": "title", "url": "URL"}}
    Output format: {"input": "question", "output": "answer"}
    """
    
    def __init__(
        self,
        max_output_length: int = 1500,
        min_output_length: int = 50,
        max_input_length: int = 200,
        chunk_size: int = 800,
    ):
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        self.max_input_length = max_input_length
        self.chunk_size = chunk_size
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        # Remove excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove excessive spaces
        text = re.sub(r" +", " ", text)
        return text.strip()

    def split_text_into_chunks(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """Split text into appropriate chunks for training."""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        # Split by sentences
        sentences = re.split(r"[。！？]", text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check length if we add this sentence
            potential_chunk = current_chunk + sentence + "。"

            if len(potential_chunk) <= chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum length
                if len(current_chunk) >= self.min_output_length:
                    chunks.append(current_chunk)
                current_chunk = sentence + "。"

        # Add the last chunk
        if len(current_chunk) >= self.min_output_length:
            chunks.append(current_chunk)

        return chunks

    def create_summary(self, text: str, max_length: int = 300) -> str:
        """Create a summary from article text."""
        # Use first paragraph or specified length as summary
        paragraphs = text.split("\n\n")
        summary = paragraphs[0] if paragraphs else text

        if len(summary) > max_length:
            # Truncate at sentence boundary
            sentences = re.split(r"[。！？]", summary)
            result = ""
            for sentence in sentences:
                if len(result + sentence + "。") <= max_length:
                    result += sentence + "。"
                else:
                    break
            summary = result if result else summary[:max_length]

        return summary.strip()

    def create_qa_pairs_from_article(self, article: Dict) -> List[Dict]:
        """Generate multiple question-answer pairs from a single article."""
        text = article["text"]
        title = article["meta"]["title"]

        # Clean text
        cleaned_text = self.clean_text(text)

        # Skip if article is too short
        if len(cleaned_text) < self.min_output_length:
            return []

        qa_pairs = []

        # 1. Summary-based Q&A pairs
        summary = self.create_summary(cleaned_text)
        if len(summary) >= self.min_output_length:
            qa_pairs.extend([
                {"input": f"{title}とは何ですか？", "output": summary},
                {"input": f"{title}について簡潔に説明してください。", "output": summary},
            ])

        # 2. Chunk-based detailed Q&A pairs
        chunks = self.split_text_into_chunks(cleaned_text)

        for i, chunk in enumerate(chunks):
            # Length check
            if len(chunk) > self.max_output_length:
                chunk = chunk[:self.max_output_length] + "..."

            if len(chunk) < self.min_output_length:
                continue

            # Chunk-based question patterns
            if i == 0:
                # First chunk as overview
                qa_pairs.append(
                    {"input": f"{title}について教えてください。", "output": chunk}
                )
            else:
                # Other chunks as detailed information
                qa_pairs.append({
                    "input": f"{title}についてさらに詳しく教えてください。",
                    "output": chunk,
                })

            # Keyword-based questions
            if "歴史" in chunk:
                qa_pairs.append(
                    {"input": f"{title}の歴史について教えてください。", "output": chunk}
                )

            if any(keyword in chunk for keyword in ["特徴", "性質", "機能"]):
                qa_pairs.append(
                    {"input": f"{title}の特徴について説明してください。", "output": chunk}
                )

        # 3. Use full article if short and no pairs generated
        if len(cleaned_text) <= self.max_output_length and not qa_pairs:
            qa_pairs.append(
                {"input": f"{title}について教えてください。", "output": cleaned_text}
            )

        # Filter by length constraints
        filtered_qa_pairs = []
        for qa in qa_pairs:
            if (
                len(qa["input"]) <= self.max_input_length
                and self.min_output_length <= len(qa["output"]) <= self.max_output_length
            ):
                filtered_qa_pairs.append(qa)

        return filtered_qa_pairs

    def process_jsonl_file(
        self, 
        input_file: str, 
        max_articles: Optional[int] = None
    ) -> List[Dict]:
        """Process a JSONL file and generate QA pairs."""
        qa_pairs = []
        processed_count = 0

        logger.info(f"Processing {input_file}...")

        with open(input_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if max_articles and processed_count >= max_articles:
                    break

                try:
                    article = json.loads(line.strip())
                    if not article.get("text") or not article.get("meta", {}).get("title"):
                        continue

                    # Skip very short articles
                    if len(article["text"]) < 100:
                        continue

                    # Generate Q&A pairs
                    pairs = self.create_qa_pairs_from_article(article)
                    qa_pairs.extend(pairs)
                    processed_count += 1

                    if processed_count % 1000 == 0:
                        logger.info(
                            f"  Processed {processed_count} articles, "
                            f"generated {len(qa_pairs)} QA pairs"
                        )

                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error on line {line_no}: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"Error processing line {line_no}: {e}")
                    continue

        logger.info(
            f"Completed processing {input_file}: "
            f"{processed_count} articles -> {len(qa_pairs)} QA pairs"
        )
        return qa_pairs

    def save_qa_pairs(self, qa_pairs: List[Dict], output_file: str) -> None:
        """Save QA pairs to JSONL format."""
        logger.info(f"Saving {len(qa_pairs)} QA pairs to {output_file}")

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for qa_pair in qa_pairs:
                json.dump(qa_pair, f, ensure_ascii=False)
                f.write("\n")

        logger.info(f"Saved to {output_file}")

    def convert(
        self, 
        input_dir: str, 
        output_dir: str, 
        train_pattern: str = "train_",
        validation_pattern: str = "validation_",
        **kwargs
    ) -> None:
        """
        Convert Japanese Wikipedia JSONL files to NeMo SFT format.
        
        Args:
            input_dir: Directory containing input JSONL files
            output_dir: Directory to save converted files
            train_pattern: Pattern to identify training files
            validation_pattern: Pattern to identify validation files
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get input files
        input_files = []
        for file in sorted(os.listdir(input_dir)):
            if file.endswith(".jsonl"):
                input_files.append(os.path.join(input_dir, file))

        logger.info(f"Found {len(input_files)} JSONL files in {input_dir}")

        # Process training files
        all_qa_pairs = []
        train_files = [f for f in input_files if train_pattern in os.path.basename(f)]
        for train_file in train_files:
            qa_pairs = self.process_jsonl_file(train_file, max_articles=None)
            all_qa_pairs.extend(qa_pairs)

        # Process validation files  
        validation_qa_pairs = []
        validation_files = [f for f in input_files if validation_pattern in os.path.basename(f)]
        for val_file in validation_files:
            qa_pairs = self.process_jsonl_file(val_file, max_articles=None)
            validation_qa_pairs.extend(qa_pairs)

        # Shuffle data
        random.seed(42)
        random.shuffle(all_qa_pairs)
        random.shuffle(validation_qa_pairs)

        logger.info(
            f"Using all data: {len(all_qa_pairs)} training samples, "
            f"{len(validation_qa_pairs)} validation samples"
        )

        # Save results
        self.save_qa_pairs(all_qa_pairs, os.path.join(output_dir, "training.jsonl"))
        self.save_qa_pairs(validation_qa_pairs, os.path.join(output_dir, "validation.jsonl"))

        logger.info(f"=== Conversion Complete ===")
        logger.info(f"Training samples: {len(all_qa_pairs)}")
        logger.info(f"Validation samples: {len(validation_qa_pairs)}")
        logger.info(f"Output directory: {output_dir}") 