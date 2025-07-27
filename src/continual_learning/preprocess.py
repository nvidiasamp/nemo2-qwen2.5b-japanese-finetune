#!/usr/bin/env python3
"""
Japanese Wikipedia Data to NeMo 2.0 SFT Format Conversion Script
Input format: {"text": "article content", "meta": {"id": "ID", "title": "title", "url": "URL"}}
Output format: {"input": "question", "output": "answer"}
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Length settings suitable for SFT
MAX_OUTPUT_LENGTH = 1500  # Maximum character count for answers
MIN_OUTPUT_LENGTH = 50  # Minimum character count for answers
MAX_INPUT_LENGTH = 200  # Maximum character count for questions
CHUNK_SIZE = 800  # Chunk size (character count)


def clean_text(text: str) -> str:
    """Clean up text"""
    # Clean up excessive line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove excessive spaces
    text = re.sub(r" +", " ", text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks of appropriate length"""
    # Split by sentences
    sentences = re.split(r"[。！？]", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Check length if sentence is added
        potential_chunk = current_chunk + sentence + "。"

        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            # Save current chunk (if it meets minimum length)
            if len(current_chunk) >= MIN_OUTPUT_LENGTH:
                chunks.append(current_chunk)
            current_chunk = sentence + "。"

    # Add last chunk
    if len(current_chunk) >= MIN_OUTPUT_LENGTH:
        chunks.append(current_chunk)

    return chunks


def create_summary(text: str, max_length: int = 300) -> str:
    """Create article summary (using the beginning part)"""
    # Use first paragraph or up to specified length as summary
    paragraphs = text.split("\n\n")
    summary = paragraphs[0] if paragraphs else text

    if len(summary) > max_length:
        # Truncate at sentence boundaries
        sentences = re.split(r"[。！？]", summary)
        result = ""
        for sentence in sentences:
            if len(result + sentence + "。") <= max_length:
                result += sentence + "。"
            else:
                break
        summary = result if result else summary[:max_length]

    return summary.strip()


def create_qa_pairs_from_article(article: Dict) -> List[Dict]:
    """Generate multiple question-answer pairs of appropriate length from article"""
    text = article["text"]
    title = article["meta"]["title"]

    # Clean up text
    cleaned_text = clean_text(text)

    # Skip if article is too short
    if len(cleaned_text) < MIN_OUTPUT_LENGTH:
        return []

    qa_pairs = []

    # 1. Summary version question-answer pairs
    summary = create_summary(cleaned_text)
    if len(summary) >= MIN_OUTPUT_LENGTH:
        qa_pairs.extend(
            [
                {"input": f"{title}とは何ですか？", "output": summary},
                {
                    "input": f"{title}について簡潔に説明してください。",
                    "output": summary,
                },
            ]
        )

    # 2. Detailed question-answer pairs by chunk splitting
    chunks = split_text_into_chunks(cleaned_text)

    for i, chunk in enumerate(chunks):
        # Length limit check
        if len(chunk) > MAX_OUTPUT_LENGTH:
            chunk = chunk[:MAX_OUTPUT_LENGTH] + "..."

        if len(chunk) < MIN_OUTPUT_LENGTH:
            continue

        # Chunk-based question patterns
        if i == 0:
            # First chunk treated as overview
            qa_pairs.append(
                {"input": f"{title}について教えてください。", "output": chunk}
            )
        else:
            # Other chunks treated as detailed information
            qa_pairs.append(
                {
                    "input": f"{title}についてさらに詳しく教えてください。",
                    "output": chunk,
                }
            )

        # Questions based on specific keywords
        if "歴史" in chunk:
            qa_pairs.append(
                {"input": f"{title}の歴史について教えてください。", "output": chunk}
            )

        if any(keyword in chunk for keyword in ["特徴", "性質", "機能"]):
            qa_pairs.append(
                {"input": f"{title}の特徴について説明してください。", "output": chunk}
            )

    # 3. If entire article is short, use the whole thing
    if len(cleaned_text) <= MAX_OUTPUT_LENGTH and not qa_pairs:
        qa_pairs.append(
            {"input": f"{title}について教えてください。", "output": cleaned_text}
        )

    # Question length check and adjustment
    filtered_qa_pairs = []
    for qa in qa_pairs:
        if (
            len(qa["input"]) <= MAX_INPUT_LENGTH
            and MIN_OUTPUT_LENGTH <= len(qa["output"]) <= MAX_OUTPUT_LENGTH
        ):
            filtered_qa_pairs.append(qa)

    return filtered_qa_pairs


def process_jsonl_file(
    input_file: str, max_articles: Optional[int] = None
) -> List[Dict]:
    """Process JSONL file to generate QA pairs"""
    qa_pairs = []
    processed_count = 0

    print(f"Processing {input_file}...")

    with open(input_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if max_articles and processed_count >= max_articles:
                break

            try:
                article = json.loads(line.strip())
                if not article.get("text") or not article.get("meta", {}).get("title"):
                    continue

                # Skip articles that are too short (ensure minimum quality)
                if len(article["text"]) < 100:
                    continue

                # Generate question-answer pairs
                pairs = create_qa_pairs_from_article(article)
                qa_pairs.extend(pairs)
                processed_count += 1

                if processed_count % 1000 == 0:
                    print(
                        f"  Processed {processed_count} articles, generated {len(qa_pairs)} QA pairs"
                    )

            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error on line {line_no}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_no}: {e}")
                continue

    print(
        f"Completed processing {input_file}: {processed_count} articles -> {len(qa_pairs)} QA pairs"
    )
    return qa_pairs


def save_qa_pairs(qa_pairs: List[Dict], output_file: str) -> None:
    """Save QA pairs in JSONL format"""
    print(f"Saving {len(qa_pairs)} QA pairs to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for qa_pair in qa_pairs:
            json.dump(qa_pair, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved to {output_file}")


def main():
    # Configuration
    input_dir = "/workspace/data/ja_wiki"
    output_dir = "/workspace/data/training_data"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get input file list
    input_files = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".jsonl"):
            input_files.append(os.path.join(input_dir, file))

    print(f"Found {len(input_files)} JSONL files in {input_dir}")

    # Process each file
    all_qa_pairs = []

    # Process training files (train_*.jsonl) - use all data
    train_files = [f for f in input_files if "train_" in os.path.basename(f)]
    for train_file in train_files:
        # Process all articles
        qa_pairs = process_jsonl_file(train_file, max_articles=None)
        all_qa_pairs.extend(qa_pairs)

    # Process validation files - use all data
    validation_files = [f for f in input_files if "validation_" in os.path.basename(f)]
    validation_qa_pairs = []
    for val_file in validation_files:
        qa_pairs = process_jsonl_file(val_file, max_articles=None)
        validation_qa_pairs.extend(qa_pairs)

    # Shuffle
    random.seed(42)
    random.shuffle(all_qa_pairs)
    random.shuffle(validation_qa_pairs)

    # No data limit - use all data
    print(
        f"Using all data: {len(all_qa_pairs)} training samples, {len(validation_qa_pairs)} validation samples"
    )

    # Save
    save_qa_pairs(all_qa_pairs, os.path.join(output_dir, "training.jsonl"))
    save_qa_pairs(validation_qa_pairs, os.path.join(output_dir, "validation.jsonl"))

    print(f"\n=== Conversion Complete ===")
    print(f"Training samples: {len(all_qa_pairs)}")
    print(f"Validation samples: {len(validation_qa_pairs)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
