#!/usr/bin/env python3
"""
日本語WikipediaデータをNeMo 2.0 SFT形式に変換するスクリプト
入力形式: {"text": "記事内容", "meta": {"id": "ID", "title": "タイトル", "url": "URL"}}
出力形式: {"input": "質問", "output": "回答"}
"""

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# SFTに適した長さの設定
MAX_OUTPUT_LENGTH = 1500  # 回答の最大文字数
MIN_OUTPUT_LENGTH = 50  # 回答の最小文字数
MAX_INPUT_LENGTH = 200  # 質問の最大文字数
CHUNK_SIZE = 800  # チャンクサイズ（文字数）


def clean_text(text: str) -> str:
    """テキストをクリーンアップ"""
    # 余分な改行を整理
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 余分な空白を削除
    text = re.sub(r" +", " ", text)
    return text.strip()


def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """テキストを適切な長さのチャンクに分割"""
    # 文単位で分割
    sentences = re.split(r"[。！？]", text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 文を追加した場合の長さをチェック
        potential_chunk = current_chunk + sentence + "。"

        if len(potential_chunk) <= chunk_size:
            current_chunk = potential_chunk
        else:
            # 現在のチャンクを保存（最小長さを満たす場合）
            if len(current_chunk) >= MIN_OUTPUT_LENGTH:
                chunks.append(current_chunk)
            current_chunk = sentence + "。"

    # 最後のチャンクを追加
    if len(current_chunk) >= MIN_OUTPUT_LENGTH:
        chunks.append(current_chunk)

    return chunks


def create_summary(text: str, max_length: int = 300) -> str:
    """記事の要約を作成（先頭部分を使用）"""
    # 最初の段落または指定長さまでを要約として使用
    paragraphs = text.split("\n\n")
    summary = paragraphs[0] if paragraphs else text

    if len(summary) > max_length:
        # 文の境界で切り詰め
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
    """記事から適切な長さの質問-回答ペアを複数生成"""
    text = article["text"]
    title = article["meta"]["title"]

    # テキストをクリーンアップ
    cleaned_text = clean_text(text)

    # 記事が短すぎる場合はスキップ
    if len(cleaned_text) < MIN_OUTPUT_LENGTH:
        return []

    qa_pairs = []

    # 1. 要約版の質問-回答ペア
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

    # 2. チャンク分割による詳細な質問-回答ペア
    chunks = split_text_into_chunks(cleaned_text)

    for i, chunk in enumerate(chunks):
        # 長さ制限チェック
        if len(chunk) > MAX_OUTPUT_LENGTH:
            chunk = chunk[:MAX_OUTPUT_LENGTH] + "..."

        if len(chunk) < MIN_OUTPUT_LENGTH:
            continue

        # チャンクベースの質問パターン
        if i == 0:
            # 最初のチャンクは概要として扱う
            qa_pairs.append(
                {"input": f"{title}について教えてください。", "output": chunk}
            )
        else:
            # 他のチャンクは詳細情報として扱う
            qa_pairs.append(
                {
                    "input": f"{title}についてさらに詳しく教えてください。",
                    "output": chunk,
                }
            )

        # 特定のキーワードに基づく質問
        if "歴史" in chunk:
            qa_pairs.append(
                {"input": f"{title}の歴史について教えてください。", "output": chunk}
            )

        if any(keyword in chunk for keyword in ["特徴", "性質", "機能"]):
            qa_pairs.append(
                {"input": f"{title}の特徴について説明してください。", "output": chunk}
            )

    # 3. 全記事が短い場合は全体を使用
    if len(cleaned_text) <= MAX_OUTPUT_LENGTH and not qa_pairs:
        qa_pairs.append(
            {"input": f"{title}について教えてください。", "output": cleaned_text}
        )

    # 質問の長さチェックと調整
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
    """JSONLファイルを処理してQAペアを生成"""
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

                # 短すぎる記事はスキップ（最低限の品質確保）
                if len(article["text"]) < 100:
                    continue

                # 質問-回答ペアを生成
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
    """QAペアをJSONL形式で保存"""
    print(f"Saving {len(qa_pairs)} QA pairs to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for qa_pair in qa_pairs:
            json.dump(qa_pair, f, ensure_ascii=False)
            f.write("\n")

    print(f"Saved to {output_file}")


def main():
    # 設定
    input_dir = "/workspace/data/ja_wiki"
    output_dir = "/workspace/data/training_data"

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力ファイルリストを取得
    input_files = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith(".jsonl"):
            input_files.append(os.path.join(input_dir, file))

    print(f"Found {len(input_files)} JSONL files in {input_dir}")

    # 各ファイルを処理
    all_qa_pairs = []

    # 訓練用ファイル（train_*.jsonl）を処理 - すべてのデータを使用
    train_files = [f for f in input_files if "train_" in os.path.basename(f)]
    for train_file in train_files:
        # すべての記事を処理
        qa_pairs = process_jsonl_file(train_file, max_articles=None)
        all_qa_pairs.extend(qa_pairs)

    # バリデーション用ファイルを処理 - すべてのデータを使用
    validation_files = [f for f in input_files if "validation_" in os.path.basename(f)]
    validation_qa_pairs = []
    for val_file in validation_files:
        qa_pairs = process_jsonl_file(val_file, max_articles=None)
        validation_qa_pairs.extend(qa_pairs)

    # シャッフル
    random.seed(42)
    random.shuffle(all_qa_pairs)
    random.shuffle(validation_qa_pairs)

    # データ数制限なし - すべてのデータを使用
    print(
        f"Using all data: {len(all_qa_pairs)} training samples, {len(validation_qa_pairs)} validation samples"
    )

    # 保存
    save_qa_pairs(all_qa_pairs, os.path.join(output_dir, "training.jsonl"))
    save_qa_pairs(validation_qa_pairs, os.path.join(output_dir, "validation.jsonl"))

    print(f"\n=== 変換完了 ===")
    print(f"Training samples: {len(all_qa_pairs)}")
    print(f"Validation samples: {len(validation_qa_pairs)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
