#!/usr/bin/env python3
"""
日本語WikipediaデータをNeMo 2.0 SFT形式に変換するスクリプト
入力形式: {"text": "記事内容", "meta": {"id": "ID", "title": "タイトル", "url": "URL"}}
出力形式: {"input": "質問", "output": "回答"}
"""

import json
import os
import re
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def extract_intro_from_text(text: str, title: str) -> str:
    """記事から導入部分を抽出"""
    # 最初の段落（通常は導入部分）を取得
    paragraphs = text.strip().split('\n\n')

    # 最初の非空の段落を見つける
    intro = ""
    for para in paragraphs:
        para = para.strip()
        if para and not para.startswith('=') and not para.startswith('#'):
            intro = para
            break

    # 長すぎる場合は最初の文のみ取得
    if len(intro) > 500:
        sentences = re.split(r'[。．]', intro)
        if sentences and len(sentences[0]) > 50:
            intro = sentences[0] + '。'

    return intro

def create_qa_pairs_from_article(article: Dict) -> List[Dict]:
    """記事から質問-回答ペアを複数生成"""
    text = article["text"]
    title = article["meta"]["title"]

    qa_pairs = []

    # 導入部分を取得
    intro = extract_intro_from_text(text, title)
    if not intro:
        return qa_pairs

    # パターン1: 「〜について教えて」
    qa_pairs.append({
        "input": f"{title}について教えてください。",
        "output": intro
    })

    # パターン2: 「〜とは何ですか？」
    qa_pairs.append({
        "input": f"{title}とは何ですか？",
        "output": intro
    })

    # パターン3: 「〜について説明してください」
    qa_pairs.append({
        "input": f"{title}について説明してください。",
        "output": intro
    })

    # 記事に「歴史」や「概要」などのセクションがある場合の追加質問
    sections = re.findall(r'\n([^=\n]+)\s*\n', text)
    if any('歴史' in s for s in sections):
        # 歴史に関する質問
        history_match = re.search(r'歴史[^\n]*\n(.*?)(?:\n\n|\n[^=\n]+\s*\n|\Z)', text, re.DOTALL)
        if history_match:
            history_text = history_match.group(1).strip()
            if len(history_text) > 50 and len(history_text) < 300:
                qa_pairs.append({
                    "input": f"{title}の歴史について教えてください。",
                    "output": history_text
                })

    return qa_pairs

def process_jsonl_file(input_file: str, max_articles: Optional[int] = None) -> List[Dict]:
    """JSONLファイルを処理してQAペアを生成"""
    qa_pairs = []
    processed_count = 0

    print(f"Processing {input_file}...")

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            if max_articles and processed_count >= max_articles:
                break

            try:
                article = json.loads(line.strip())
                if not article.get("text") or not article.get("meta", {}).get("title"):
                    continue

                # 質問-回答ペアを生成
                pairs = create_qa_pairs_from_article(article)
                qa_pairs.extend(pairs)
                processed_count += 1

                if processed_count % 1000 == 0:
                    print(f"  Processed {processed_count} articles, generated {len(qa_pairs)} QA pairs")

            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error on line {line_no}: {e}")
                continue
            except Exception as e:
                print(f"  Warning: Error processing line {line_no}: {e}")
                continue

    print(f"Completed processing {input_file}: {processed_count} articles -> {len(qa_pairs)} QA pairs")
    return qa_pairs

def save_qa_pairs(qa_pairs: List[Dict], output_file: str):
    """QAペアをJSONLファイルとして保存"""
    print(f"Saving {len(qa_pairs)} QA pairs to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        for qa in qa_pairs:
            json.dump(qa, f, ensure_ascii=False)
            f.write('\n')

    print(f"Saved to {output_file}")

def main():
    # 設定
    input_dir = "/workspace/data/ja_wiki"
    output_dir = "/workspace/data/sft"

    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)

    # 入力ファイルリストを取得
    input_files = []
    for file in sorted(os.listdir(input_dir)):
        if file.endswith('.jsonl'):
            input_files.append(os.path.join(input_dir, file))

    print(f"Found {len(input_files)} JSONL files in {input_dir}")

    # 各ファイルを処理
    all_qa_pairs = []

        # 訓練用ファイル（train_*.jsonl）を処理 - すべてのデータを使用
    train_files = [f for f in input_files if 'train_' in os.path.basename(f)]
    for train_file in train_files:
        # すべての記事を処理
        qa_pairs = process_jsonl_file(train_file, max_articles=None)
        all_qa_pairs.extend(qa_pairs)

    # バリデーション用ファイルを処理 - すべてのデータを使用
    validation_files = [f for f in input_files if 'validation_' in os.path.basename(f)]
    validation_qa_pairs = []
    for val_file in validation_files:
        qa_pairs = process_jsonl_file(val_file, max_articles=None)
        validation_qa_pairs.extend(qa_pairs)

    # シャッフル
    random.seed(42)
    random.shuffle(all_qa_pairs)
    random.shuffle(validation_qa_pairs)

        # データ数制限なし - すべてのデータを使用
    print(f"Using all data: {len(all_qa_pairs)} training samples, {len(validation_qa_pairs)} validation samples")

    # 保存
    save_qa_pairs(all_qa_pairs, os.path.join(output_dir, "training.jsonl"))
    save_qa_pairs(validation_qa_pairs, os.path.join(output_dir, "validation.jsonl"))

    print(f"\n=== 変換完了 ===")
    print(f"Training samples: {len(all_qa_pairs)}")
    print(f"Validation samples: {len(validation_qa_pairs)}")
    print(f"Output directory: {output_dir}")

if __name__ == "__main__":
    main()
