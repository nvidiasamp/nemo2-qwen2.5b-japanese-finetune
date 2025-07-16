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

def clean_text(text: str) -> str:
    """テキストをクリーンアップ"""
    # 余分な改行を整理
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 余分な空白を削除
    text = re.sub(r' +', ' ', text)
    return text.strip()

def create_qa_pairs_from_article(article: Dict) -> List[Dict]:
    """記事から質問-回答ペアを複数生成"""
    text = article["text"]
    title = article["meta"]["title"]

    # テキストをクリーンアップ
    cleaned_text = clean_text(text)

    qa_pairs = []

    # パターン1: 「〜について教えて」- 記事全体を回答として使用
    qa_pairs.append({
        "input": f"{title}について教えてください。",
        "output": cleaned_text
    })

    # パターン2: 「〜とは何ですか？」- 記事全体を回答として使用
    qa_pairs.append({
        "input": f"{title}とは何ですか？",
        "output": cleaned_text
    })

    # パターン3: 「〜について詳しく説明してください」
    qa_pairs.append({
        "input": f"{title}について詳しく説明してください。",
        "output": cleaned_text
    })

    # パターン4: 「〜に関する情報を教えてください」
    qa_pairs.append({
        "input": f"{title}に関する情報を教えてください。",
        "output": cleaned_text
    })

    # セクション別の質問も生成（歴史、概要など）
    if '歴史' in text and len(text) > 1000:
        qa_pairs.append({
            "input": f"{title}の歴史について教えてください。",
            "output": cleaned_text
        })

    if any(keyword in text for keyword in ['概要', '定義', '特徴']) and len(text) > 1000:
        qa_pairs.append({
            "input": f"{title}の概要について説明してください。",
            "output": cleaned_text
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

                # 短すぎる記事はスキップ（最低限の品質確保）
                if len(article["text"]) < 100:
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

// ... existing code ...

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
