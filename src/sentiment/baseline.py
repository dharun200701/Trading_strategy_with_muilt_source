from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from transformers import pipeline

from src.config import ROOT_DIR

CLEAN_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_clean.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_baseline_sentiment.csv"
MODEL_NAME = "ProsusAI/finbert"
LABEL_MAP = {"positive": "POSITIVE", "neutral": "NEUTRAL", "negative": "NEGATIVE"}


def run_baseline(input_path: Path = CLEAN_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME, return_all_scores=True, truncation=True, max_length=256)
    rows = []
    for text in data["clean_text"].fillna("").tolist():
        scores = classifier(text)
        if scores and isinstance(scores[0], list):
            scores = scores[0]
        probabilities = {LABEL_MAP.get(str(item["label"]).lower(), str(item["label"]).upper()): float(item["score"]) for item in scores}
        probabilities = {label: probabilities.get(label, 0.0) for label in ["POSITIVE", "NEUTRAL", "NEGATIVE"]}
        sentiment = max(probabilities, key=probabilities.get)
        confidence = probabilities[sentiment]
        rows.append({"sentiment": sentiment, "positive_probability": probabilities["POSITIVE"], "neutral_probability": probabilities["NEUTRAL"], "negative_probability": probabilities["NEGATIVE"], "confidence": confidence, "model_name": MODEL_NAME, "label_source": "WEAK_LABEL"})
    result = pd.concat([data[["article_id", "date", "headline"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Baseline model: {MODEL_NAME}")
    print(f"Articles scored: {len(result)}")
    print(f"Output: {output_path}")
    return result


if __name__ == "__main__":
    run_baseline()
