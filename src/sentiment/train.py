from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import ROOT_DIR

DATA_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_sentiment_training.csv"
MODEL_DIR = ROOT_DIR / "models" / "trained" / "news_sentiment_model"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "news_sentiment_model_metadata.json"
REPORT_PATH = ROOT_DIR / "reports" / "model" / "news_sentiment_report.json"
BASE_MODEL = "ProsusAI/finbert"
LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index])
        return item


def train_model() -> dict:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data = pd.read_csv(DATA_PATH).sort_values("date").reset_index(drop=True)
    if data.empty:
        raise ValueError("Sentiment training dataset is empty.")
    if set(data["label_source"].dropna()) == {"WEAK_LABEL"}:
        print("WARNING: training uses weak/pseudo-labels only; no ground-truth evaluation is available.")
    train_end = int(len(data) * 0.70)
    validation_end = train_end + int(len(data) * 0.15)
    train, validation, test = data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=3, ignore_mismatched_sizes=True)
    train_ds = TextDataset(train.text, train.label.map(LABEL_TO_ID), tokenizer)
    validation_ds = TextDataset(validation.text, validation.label.map(LABEL_TO_ID), tokenizer)
    loader = DataLoader(train_ds, batch_size=8, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    for _ in range(1):
        for batch in loader:
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)
    metadata = {"model_name": "news_sentiment_model", "base_model": BASE_MODEL, "training_data_path": str(DATA_PATH), "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test), "class_distribution": data.label.value_counts().to_dict(), "label_source": data.label_source.unique().tolist(), "training_timestamp_utc": pd.Timestamp.now("UTC").isoformat(), "max_sequence_length": 128, "epochs": 1, "learning_rate": 2e-5, "ground_truth_evaluation_available": False, "limitations": ["Labels are weak/pseudo-labels from FinBERT; manually validated ground truth is not available."]}
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Train/validation/test: {len(train)}/{len(validation)}/{len(test)}")
    print(f"Model saved: {MODEL_DIR}")
    return metadata


if __name__ == "__main__":
    train_model()
