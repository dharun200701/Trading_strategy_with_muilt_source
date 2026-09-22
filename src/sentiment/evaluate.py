from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.config import ROOT_DIR

TRAINING_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_sentiment_training.csv"
MANUAL_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_manual_validation.csv"
REPORT_PATH = ROOT_DIR / "reports" / "model" / "news_sentiment_report.json"
METADATA_PATH = ROOT_DIR / "models" / "metadata" / "news_sentiment_model_metadata.json"
LABELS = ["POSITIVE", "NEUTRAL", "NEGATIVE"]


def leakage_check(data: pd.DataFrame) -> dict:
    train_end = int(len(data) * 0.70)
    validation_end = train_end + int(len(data) * 0.15)
    partitions = [data.iloc[:train_end], data.iloc[train_end:validation_end], data.iloc[validation_end:]]
    issues = []
    for column in ["article_id", "url", "headline", "text"]:
        if column in data.columns:
            for left_index in range(len(partitions)):
                for right_index in range(left_index + 1, len(partitions)):
                    overlap = set(partitions[left_index][column].dropna()) & set(partitions[right_index][column].dropna())
                    if overlap:
                        issues.append(f"{column} overlap between split {left_index} and {right_index}: {len(overlap)}")
    return {"passed": not issues, "issues": issues}


def evaluate_model() -> dict:
    data = pd.read_csv(TRAINING_PATH).sort_values("date").reset_index(drop=True)
    train_end = int(len(data) * 0.70)
    validation_end = train_end + int(len(data) * 0.15)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    result = {
        "dataset": {"total_articles": len(data), "first_date": str(data["date"].min()), "last_date": str(data["date"].max())},
        "train": {"rows": train_end, "first_date": str(data.iloc[0]["date"]), "last_date": str(data.iloc[train_end - 1]["date"])},
        "validation": {"rows": validation_end - train_end, "first_date": str(data.iloc[train_end]["date"]), "last_date": str(data.iloc[validation_end - 1]["date"])},
        "test": {"rows": len(data) - validation_end, "first_date": str(data.iloc[validation_end]["date"]), "last_date": str(data.iloc[-1]["date"])},
        "class_distribution": data["label"].value_counts().to_dict(),
        "model": {"base_model": metadata.get("base_model", "ProsusAI/finbert"), "fine_tuned": bool(metadata), "label_source": metadata.get("label_source", ["WEAK_LABEL"])},
        "ground_truth_evaluation_available": False,
        "message": "Ground-truth evaluation dataset not available.",
        "accuracy": None, "macro_precision": None, "macro_recall": None, "macro_f1": None, "weighted_f1": None,
        "majority_baseline_accuracy": None, "pretrained_model_accuracy": None, "confusion_matrix": None,
        "leakage_check": leakage_check(data),
    }
    if MANUAL_PATH.exists():
        manual = pd.read_csv(MANUAL_PATH)
        if "manual_label" in manual.columns and manual["manual_label"].notna().any():
            result["ground_truth_evaluation_available"] = True
            result["message"] = "Manual labels are available; model inference evaluation must be run against the manual set."
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    evaluate_model()
