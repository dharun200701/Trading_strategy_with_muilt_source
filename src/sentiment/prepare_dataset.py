from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import ROOT_DIR

CLEAN_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_clean.csv"
BASELINE_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_baseline_sentiment.csv"
MANUAL_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_manual_validation.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_sentiment_training.csv"


def prepare_dataset() -> pd.DataFrame:
    clean = pd.read_csv(CLEAN_PATH)
    manual = pd.read_csv(MANUAL_PATH) if MANUAL_PATH.exists() else pd.DataFrame()
    if not MANUAL_PATH.exists():
        template = clean[["article_id", "headline", "clean_text"]].head(50).copy()
        template["manual_label"] = ""
        template["review_notes"] = ""
        MANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
        template.to_csv(MANUAL_PATH, index=False)
    if not manual.empty and "manual_label" in manual.columns:
        result = clean.merge(manual[["article_id", "manual_label"]], on="article_id", how="inner")
        result["label"] = result["manual_label"]
        result["label_source"] = "MANUAL_VALIDATION"
    else:
        baseline = pd.read_csv(BASELINE_PATH)
        result = clean.merge(baseline[["article_id", "sentiment"]], on="article_id", how="inner")
        result["label"] = result["sentiment"]
        result["label_source"] = "WEAK_LABEL"
    result = result.assign(text=result["clean_text"].fillna(result["cleaned_headline"]))
    result = result[["article_id", "date", "text", "label", "label_source"]].dropna(subset=["text", "label"])
    result = result.sort_values("date").drop_duplicates("article_id").reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"Examples: {len(result)}")
    print(result["label"].value_counts().to_string())
    print(f"Label source: {result['label_source'].unique().tolist()}")
    print(f"Ground-truth evaluation dataset available: {MANUAL_PATH.exists() and not manual.empty}")
    return result


if __name__ == "__main__":
    prepare_dataset()
