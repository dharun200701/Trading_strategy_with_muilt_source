from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

RAW_PATH = ROOT_DIR / "data" / "raw" / "news" / "HDFCBANK_news_raw.csv"
CLEAN_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_clean.csv"


def clean_text(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def preprocess_news(input_path: Path = RAW_PATH, output_path: Path = CLEAN_PATH) -> tuple[pd.DataFrame, dict]:
    raw = pd.read_csv(input_path)
    before = len(raw)
    raw["published_at"] = pd.to_datetime(raw["published_at"], errors="coerce", utc=True)
    raw["date"] = raw["published_at"].dt.date.astype("string")
    raw["original_headline"] = raw["headline"].fillna("").astype(str)
    raw["cleaned_headline"] = raw["headline"].map(clean_text)
    raw["cleaned_description"] = raw["description"].map(clean_text)
    raw["cleaned_content"] = raw["content"].map(clean_text)
    raw["clean_text"] = (raw["cleaned_headline"] + " " + raw["cleaned_description"] + " " + raw["cleaned_content"]).map(clean_text)
    no_text = raw["clean_text"].eq("")
    raw = raw.loc[~no_text].copy()
    after_text = len(raw)
    exact_duplicates = int(raw.duplicated().sum())
    raw = raw.drop_duplicates().copy()
    duplicate_urls = int(raw["url"].dropna().duplicated().sum())
    raw = raw.drop_duplicates(subset=["url"], keep="first")
    search = (raw["cleaned_headline"] + " " + raw["cleaned_description"]).str.lower()
    related = search.str.contains(r"hdfc\s*bank|hdfcbank", regex=True, na=False) | raw["query"].fillna("").str.contains(r"hdfc\s*bank|hdfcbank", case=False, regex=True)
    unrelated = int((~related).sum())
    raw = raw.loc[related].copy()
    raw = raw.sort_values("published_at").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_path, index=False)
    report = {"rows_before": before, "removed_no_text": before - after_text, "removed_exact_duplicates": exact_duplicates, "removed_duplicate_urls": duplicate_urls, "removed_unrelated": unrelated, "rows_after": len(raw), "output": str(output_path)}
    print(json.dumps(report, indent=2))
    return raw, report


if __name__ == "__main__":
    preprocess_news()
