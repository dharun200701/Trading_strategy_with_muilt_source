from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import ROOT_DIR

INPUT_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_baseline_sentiment.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_daily_sentiment.csv"


def aggregate_daily(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    data = pd.read_csv(input_path)
    grouped = data.groupby("date", as_index=False)
    daily = grouped.agg(article_count=("article_id", "count"), positive_count=("sentiment", lambda values: int((values == "POSITIVE").sum())), neutral_count=("sentiment", lambda values: int((values == "NEUTRAL").sum())), negative_count=("sentiment", lambda values: int((values == "NEGATIVE").sum())), average_positive_probability=("positive_probability", "mean"), average_neutral_probability=("neutral_probability", "mean"), average_negative_probability=("negative_probability", "mean"))
    daily["positive_ratio"] = daily["positive_count"] / daily["article_count"]
    daily["neutral_ratio"] = daily["neutral_count"] / daily["article_count"]
    daily["negative_ratio"] = daily["negative_count"] / daily["article_count"]
    daily["daily_sentiment_score"] = daily["average_positive_probability"] - daily["average_negative_probability"]
    daily["daily_sentiment_confidence"] = daily[["average_positive_probability", "average_neutral_probability", "average_negative_probability"]].max(axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_path, index=False)
    print(f"Daily rows: {len(daily)}")
    print(f"Output: {output_path}")
    return daily


if __name__ == "__main__":
    aggregate_daily()
