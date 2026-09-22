from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from src.config import ROOT_DIR

EVENT_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_event_labeled.csv"
TRAIN_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_event_model_training.csv"
FUTURE_COLUMNS = {"post_news_return", "next_day_return", "abnormal_return", "impact_score", "volume_after", "volatility_after", "news_day_close", "next_day_close", "market_reaction"}


def prepare_model2_dataset() -> pd.DataFrame:
    data = pd.read_csv(EVENT_PATH).sort_values(["news_date", "article_id"]).reset_index(drop=True)
    input_columns = [column for column in ["article_id", "news_date", "headline", "clean_text", "source_name", "text_sentiment", "text_positive_probability", "text_neutral_probability", "text_negative_probability", "timestamp_available", "expected_value", "actual_value"] if column in data.columns]
    if FUTURE_COLUMNS.intersection(input_columns):
        raise RuntimeError(f"Future-derived columns leaked into Model 2 inputs: {FUTURE_COLUMNS.intersection(input_columns)}")
    result = data[input_columns + ["target_label", "label_confidence", "label_quality"]].copy()
    result.to_csv(TRAIN_PATH, index=False)
    report = {"rows": len(result), "input_columns": input_columns, "future_columns_excluded": sorted(FUTURE_COLUMNS), "chronological": bool(pd.to_datetime(result["news_date"]).is_monotonic_increasing), "output": str(TRAIN_PATH)}
    (ROOT_DIR / "reports" / "validation").mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / "reports" / "validation" / "news_event_model2_leakage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return result


if __name__ == "__main__":
    prepare_model2_dataset()
