from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import ROOT_DIR
from src.news.market_reaction import load_market, market_window
from src.news.impact_scorer import score_event

NEWS_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_clean.csv"
BASELINE_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_baseline_sentiment.csv"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_event_labeled.csv"
REVIEW_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_event_review.csv"


def label_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    news = pd.read_csv(NEWS_PATH)
    baseline = pd.read_csv(BASELINE_PATH) if BASELINE_PATH.exists() else pd.DataFrame()
    market = load_market()
    if not baseline.empty:
        news = news.merge(baseline[["article_id", "sentiment", "positive_probability", "neutral_probability", "negative_probability"]], on="article_id", how="left")
    else:
        for column in ["sentiment", "positive_probability", "neutral_probability", "negative_probability"]: news[column] = None
    news["news_datetime"] = pd.to_datetime(news["published_at"], errors="coerce", utc=True)
    news["news_date"] = pd.to_datetime(news["date"], errors="coerce").dt.date.astype("string")
    output_rows = []
    for _, article in news.iterrows():
        window = market_window(market, pd.Timestamp(article["news_date"])) if pd.notna(article["news_date"]) else {"market_data_available": False}
        score = score_event(window)
        row = {"article_id": article["article_id"], "news_datetime": article["news_datetime"].isoformat() if pd.notna(article["news_datetime"]) else None, "news_date": article["news_date"], "headline": article.get("headline"), "clean_text": article.get("clean_text"), "source_name": article.get("source_name"), "url": article.get("url"), "timestamp_available": bool(pd.notna(article["news_datetime"])), "reaction_method": window.get("reaction_method", "unavailable"), "market_data_available": window.get("market_data_available", False), "text_sentiment": article.get("sentiment"), "text_positive_probability": article.get("positive_probability"), "text_neutral_probability": article.get("neutral_probability"), "text_negative_probability": article.get("negative_probability"), **{key: window.get(key) for key in ["previous_trading_day", "previous_close", "news_day_open", "news_day_high", "news_day_low", "news_day_close", "next_day_close", "previous_day_return", "news_day_return", "post_news_return", "next_day_return", "volume_before", "volume_after", "volume_ratio", "volume_change_pct", "volatility_before", "volatility_after", "volatility_change"]}, "return_15m_after_news": None, "return_30m_after_news": None, "return_1h_after_news": None, "return_2h_after_news": None, "benchmark_return": None, "benchmark_available": False, "expected_value": None, "actual_value": None, "expectation_surprise": None, "expectation_surprise_pct": None, "expectation_data_available": False, **score, "target_label": score["market_reaction"], "confounding_event": False, "confounding_reason": "", "label_reason": score["label_reason"]}
        output_rows.append(row)
    result = pd.DataFrame(output_rows).sort_values(["news_date", "article_id"]).reset_index(drop=True)
    result["duplicate_news_date_count"] = result.groupby("news_date")["article_id"].transform("count")
    result.loc[result["duplicate_news_date_count"] > 1, "confounding_event"] = True
    result.loc[result["duplicate_news_date_count"] > 1, "confounding_reason"] = "Multiple HDFC Bank articles share the same date; daily attribution is ambiguous."
    result.loc[result["duplicate_news_date_count"] > 1, "label_quality"] = "LOW"
    result.loc[result["duplicate_news_date_count"] > 1, "label_confidence"] = (result.loc[result["duplicate_news_date_count"] > 1, "label_confidence"] * 0.6).round(3)
    result = result.drop(columns=["duplicate_news_date_count"])
    review = result[(result["label_quality"] == "LOW") | (result["impact_level"].isin(["HIGH", "VERY_HIGH"]))].copy()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    review.to_csv(REVIEW_PATH, index=False)
    return result, review
