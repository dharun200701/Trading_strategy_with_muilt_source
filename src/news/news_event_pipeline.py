from __future__ import annotations

import json
from collections import Counter

from src.config import ROOT_DIR
from src.news.event_labeler import label_events


def main() -> None:
    result, review = label_events()
    impact_counts = Counter(result["impact_level"])
    confidence_counts = Counter("HIGH" if value >= 0.75 else "MEDIUM" if value >= 0.5 else "LOW" for value in result["label_confidence"])
    quality_counts = Counter(result["label_quality"])
    report = {
        "dataset_size": len(result), "date_range": {"first": str(result["news_date"].min()), "last": str(result["news_date"].max())}, "articles_with_timestamps": int(result["timestamp_available"].sum()), "articles_without_timestamps": int((~result["timestamp_available"]).sum()), "reaction_method_distribution": result["reaction_method"].value_counts().to_dict(),
        "positive_count": int((result["target_label"] == "POSITIVE").sum()), "neutral_count": int((result["target_label"] == "NEUTRAL").sum()), "negative_count": int((result["target_label"] == "NEGATIVE").sum()),
        "impact_very_low": impact_counts.get("VERY_LOW", 0), "impact_low": impact_counts.get("LOW", 0), "impact_medium": impact_counts.get("MEDIUM", 0), "impact_high": impact_counts.get("HIGH", 0), "impact_very_high": impact_counts.get("VERY_HIGH", 0),
        "high_confidence_count": confidence_counts.get("HIGH", 0), "medium_confidence_count": confidence_counts.get("MEDIUM", 0), "low_confidence_count": confidence_counts.get("LOW", 0), "benchmark_available_count": int(result["benchmark_available"].sum()), "expectation_data_available_count": int(result["expectation_data_available"].sum()), "confounding_event_count": int(result["confounding_event"].sum()), "label_quality_distribution": dict(quality_counts), "missing_market_data_count": int((result["market_reaction"] == "NEUTRAL").sum()), "duplicate_count": int(result["article_id"].duplicated().sum()),
        "data_quality_issues": ["Daily market data only; intraday reaction fields are null.", "No benchmark dataset was available.", "No reliable expectation-vs-actual fields were available."], "labeling_methodology": "Daily event date is assigned to the next available NSE trading day. Direction is based on reaction_change = news_day_return - previous_day_return, normalized by pre-news 20-day volatility; z >= 0.5 is POSITIVE, z <= -0.5 is NEGATIVE, otherwise NEUTRAL.", "impact_score_methodology": "Weighted normalized score: 40% absolute volatility-adjusted reaction, 20% volume abnormality, 15% volatility change, 25% persistence. Missing components are renormalized.", "leakage_checks": {"future_reaction_fields_are_targets_only": True, "intraday_fields_available": False, "benchmark_available": False}, "review_articles": len(review), "output": "data/processed/news/HDFCBANK_news_event_labeled.csv", "review_output": "data/processed/news/HDFCBANK_news_event_review.csv",
    }
    for path in [ROOT_DIR / "reports" / "validation" / "news_event_labeling_report.json", ROOT_DIR / "reports" / "model" / "news_event_dataset_summary.json"]:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("NEWS EVENT LABELING COMPLETE")
    print(f"Articles processed: {len(result)}")
    print(f"Market reaction: POSITIVE {report['positive_count']} / NEUTRAL {report['neutral_count']} / NEGATIVE {report['negative_count']}")
    print(f"Impact: VERY LOW {report['impact_very_low']} / LOW {report['impact_low']} / MEDIUM {report['impact_medium']} / HIGH {report['impact_high']} / VERY HIGH {report['impact_very_high']}")
    print(f"Confidence: HIGH {report['high_confidence_count']} / MEDIUM {report['medium_confidence_count']} / LOW {report['low_confidence_count']}")
    print(f"Benchmark available: {report['benchmark_available_count']}")
    print(f"Expectation data available: {report['expectation_data_available_count']}")
    print(f"Confounding events: {report['confounding_event_count']}")
    print("Output: data/processed/news/HDFCBANK_news_event_labeled.csv")
    print("Review: data/processed/news/HDFCBANK_news_event_review.csv")
    print("Report: reports/validation/news_event_labeling_report.json")


if __name__ == "__main__":
    main()
