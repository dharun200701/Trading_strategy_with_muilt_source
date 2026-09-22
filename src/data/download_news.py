from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.news.news_provider import fetch_hdfc_news
from src.validation.news_data_validator import validate_news_data
from src.config import ROOT_DIR

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download real HDFC Bank news data.")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore the existing news cache.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    raw_path = ROOT_DIR / "data" / "raw" / "news" / "HDFCBANK_news_raw.csv"
    validation_path = ROOT_DIR / "reports" / "validation" / "news_data_validation.json"
    try:
        data, metadata = fetch_hdfc_news(force_refresh=args.force_refresh)
        validation = validate_news_data(data)
    except Exception as exc:
        print(f"News download failed: {exc}")
        return 1
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(json.dumps(validation, indent=2), encoding="utf-8")
    metadata_path = ROOT_DIR / "data" / "raw" / "news" / "metadata.json"
    print("NEWS DATA DOWNLOAD")
    print("------------------")
    print("Company: HDFC Bank Ltd")
    print("Symbol: HDFCBANK")
    print(f"Provider(s): {', '.join(metadata['providers'])}")
    print(f"Articles: {len(data)}")
    print(f"First article: {metadata['actual_start_date']}")
    print(f"Last article: {metadata['actual_end_date']}")
    print(f"Duplicates removed: {metadata['duplicate_count_removed']}")
    print(f"Validation: {validation['status']}")
    print(f"Raw file: {raw_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Validation report: {validation_path}")
    return 0 if validation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
