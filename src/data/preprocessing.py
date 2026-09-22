from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)


def load_raw_market_data(path: Path | str | None = None) -> pd.DataFrame:
    if path is None:
        path = ROOT_DIR / "data" / "raw" / "market" / "HDFCBANK_NSE_raw.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def preprocess_market_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    processed = raw_df.copy()
    processed = processed.sort_values("date").reset_index(drop=True)
    processed = processed.drop_duplicates(subset=["date"]).copy()
    for col in ["open", "high", "low", "close", "volume"]:
        processed[col] = pd.to_numeric(processed[col], errors="coerce")
    processed = processed.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    processed["daily_return"] = processed["close"].pct_change().fillna(0.0)
    processed["previous_close"] = processed["close"].shift(1)
    processed["high_low_range"] = processed["high"] - processed["low"]
    processed["open_close_change"] = processed["close"] - processed["open"]
    processed["volume_change"] = processed["volume"].pct_change().fillna(0.0)
    output_path = ROOT_DIR / "data" / "processed" / "market" / "HDFCBANK_daily.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)
    legacy_output_path = ROOT_DIR / "data" / "processed" / "hdfcbank_preprocessed.csv"
    processed.to_csv(legacy_output_path, index=False)
    logger.info("Saved preprocessed data to %s", output_path)
    return processed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raw_df = load_raw_market_data()
    preprocess_market_data(raw_df)
