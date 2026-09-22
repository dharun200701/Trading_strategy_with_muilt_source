from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import ROOT_DIR

MARKET_PATH = ROOT_DIR / "data" / "raw" / "market" / "HDFCBANK_NSE_raw.csv"


def load_market(path: Path = MARKET_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    rename = {str(column).strip().lower(): column for column in data.columns}
    required_aliases = {"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}
    missing = [key for key in required_aliases if key not in rename]
    if missing:
        raise ValueError(f"Market data missing required columns: {missing}")
    data = data.rename(columns={rename[key]: key for key in required_aliases})
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    data = data[(data["open"] > 0) & (data["high"] >= data["low"]) & (data["close"] >= data["low"]) & (data["close"] <= data["high"]) & (data["volume"] >= 0)]
    data = data.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    data["return"] = data["close"].pct_change()
    data["rolling_volatility_20"] = data["return"].rolling(20, min_periods=10).std()
    data["average_volume_20"] = data["volume"].rolling(20, min_periods=10).mean()
    return data


def market_window(market: pd.DataFrame, news_date: pd.Timestamp) -> dict[str, Any]:
    dates = market["date"]
    candidates = market.index[dates >= news_date.normalize()]
    if len(candidates) == 0:
        return {"market_data_available": False}
    news_index = int(candidates[0])
    if news_index == 0:
        return {"market_data_available": False}
    previous_index = news_index - 1
    next_index = news_index + 1 if news_index + 1 < len(market) else None
    row = market.iloc[news_index]
    previous = market.iloc[previous_index]
    next_close = float(market.iloc[next_index]["close"]) if next_index is not None else None
    previous_previous_close = float(market.iloc[previous_index - 1]["close"]) if previous_index > 0 else None
    previous_day_return = (float(previous["close"]) / previous_previous_close - 1) if previous_previous_close else None
    news_day_return = float(row["close"]) / float(previous["close"]) - 1
    next_day_return = (next_close / float(row["close"]) - 1) if next_close is not None else None
    pre_avg_volume = float(previous["average_volume_20"]) if pd.notna(previous["average_volume_20"]) else None
    volume_ratio = float(row["volume"]) / pre_avg_volume if pre_avg_volume and pre_avg_volume > 0 else None
    volatility_before = float(previous["rolling_volatility_20"]) if pd.notna(previous["rolling_volatility_20"]) else None
    recent_returns = market.iloc[max(0, news_index - 19):news_index + 1]["return"].dropna()
    volatility_after = float(recent_returns.std()) if len(recent_returns) >= 10 else None
    return {
        "market_data_available": True, "reaction_method": "daily_date_based", "previous_trading_day": previous["date"].date().isoformat(), "reaction_trading_date": row["date"].date().isoformat(),
        "previous_close": float(previous["close"]), "news_day_open": float(row["open"]), "news_day_high": float(row["high"]), "news_day_low": float(row["low"]), "news_day_close": float(row["close"]), "next_day_close": next_close,
        "previous_day_return": previous_day_return, "news_day_return": news_day_return, "post_news_return": news_day_return, "next_day_return": next_day_return,
        "volume_before": pre_avg_volume, "volume_after": float(row["volume"]), "volume_ratio": volume_ratio, "volume_change_pct": (volume_ratio - 1) * 100 if volume_ratio is not None else None,
        "volatility_before": volatility_before, "volatility_after": volatility_after, "volatility_change": (volatility_after - volatility_before) if volatility_after is not None and volatility_before is not None else None,
    }
