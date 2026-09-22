from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    ticker: str = os.getenv("TICKER", "HDFCBANK")
    market_data_source: str = os.getenv("DATA_SOURCE", "NSE India")
    market_data_provider: str = os.getenv("DATA_PROVIDER", "nsepython")
    model_name: str = os.getenv("MODEL_NAME", "hdfcbank_xgboost")
    target_threshold: float = float(os.getenv("TARGET_THRESHOLD", "0.01"))
    min_confidence: float = float(os.getenv("MIN_CONFIDENCE", "0.70"))
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    account_capital: float = float(os.getenv("ACCOUNT_CAPITAL", "100000.0"))
    transaction_cost: float = float(os.getenv("TRANSACTION_COST", "0.001"))
    data_lookback_days: int = int(os.getenv("DATA_LOOKBACK_DAYS", "1250"))
    seed: int = int(os.getenv("SEED", "42"))
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8001"))


settings = Settings()
