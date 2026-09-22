from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MarketLatest(BaseModel):
    ticker: str = Field(..., description='Ticker symbol')
    date: str = Field(..., description='Latest available date')
    close: float = Field(..., description='Latest close price')
    volume: int = Field(..., description='Latest volume')


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    signal: str
    timestamp: str
    model_version: str
    explanation: dict[str, Any]
    risk: dict[str, Any]
    backtest_metrics: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    message: str
