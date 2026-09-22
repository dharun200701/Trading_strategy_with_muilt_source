from __future__ import annotations

from typing import Any


def generate_signal(prediction: str, confidence: float, technical_metrics: dict[str, Any] | None = None, min_confidence: float = 0.55, volatility_limit: float = 0.05) -> str:
    technical_metrics = technical_metrics or {}
    volatility = float(technical_metrics.get('volatility', 0.0))
    atr = float(technical_metrics.get('atr', 0.0))

    if prediction not in {'UP', 'DOWN', 'NEUTRAL'}:
        return 'HOLD'
    if confidence < min_confidence:
        return 'HOLD'
    if volatility > volatility_limit:
        return 'HOLD'
    if atr <= 0:
        return 'HOLD'

    if prediction == 'UP':
        return 'BUY'
    if prediction == 'DOWN':
        return 'SELL'
    return 'HOLD'
