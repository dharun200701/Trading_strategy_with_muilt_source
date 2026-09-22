from __future__ import annotations

import math
from typing import Any

import numpy as np


def clip01(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return max(0.0, min(1.0, float(value)))


def score_event(window: dict[str, Any]) -> dict[str, Any]:
    post = window.get("post_news_return")
    pre = window.get("previous_day_return")
    next_return = window.get("next_day_return")
    volatility = window.get("volatility_before")
    reaction_change = post - pre if post is not None and pre is not None else None
    z_score = reaction_change / volatility if reaction_change is not None and volatility and volatility > 0 else None
    if z_score is None:
        market_label = "NEUTRAL"
    elif z_score >= 0.5:
        market_label = "POSITIVE"
    elif z_score <= -0.5:
        market_label = "NEGATIVE"
    else:
        market_label = "NEUTRAL"
    price_component = clip01(abs(z_score) / 3.0)
    volume_ratio = window.get("volume_ratio")
    volume_component = clip01((volume_ratio - 1) / 2) if volume_ratio is not None else None
    volatility_change = window.get("volatility_change")
    volatility_component = clip01(abs(volatility_change) / (volatility or 1e-9)) if volatility_change is not None and volatility else None
    persistence_component = None
    if post is not None and next_return is not None:
        same_direction = np.sign(post) == np.sign(next_return) and np.sign(post) != 0
        persistence_component = 1.0 if same_direction else 0.25
    components = [(price_component, 0.40), (volume_component, 0.20), (volatility_component, 0.15), (persistence_component, 0.25)]
    valid = [(value, weight) for value, weight in components if value is not None]
    impact_score = round(100 * sum(value * weight for value, weight in valid) / sum(weight for _, weight in valid), 2) if valid else 0.0
    if impact_score < 20: impact_level = "VERY_LOW"
    elif impact_score < 40: impact_level = "LOW"
    elif impact_score < 60: impact_level = "MEDIUM"
    elif impact_score < 80: impact_level = "HIGH"
    else: impact_level = "VERY_HIGH"
    if impact_score >= 80: priority = "CRITICAL"
    elif impact_score >= 60: priority = "HIGH"
    elif impact_score >= 40: priority = "MEDIUM"
    else: priority = "LOW"
    available_count = sum(value is not None for value in [post, pre, next_return, volume_ratio, volatility])
    confidence = min(1.0, 0.25 + available_count * 0.12 + (0.15 if abs(z_score or 0) >= 1.5 else 0))
    quality = "HIGH" if confidence >= 0.75 and abs(z_score or 0) >= 1.0 else "MEDIUM" if confidence >= 0.5 else "LOW"
    reason = "Small price movement within normal volatility; no meaningful abnormal reaction."
    if market_label == "POSITIVE": reason = "Positive HDFC Bank reaction relative to the previous trading-day movement."
    elif market_label == "NEGATIVE": reason = "Negative HDFC Bank reaction relative to the previous trading-day movement."
    if volume_ratio is not None and volume_ratio >= 2: reason += " Elevated trading volume increased event importance."
    if next_return is None: quality = "LOW"
    return {"pre_news_return": pre, "reaction_change": reaction_change, "abnormal_return": reaction_change, "market_reaction": market_label, "impact_score": impact_score, "impact_level": impact_level, "reaction_priority": priority, "label_confidence": round(confidence, 3), "label_quality": quality, "label_reason": reason, "persistence_component": persistence_component, "z_score": z_score}
