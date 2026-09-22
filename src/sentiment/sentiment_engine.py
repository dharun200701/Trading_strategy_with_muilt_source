from __future__ import annotations

from typing import Any


class SentimentEngine:
    def __init__(self) -> None:
        self.enabled = False

    def get_sentiment_features(self, text: str | None = None) -> dict[str, Any]:
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'source': 'disabled',
        }
