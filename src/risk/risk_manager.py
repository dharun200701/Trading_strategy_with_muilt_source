from __future__ import annotations

import math
from typing import Any


class RiskManager:
    def __init__(self, account_capital: float = 100000.0, risk_per_trade: float = 0.01, stop_loss_distance: float = 0.05):
        self.account_capital = float(account_capital)
        self.risk_per_trade = float(risk_per_trade)
        self.stop_loss_distance = float(stop_loss_distance)

    def compute_position_size(self, stop_loss_distance: float | None = None) -> float:
        stop_distance = self.stop_loss_distance if stop_loss_distance is None else float(stop_loss_distance)
        if stop_distance <= 0:
            return 0.0
        risk_amount = self.account_capital * self.risk_per_trade
        return risk_amount / stop_distance

    def assess_risk(self, volatility: float, atr: float) -> dict[str, Any]:
        risk_score = min(1.0, max(0.0, (volatility + atr) / 2.0))
        return {
            'volatility': float(volatility),
            'atr': float(atr),
            'risk_score': float(risk_score),
            'stop_loss_distance': self.stop_loss_distance,
            'position_size': self.compute_position_size(self.stop_loss_distance),
            'risk_per_trade': self.risk_per_trade,
            'account_capital': self.account_capital,
            'status': 'acceptable' if risk_score < 0.6 else 'elevated'
        }
