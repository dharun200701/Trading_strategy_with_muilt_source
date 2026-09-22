from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from src.config import ROOT_DIR, settings
from src.models.train import prepare_training_frame

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    initial_capital: float
    final_capital: float
    total_return: float
    cumulative_return: float
    max_drawdown: float
    sharpe_ratio: float
    win_rate: float
    number_of_trades: int
    average_trade_return: float
    trades: list[dict[str, Any]]


def calculate_drawdown(values: pd.Series) -> float:
    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    return float(drawdown.min()) if not drawdown.empty else 0.0


def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    if returns.empty:
        return 0.0
    excess = returns - risk_free_rate
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float((excess.mean() / std) * np.sqrt(252))


def run_backtest() -> dict[str, Any]:
    model_path = ROOT_DIR / 'models' / 'trained' / f'{settings.model_name}.pkl'
    metadata_path = ROOT_DIR / 'models' / 'metadata' / f'{settings.model_name}_metadata.json'
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError('Model 1 artifact and metadata are required for a backtest.')
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    raw = pd.read_csv(ROOT_DIR / 'data' / 'raw' / 'market' / 'HDFCBANK_NSE_raw.csv')
    frame = prepare_training_frame(raw).dropna().reset_index(drop=True)
    feature_columns = metadata.get('feature_columns') or metadata.get('feature_names', [])
    test_start = int(len(frame) * 0.85)
    test = frame.iloc[test_start:].copy()
    predictions = model.predict(test[feature_columns]).astype(int)
    probabilities = model.predict_proba(test[feature_columns])
    initial_capital = 100000.0
    capital = initial_capital
    trades: list[dict[str, Any]] = []
    equity_curve = [initial_capital]
    returns: list[float] = []
    trade_returns: list[float] = []
    labels = metadata.get('target_classes') or ['UP', 'DOWN', 'NEUTRAL']
    for offset, (idx, row) in enumerate(test.iloc[:-1].iterrows()):
        signal_label = labels[int(predictions[offset])]
        confidence = float(probabilities[offset][int(predictions[offset])])
        signal = {'UP': 'BUY', 'DOWN': 'SELL'}.get(signal_label, 'HOLD') if confidence >= settings.min_confidence else 'HOLD'
        next_close = float(test.iloc[offset + 1]['close'])
        current_close = float(row['close'])
        trade_return = ((next_close - current_close) / current_close) if signal == 'BUY' else ((current_close - next_close) / current_close) if signal == 'SELL' else 0.0
        if signal != 'HOLD':
            net_return = trade_return - settings.transaction_cost
            capital *= 1 + net_return
            returns.append(net_return)
            trade_returns.append(net_return)
            trades.append({'type': signal, 'return': net_return, 'date': str(row['date']), 'confidence': confidence})
        equity_curve.append(capital)
    final_capital = capital
    return_series = pd.Series(returns, dtype=float)
    downside = return_series[return_series < 0]
    sortino = float(return_series.mean() / downside.std(ddof=1) * np.sqrt(252)) if len(downside) > 1 and downside.std(ddof=1) else 0.0
    result = {
        'initial_capital': initial_capital,
        'final_capital': float(final_capital),
        'total_return': float((final_capital - initial_capital) / initial_capital),
        'cumulative_return': float((final_capital - initial_capital) / initial_capital),
        'max_drawdown': float(calculate_drawdown(pd.Series(equity_curve))),
        'sharpe_ratio': float(calculate_sharpe(return_series)),
        'sortino_ratio': sortino,
        'win_rate': float(np.mean(np.array(trade_returns) > 0)) if trade_returns else 0.0,
        'number_of_trades': int(len(trades)),
        'average_trade_return': float(np.mean(trade_returns)) if trade_returns else 0.0,
        'largest_loss': float(min(trade_returns)) if trade_returns else 0.0,
        'largest_gain': float(max(trade_returns)) if trade_returns else 0.0,
        'test_period': {'start': str(test['date'].min()), 'end': str(test['date'].max())},
        'signal_source': 'stored Model 1 predictions generated without realized next-day returns',
        'equity_curve': equity_curve,
        'trades': trades,
        'buy_trades': sum(1 for t in trades if t['type'] == 'BUY'),
        'sell_trades': sum(1 for t in trades if t['type'] == 'SELL'),
        'hold_periods': int(len(test) - 1 - len(trades)),
    }

    out_path = ROOT_DIR / 'reports' / 'backtest' / 'backtest_report.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(result, file, indent=2)
    logger.info('Saved backtest report to %s', out_path)
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_backtest()
