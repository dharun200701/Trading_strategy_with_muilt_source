from __future__ import annotations

from typing import Any


VALID_PREDICTIONS = {'UP', 'DOWN', 'NEUTRAL'}
VALID_SIGNALS = {'BUY', 'HOLD', 'SELL'}


def validate_prediction_output(output: dict[str, Any]) -> bool:
    if not isinstance(output, dict):
        return False
    if output.get('prediction') not in VALID_PREDICTIONS:
        return False
    if output.get('signal') not in VALID_SIGNALS:
        return False
    if not isinstance(output.get('confidence'), (int, float)):
        return False
    confidence = float(output['confidence'])
    if not 0.0 <= confidence <= 1.0:
        return False
    if not output.get('timestamp'):
        return False
    if not output.get('model_version'):
        return False
    if 'risk' not in output or output['risk'] is None:
        return False
    if 'backtest_metrics' not in output or output['backtest_metrics'] is None:
        return False
    for value in output.get('backtest_metrics', {}).values():
        if isinstance(value, float) and (value != value):
            return False
    return True
