from __future__ import annotations

import numpy as np
import pandas as pd


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    result = df.copy()
    result.columns = [str(col).strip().lower() for col in result.columns]
    if 'date' not in result.columns and 'datetime' in result.columns:
        result = result.rename(columns={'datetime': 'date'})
    if 'date' in result.columns:
        result['date'] = pd.to_datetime(result['date'])
        result = result.sort_values('date').reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col not in result.columns:
            if col == 'open':
                result['open'] = result.get('close', 0.0)
            elif col == 'high':
                result['high'] = result.get('close', 0.0)
            elif col == 'low':
                result['low'] = result.get('close', 0.0)
            elif col == 'close':
                result['close'] = result.get('close', 0.0)
            else:
                result['volume'] = 0.0
        result[col] = pd.to_numeric(result[col], errors='coerce')

    result['daily_return'] = result['close'].pct_change().fillna(0.0)
    result['previous_close'] = result['close'].shift(1)
    result['high_low_range'] = result['high'] - result['low']
    result['open_close_change'] = result['close'] - result['open']
    result['volume_change'] = result['volume'].pct_change().fillna(0.0)

    for window in [5, 10, 20, 50]:
        result[f'sma_{window}'] = result['close'].rolling(window=window, min_periods=window).mean()

    for window in [5, 10, 20]:
        result[f'ema_{window}'] = result['close'].ewm(span=window, adjust=False).mean()

    result['close_over_sma20'] = result['close'] / result['sma_20']
    result['close_over_sma50'] = result['close'] / result['sma_50']
    result['sma5_over_sma20'] = result['sma_5'] / result['sma_20']
    result['sma20_over_sma50'] = result['sma_20'] / result['sma_50']

    delta = result['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result['rsi_14'] = 100 - (100 / (1 + rs))
    result['rsi_14'] = result['rsi_14'].fillna(50.0)

    ema12 = result['close'].ewm(span=12, adjust=False).mean()
    ema26 = result['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    result['macd'] = macd
    result['macd_signal'] = signal
    result['macd_histogram'] = macd - signal
    result['roc_10'] = result['close'].pct_change(10).fillna(0.0)

    high_low = result['high'] - result['low']
    true_range = pd.concat([
        high_low,
        (result['high'] - result['close'].shift(1)).abs(),
        (result['low'] - result['close'].shift(1)).abs()
    ], axis=1).max(axis=1)
    result['atr_14'] = true_range.rolling(window=14, min_periods=14).mean()
    result['rolling_volatility_20'] = result['daily_return'].rolling(window=20, min_periods=20).std().fillna(0.0)

    sma_close = result['close'].rolling(window=20, min_periods=20).mean()
    std_close = result['close'].rolling(window=20, min_periods=20).std().fillna(0.0)
    result['bollinger_upper'] = sma_close + (2 * std_close)
    result['bollinger_lower'] = sma_close - (2 * std_close)
    result['bollinger_width'] = result['bollinger_upper'] - result['bollinger_lower']

    result['volume_sma_20'] = result['volume'].rolling(window=20, min_periods=20).mean()
    result['volume_ratio'] = result['volume'] / result['volume_sma_20']

    numeric_columns = result.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        result[column] = result[column].replace([np.inf, -np.inf], np.nan)
    # Forward-fill only uses information available at or before each row.
    result = result.ffill().fillna(0.0)
    return result
