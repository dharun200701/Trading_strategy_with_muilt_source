import pandas as pd

from src.features.technical_features import add_technical_features


def test_add_technical_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            'Date': pd.date_range('2020-01-01', periods=60, freq='D'),
            'Open': [100.0 + i for i in range(60)],
            'High': [101.0 + i for i in range(60)],
            'Low': [99.0 + i for i in range(60)],
            'Close': [100.5 + i for i in range(60)],
            'Volume': [1000 + i * 10 for i in range(60)],
        }
    )
    result = add_technical_features(df)
    assert 'daily_return' in result.columns
    assert 'sma_5' in result.columns
    assert 'ema_20' in result.columns
    assert 'rsi_14' in result.columns
    assert 'atr_14' in result.columns
    assert 'bollinger_upper' in result.columns
