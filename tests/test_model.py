import pandas as pd

from src.models.train import chronological_split, prepare_training_frame


def test_prepare_training_frame_includes_target_and_timestamp():
    df = [
        {'Date': '2020-01-01', 'Close': 100.0, 'daily_return': 0.01, 'sma_5': 99.0, 'rsi_14': 50.0},
        {'Date': '2020-01-02', 'Close': 101.0, 'daily_return': 0.02, 'sma_5': 100.0, 'rsi_14': 55.0},
        {'Date': '2020-01-03', 'Close': 102.0, 'daily_return': 0.01, 'sma_5': 101.0, 'rsi_14': 60.0},
    ]
    prepared = prepare_training_frame(df)
    assert 'target' in prepared.columns
    assert 'Date' in prepared.columns
    assert prepared['target'].isin(['UP', 'DOWN', 'NEUTRAL']).all()


def test_chronological_split_uses_2021_2024_train_2025_validation_and_2026_test():
    dates = pd.date_range('2021-01-01', '2026-12-31', freq='D')
    df = pd.DataFrame({'Date': dates, 'Close': range(len(dates))})
    train, validation, test = chronological_split(df)

    assert train['Date'].min() >= pd.Timestamp('2021-01-01')
    assert train['Date'].max() < pd.Timestamp('2025-01-01')
    assert validation['Date'].min() >= pd.Timestamp('2025-01-01')
    assert validation['Date'].max() < pd.Timestamp('2026-01-01')
    assert test['Date'].min() >= pd.Timestamp('2026-01-01')
