from src.validation.output_validator import validate_prediction_output


def test_validate_prediction_output_accepts_valid_record():
    record = {
        'prediction': 'UP',
        'confidence': 0.75,
        'signal': 'BUY',
        'timestamp': '2024-01-01',
        'model_version': 'v1',
        'risk': {'volatility': 0.02},
        'backtest_metrics': {'final_capital': 1100.0}
    }
    assert validate_prediction_output(record) is True
