from src.signals.signal_engine import generate_signal


def test_generate_signal_returns_valid_label():
    signal = generate_signal('UP', 0.8, {'volatility': 0.02, 'atr': 1.5}, 0.01, 0.1)
    assert signal in {'BUY', 'HOLD', 'SELL'}
