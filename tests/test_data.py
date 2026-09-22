from pathlib import Path


def test_data_folders_exist():
    root = Path(__file__).resolve().parent.parent
    assert (root / 'data' / 'raw' / 'market').exists()
    assert (root / 'data' / 'processed').exists()
    assert (root / 'models').exists()
    assert (root / 'reports').exists()
