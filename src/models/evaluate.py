from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd

from src.config import ROOT_DIR, settings
from src.models.train import prepare_training_frame, chronological_split

logger = logging.getLogger(__name__)


def evaluate_model() -> dict:
    model_path = ROOT_DIR / 'models' / 'trained' / f'{settings.model_name}.pkl'
    metadata_path = ROOT_DIR / 'models' / 'metadata' / f'{settings.model_name}_metadata.json'
    if not model_path.exists():
        raise FileNotFoundError(f'Model not found at {model_path}. Run training first.')

    model = joblib.load(model_path)
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)

    dataset = pd.read_csv(ROOT_DIR / 'data' / 'processed' / 'hdfcbank_preprocessed.csv')
    prepared = prepare_training_frame(dataset)
    _, _, test_df = chronological_split(prepared)
    feature_columns = metadata['feature_columns']
    X_test = test_df[feature_columns]
    y_true = test_df['target']
    encoded_pred = model.predict(X_test).astype(int)
    target_classes = metadata.get('target_classes', ['DOWN', 'NEUTRAL', 'UP'])
    y_pred = [target_classes[index] for index in encoded_pred]

    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    report = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='macro', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='macro', zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=['UP', 'DOWN', 'NEUTRAL']).tolist(),
        'class_metrics': {
            'UP': {
                'precision': float(precision_score(y_true, y_pred, labels=['UP'], average='macro', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, labels=['UP'], average='macro', zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, labels=['UP'], average='macro', zero_division=0)),
            },
            'DOWN': {
                'precision': float(precision_score(y_true, y_pred, labels=['DOWN'], average='macro', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, labels=['DOWN'], average='macro', zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, labels=['DOWN'], average='macro', zero_division=0)),
            },
            'NEUTRAL': {
                'precision': float(precision_score(y_true, y_pred, labels=['NEUTRAL'], average='macro', zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, labels=['NEUTRAL'], average='macro', zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, labels=['NEUTRAL'], average='macro', zero_division=0)),
            },
        },
        'model_version': metadata.get('training_date'),
        'dataset_period': {
            'start': prepared['date'].min().isoformat() if 'date' in prepared.columns else None,
            'end': prepared['date'].max().isoformat() if 'date' in prepared.columns else None,
        },
    }

    out_path = ROOT_DIR / 'reports' / 'model' / 'evaluation_report.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as file:
        json.dump(report, file, indent=2)
    logger.info('Saved evaluation report to %s', out_path)
    return report


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    evaluate_model()
