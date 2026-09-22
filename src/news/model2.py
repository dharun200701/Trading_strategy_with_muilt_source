from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from src.config import ROOT_DIR, settings

EVENT_PATH = ROOT_DIR / 'data' / 'processed' / 'news' / 'HDFCBANK_news_event_labeled.csv'
MODEL_PATH = ROOT_DIR / 'models' / 'trained' / 'hdfcbank_news_event_model.pkl'
METADATA_PATH = ROOT_DIR / 'models' / 'metadata' / 'hdfcbank_news_event_model_metadata.json'
REPORT_PATH = ROOT_DIR / 'reports' / 'model' / 'model2_report.json'
FUTURE_INPUTS = {'post_news_return', 'next_day_return', 'abnormal_return', 'impact_score', 'volume_after', 'volatility_after', 'news_day_close', 'next_day_close', 'market_reaction'}


def _inputs(data: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    text = (data.get('headline', '').fillna('') + ' ' + data.get('clean_text', '').fillna('')).astype(str)
    numeric = data[['text_positive_probability', 'text_neutral_probability', 'text_negative_probability', 'timestamp_available']].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    return text, numeric


def train_model2() -> dict[str, Any]:
    data = pd.read_csv(EVENT_PATH).sort_values(['news_date', 'article_id']).reset_index(drop=True)
    if FUTURE_INPUTS.intersection(data.columns):
        input_columns = {'headline', 'clean_text', 'text_positive_probability', 'text_neutral_probability', 'text_negative_probability', 'timestamp_available'}
    else:
        raise RuntimeError('Event dataset does not contain the expected publication-time fields.')
    data = data.dropna(subset=['target_label']).reset_index(drop=True)
    data['news_date'] = pd.to_datetime(data['news_date'], errors='coerce')
    data = data.dropna(subset=['news_date']).sort_values('news_date').reset_index(drop=True)
    train = data[data['news_date'] <= pd.Timestamp('2024-12-31')].copy()
    validation = data[(data['news_date'] >= pd.Timestamp('2025-01-01')) & (data['news_date'] <= pd.Timestamp('2025-12-31'))].copy()
    test = data[data['news_date'] >= pd.Timestamp('2026-01-01')].copy()
    if train.empty or validation.empty or test.empty:
        raise RuntimeError('News data does not contain the required 2021-2024 / 2025 / 2026 chronological windows.')
    partitions = [train, validation, test]
    vectorizer = TfidfVectorizer(max_features=2500, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    scaler = StandardScaler()
    train_text, train_numeric = _inputs(partitions[0])
    train_matrix = hstack([vectorizer.fit_transform(train_text), scaler.fit_transform(train_numeric)])
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=settings.seed)
    model.fit(train_matrix, partitions[0]['target_label'])
    metrics = {}
    for name, frame in zip(['train', 'validation', 'test'], partitions):
        text, numeric = _inputs(frame)
        matrix = hstack([vectorizer.transform(text), scaler.transform(numeric)])
        predicted = model.predict(matrix)
        metrics[name] = {'accuracy': float(accuracy_score(frame['target_label'], predicted)), 'balanced_accuracy': float(balanced_accuracy_score(frame['target_label'], predicted)), 'macro_f1': float(f1_score(frame['target_label'], predicted, average='macro', zero_division=0)), 'classification_report': classification_report(frame['target_label'], predicted, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(frame['target_label'], predicted, labels=['POSITIVE', 'NEUTRAL', 'NEGATIVE']).tolist(), 'rows': len(frame)}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({'model': model, 'vectorizer': vectorizer, 'scaler': scaler, 'classes': list(model.classes_)}, MODEL_PATH)
    metadata = {'model': 'TF-IDF + balanced LogisticRegression', 'inputs': sorted(input_columns), 'future_inputs_excluded': sorted(FUTURE_INPUTS), 'split': {'train': len(partitions[0]), 'validation': len(partitions[1]), 'test': len(partitions[2])}, 'test_period': {'start': str(partitions[2]['news_date'].min()), 'end': str(partitions[2]['news_date'].max())}, 'label_caveat': 'Targets are observed daily market-reaction labels; 617 records are low-quality/confounded and there are no benchmark or intraday observations.', 'metrics': metrics}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    REPORT_PATH.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def predict_text(data: pd.DataFrame) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model 2 artifact missing at {MODEL_PATH}.')
    artifact = joblib.load(MODEL_PATH)
    text, numeric = _inputs(data)
    matrix = hstack([artifact['vectorizer'].transform(text), artifact['scaler'].transform(numeric)])
    probabilities = artifact['model'].predict_proba(matrix)[0]
    predicted = artifact['model'].classes_[int(np.argmax(probabilities))]
    return {'prediction': str(predicted), 'confidence': float(np.max(probabilities)), 'probabilities': {str(label): float(probability) for label, probability in zip(artifact['model'].classes_, probabilities)}}


if __name__ == '__main__':
    print(json.dumps(train_model2(), indent=2))
