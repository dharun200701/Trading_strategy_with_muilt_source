from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import ROOT_DIR, settings
from src.data.nse_provider import fetch_hdfc_market_data
from src.models.train import add_features, load_data
from src.models.evaluate import evaluate_model
from src.models.train import train_model
from src.risk.risk_manager import RiskManager
from src.signals.signal_engine import generate_signal
from src.validation.output_validator import validate_prediction_output

logger = logging.getLogger(__name__)


class PipelineService:
    def __init__(self) -> None:
        self.model_path = ROOT_DIR / 'models' / 'trained' / f'{settings.model_name}.pkl'
        self.metadata_path = ROOT_DIR / 'models' / 'metadata' / f'{settings.model_name}_metadata.json'

    def fetch_latest_market_data(self) -> pd.DataFrame:
        raw_path = ROOT_DIR / 'data' / 'raw' / 'market' / 'HDFCBANK_NSE_raw.csv'
        if not raw_path.exists():
            return fetch_hdfc_market_data(force_refresh=False)
        data = pd.read_csv(raw_path)
        if data.empty:
            raise RuntimeError(f'Market data file is empty: {raw_path}')
        return data

    def market_metadata(self) -> dict[str, Any]:
        metadata_path = ROOT_DIR / 'data' / 'raw' / 'market' / 'metadata.json'
        if metadata_path.exists():
            return json.loads(metadata_path.read_text(encoding='utf-8'))
        return {}

    def latest_risk(self) -> dict[str, Any]:
        prepared = add_features(load_data())
        if prepared.empty:
            raise RuntimeError('No valid feature row is available for risk assessment.')
        latest = prepared.dropna(subset=['rolling_volatility_20', 'atr_14']).iloc[-1]
        risk_manager = RiskManager(account_capital=settings.account_capital, risk_per_trade=settings.risk_per_trade)
        return risk_manager.assess_risk(float(latest['rolling_volatility_20']), float(latest['atr_14']))

    def latest_news(self, limit: int = 10) -> dict[str, Any]:
        path = ROOT_DIR / 'data' / 'processed' / 'news' / 'HDFCBANK_news_event_labeled.csv'
        if not path.exists():
            raise FileNotFoundError(f'News event dataset missing at {path}.')
        data = pd.read_csv(path).sort_values(['news_date', 'article_id'], ascending=[False, False]).head(limit)
        columns = ['article_id', 'news_datetime', 'news_date', 'headline', 'source_name', 'url', 'text_sentiment',
                   'text_positive_probability', 'text_neutral_probability', 'text_negative_probability',
                   'market_reaction', 'impact_score', 'impact_level', 'reaction_priority', 'label_confidence',
                   'label_quality', 'target_label', 'label_reason']
        selected = [column for column in columns if column in data.columns]
        return {'status': 'AVAILABLE', 'source': 'project news dataset',
                'timestamp_note': 'Publication timestamps are unavailable for some records.',
                'data': data[selected].where(data[selected].notna(), None).to_dict(orient='records')}

    def latest_signal(self) -> dict[str, Any]:
        prediction = self.latest_prediction()
        article = (self.latest_news(limit=1).get('data') or [None])[0]
        news_direction = article.get('market_reaction') if article else None
        impact = float(article.get('impact_score') or 0.0) if article else 0.0
        confidence = float(prediction.get('confidence') or 0.0)
        signal = 'HOLD'
        reason = 'Confidence filter or unavailable news/risk data kept the signal at HOLD.'
        if confidence >= settings.min_confidence:
            signal = {'UP': 'BUY', 'DOWN': 'SELL'}.get(prediction['prediction'], 'HOLD')
            if impact >= 60 and news_direction in {'POSITIVE', 'NEGATIVE'}:
                conflict = (signal == 'BUY' and news_direction == 'NEGATIVE') or (signal == 'SELL' and news_direction == 'POSITIVE')
                agreement = (signal == 'BUY' and news_direction == 'POSITIVE') or (signal == 'SELL' and news_direction == 'NEGATIVE')
                if conflict:
                    signal, reason = 'HOLD', 'High-impact news conflicts with the market model.'
                elif agreement:
                    reason = 'Market model and high-impact event direction agree.'
        return {'status': 'AVAILABLE', 'signal': signal, 'model1_prediction': prediction,
                'news_direction': news_direction, 'news_impact_score': impact,
                'confidence': confidence, 'reason': reason}

    def model_status(self) -> dict[str, Any]:
        event_path = ROOT_DIR / 'data' / 'processed' / 'news' / 'HDFCBANK_news_event_labeled.csv'
        model2_path = ROOT_DIR / 'models' / 'trained' / 'hdfcbank_news_event_model.pkl'
        return {'status': 'AVAILABLE', 'model1': {'status': 'AVAILABLE' if self.model_path.exists() else 'UNAVAILABLE', 'artifact': str(self.model_path)},
                'model2': {'status': 'AVAILABLE' if model2_path.exists() else 'UNAVAILABLE', 'artifact': str(model2_path),
                           'label_dataset': str(event_path), 'note': 'Event labels use observed daily market reactions; intraday and benchmark inputs are unavailable.'}}

    def latest_news_prediction(self) -> dict[str, Any]:
        from src.news.model2 import predict_text
        data = pd.read_csv(ROOT_DIR / 'data' / 'processed' / 'news' / 'HDFCBANK_news_event_labeled.csv').sort_values(['news_date', 'article_id']).tail(1)
        prediction = predict_text(data)
        article = data.iloc[0]
        return {'status': 'AVAILABLE', 'article_id': article.get('article_id'), 'news_date': article.get('news_date'), **prediction}

    def model_explanation(self) -> dict[str, Any]:
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f'Model missing at {self.model_path}. Run training first.')
        model = joblib.load(self.model_path)
        metadata = json.loads(self.metadata_path.read_text(encoding='utf-8'))
        feature_columns = metadata.get('feature_columns') or metadata.get('feature_names', [])
        importances = getattr(model, 'feature_importances_', None)
        if importances is None or len(importances) != len(feature_columns):
            raise RuntimeError('The trained model does not expose compatible feature importance data.')
        features = [
            {'name': name, 'importance': float(importance)}
            for name, importance in sorted(zip(feature_columns, importances), key=lambda item: item[1], reverse=True)
        ]
        return {
            'status': 'AVAILABLE',
            'model': 'XGBoost',
            'features': features,
            'note': 'Features that contributed strongly to the model prediction.',
        }

    def latest_prediction(self) -> dict[str, Any]:
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f'Model missing at {self.model_path}. Run training first.')

        latest_df = load_data()
        prepared = add_features(latest_df).replace([float('inf'), float('-inf')], pd.NA).dropna().reset_index(drop=True)
        metadata = json.loads(Path(self.metadata_path).read_text(encoding='utf-8'))
        feature_columns = metadata.get('feature_columns') or metadata.get('feature_names', [])
        missing = [column for column in feature_columns if column not in prepared.columns]
        if missing:
            raise ValueError(f'Model features are missing from the canonical feature frame: {missing[:5]}')
        X = prepared[feature_columns]
        model = joblib.load(self.model_path)
        probs = model.predict_proba(X)
        prediction = model.predict(X).astype(int)
        target_classes = metadata.get('target_classes') or ['UP', 'DOWN', 'NEUTRAL']
        class_labels = {int(class_id): label for class_id, label in enumerate(target_classes)}
        pred_label = class_labels[int(prediction[-1])]
        prob_row = probs[-1]
        class_order = model.classes_
        confidence = float(prob_row[int(prediction[-1])])
        risk_manager = RiskManager(account_capital=settings.account_capital, risk_per_trade=settings.risk_per_trade)
        technical_metrics = {
            'volatility': float(prepared['rolling_volatility_20'].iloc[-1]),
            'atr': float(prepared['atr_14'].iloc[-1]),
        }
        signal = generate_signal(pred_label, confidence, technical_metrics, min_confidence=settings.min_confidence)
        risk = risk_manager.assess_risk(technical_metrics['volatility'], technical_metrics['atr'])
        output = {
            'prediction': pred_label,
            'confidence': confidence,
            'probability': confidence,
            'probability_up': float(prob_row[list(class_order).index(0)]) if 0 in class_order else None,
            'probability_down': float(prob_row[list(class_order).index(1)]) if 1 in class_order else None,
            'probability_neutral': float(prob_row[list(class_order).index(2)]) if 2 in class_order else None,
            'signal': signal,
            'timestamp': str(prepared['Date'].iloc[-1]),
            'prediction_timestamp': str(prepared['Date'].iloc[-1]),
            'model_version': metadata.get('model_name', settings.model_name),
            'feature_version': 'market_features_v2_canonical',
            'symbol': settings.ticker,
            'data_source': settings.market_data_source,
            'risk': risk,
            'backtest_metrics': {},
        }
        is_valid = validate_prediction_output(output)
        output['status'] = 'VALIDATED' if is_valid else 'FILTERED'
        output['validation_warning'] = 'Validation OK.' if is_valid else 'Validation failed.'
        if is_valid:
            return output
        return {
            'prediction': 'NEUTRAL',
            'confidence': 0.0,
            'signal': 'HOLD',
            'timestamp': str(prepared['Date'].iloc[-1]),
            'model_version': 'unavailable',
            'risk': risk,
            'backtest_metrics': run_backtest(),
            'validation_warning': 'Validation failed; output set to HOLD / unavailable state.'
        }

    def run_training(self) -> dict[str, Any]:
        artifact = train_model()
        report = evaluate_model()
        return {'model': artifact.metadata, 'report': report}

    def get_model_metrics(self) -> dict[str, Any]:
        report_path = ROOT_DIR / 'reports' / 'model' / 'evaluation_report.json'
        if not report_path.exists():
            raise FileNotFoundError(f'Model metrics missing at {report_path}. Run the evaluation step first.')
        return json.loads(report_path.read_text(encoding='utf-8'))
