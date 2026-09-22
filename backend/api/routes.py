from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi import status

from backend.services.pipeline_service import PipelineService
from src.config import settings
from src.config import ROOT_DIR
from src.data.nse_provider import fetch_hdfc_market_data
from src.features.technical_features import add_technical_features

router = APIRouter(prefix="/api")
service = PipelineService()


def unavailable(detail: str, code: str) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail={'code': code, 'message': detail})


@router.get("/health")
def health() -> dict:
    model_status = service.model_status()
    try:
        market = service.fetch_latest_market_data()
        market_status = 'available' if not market.empty else 'unavailable'
        last_update = str(market['date'].max())
    except Exception:
        market_status, last_update = 'unavailable', None
    try:
        news_status = 'available' if service.latest_news(limit=1).get('data') else 'unavailable'
    except Exception:
        news_status = 'unavailable'
    return {
        "status": "ok",
        "message": "AI-assisted decision support. Not financial advice.",
        "api_port": settings.port,
        "data_source": settings.market_data_source,
        "provider": settings.market_data_provider,
        "market_data_status": market_status,
        "news_status": news_status,
        "model1_status": model_status['model1']['status'].lower(),
        "model2_status": model_status['model2']['status'].lower(),
        "storage_status": 'available',
        "last_successful_update": last_update,
    }


@router.get("/market/latest")
def market_latest() -> dict:
    try:
        data = service.fetch_latest_market_data()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise unavailable(str(exc), 'MARKET_DATA_UNAVAILABLE') from exc
    if data.empty:
        raise unavailable("No HDFCBANK market data is available.", 'MARKET_DATA_UNAVAILABLE')
    latest = data.iloc[-1]
    metadata = service.market_metadata()
    return {
        "symbol": settings.ticker,
        "name": "HDFC BANK",
        "date": str(latest['date']),
        "open": float(latest['open']),
        "high": float(latest['high']),
        "low": float(latest['low']),
        "close": float(latest['close']),
        "volume": int(latest['volume']),
        "previous_close": float(latest['previous_close']) if 'previous_close' in latest and latest['previous_close'] == latest['previous_close'] else None,
        "data_source": settings.market_data_source,
        "provider": settings.market_data_provider,
        "data_timestamp": metadata.get('download_timestamp', str(latest['date'])),
        "status": "AVAILABLE",
    }


@router.get('/market/hdfcbank')
def market_hdfcbank() -> dict:
    return market_latest()


@router.get("/market/history")
def market_history() -> dict:
    try:
        data = add_technical_features(service.fetch_latest_market_data())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise unavailable(str(exc), 'MARKET_DATA_UNAVAILABLE') from exc
    records = data.where(data.notna(), None).to_dict(orient="records")
    return {"symbol": settings.ticker, "status": "AVAILABLE", "data_source": settings.market_data_source, "provider": settings.market_data_provider, "data": records}


@router.get('/market/hdfcbank/chart')
def market_chart() -> dict:
    return market_history()


@router.get('/market/hdfcbank/indicators')
def market_indicators() -> dict:
    payload = market_history()
    return {'symbol': settings.ticker, 'status': payload['status'], 'data': payload['data'][-1] if payload['data'] else None}


@router.get("/prediction/latest")
def prediction_latest() -> dict:
    if not service.model_path.exists() or not service.metadata_path.exists():
        raise unavailable(f'Model missing at {service.model_path}. Run training first.', 'MODEL_NOT_READY')
    try:
        return service.latest_prediction()
    except FileNotFoundError as exc:
        raise unavailable(str(exc), 'MARKET_DATA_UNAVAILABLE') from exc
    except (RuntimeError, ValueError, KeyError) as exc:
        raise unavailable(str(exc), 'PREDICTION_UNAVAILABLE') from exc


@router.get("/signal/latest")
def signal_latest() -> dict:
    try:
        return service.latest_signal()
    except FileNotFoundError as exc:
        raise unavailable(str(exc), 'MODEL_NOT_READY') from exc


@router.get("/risk/latest")
def risk_latest() -> dict:
    try:
        return service.latest_risk()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise unavailable(str(exc), 'RISK_UNAVAILABLE') from exc


@router.get("/backtest")
def backtest() -> dict:
    backtest_path = ROOT_DIR / "reports" / "backtest" / "backtest_report.json"
    if not backtest_path.exists():
        raise unavailable("Backtest report not generated yet. Run the backtest pipeline first.", 'BACKTEST_NOT_READY')
    return {**json.loads(backtest_path.read_text(encoding="utf-8")), "status": "AVAILABLE"}


@router.get("/model/metrics")
def model_metrics() -> dict:
    try:
        return service.get_model_metrics()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get('/news/hdfcbank')
def news_hdfcbank(limit: int = 10) -> dict:
    try:
        return service.latest_news(limit=max(1, min(limit, 100)))
    except (FileNotFoundError, ValueError) as exc:
        raise unavailable(str(exc), 'NEWS_UNAVAILABLE') from exc


@router.get('/prediction/hdfcbank')
def prediction_hdfcbank() -> dict:
    return prediction_latest()


@router.get('/signal/hdfcbank')
def signal_hdfcbank() -> dict:
    return signal_latest()


@router.get('/backtest/hdfcbank')
def backtest_hdfcbank() -> dict:
    return backtest()


@router.get('/model/performance')
def model_performance() -> dict:
    return service.model_status()


@router.get('/news/prediction/hdfcbank')
def news_prediction_hdfcbank() -> dict:
    try:
        return service.latest_news_prediction()
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise unavailable(str(exc), 'MODEL2_UNAVAILABLE') from exc


@router.get('/data/status')
def data_status() -> dict:
    try:
        market = service.fetch_latest_market_data()
        news = service.latest_news(limit=1)
        return {'status': 'AVAILABLE', 'market_rows': len(market), 'market_start': str(market['date'].min()), 'market_end': str(market['date'].max()), 'news': news['status']}
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise unavailable(str(exc), 'DATA_UNAVAILABLE') from exc


@router.post('/refresh')
def refresh() -> dict:
    try:
        data = fetch_hdfc_market_data(force_refresh=False)
        return {'status': 'AVAILABLE', 'message': 'Latest available NSE data refreshed from the configured nsepython integration.', 'rows': len(data), 'latest_market_timestamp': str(data['date'].max())}
    except Exception as exc:
        raise unavailable(f'Latest NSE data could not be retrieved: {exc}', 'REFRESH_UNAVAILABLE') from exc


@router.get("/explanation")
def explanation() -> dict:
    if not service.model_path.exists() or not service.metadata_path.exists():
        raise unavailable(f'Model missing at {service.model_path}. Run training first.', 'MODEL_NOT_READY')
    try:
        return service.model_explanation()
    except FileNotFoundError as exc:
        raise unavailable(str(exc), 'MARKET_DATA_UNAVAILABLE') from exc
    except RuntimeError as exc:
        raise unavailable(str(exc), 'EXPLANATION_UNAVAILABLE') from exc


@router.post("/train")
def train() -> dict:
    return service.run_training()


@router.post("/predict")
def predict() -> dict:
    try:
        return service.latest_prediction()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
