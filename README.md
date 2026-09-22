# AI-Assisted HDFC Bank Trading Prediction System

This project builds an explainable AI-assisted trading framework for HDFC Bank Limited (NSE symbol: HDFCBANK) using approximately the most recent 5 years of daily NSE India market data. The system uses the Python `nsepython` library for programmatic NSE access and includes feature engineering, model training, evaluation, risk checks, signal generation, backtesting, and a simple API/dashboard interface.

## Scope

- Focused on HDFC Bank only in Phase 1
- Architecture is modular so additional Indian banking stocks can be added later
- AI-assisted decision support only; not autonomous trading or financial advice

## Project goals

- Fetch historical market data automatically
- Build a clean feature set with technical indicators
- Train an XGBoost classifier for UP / DOWN / NEUTRAL movements
- Generate confidence scores and validation filters
- Convert predictions into BUY / HOLD / SELL signals with risk rules
- Run backtests against buy-and-hold and AI strategy
- Expose results through FastAPI
- Display a simple dashboard frontend

## Folder structure

- backend/
- src/
- data/
- models/
- reports/
- frontend/
- tests/

## Python environment

This project targets Python 3.11+.

### Create environment

```bash
cd d:\banking-ai-trading
python -m venv .venv
```

On Windows PowerShell:

```powershell
cd d:\banking-ai-trading
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Environment configuration

Copy the example environment file and adjust values if needed:

```bash
copy .env.example .env
```

## Data download

```bash
python -m src.data.downloader
```

This fetches approximately 5 years of daily HDFC Bank data from NSE India through `nsepython` into:

- data/raw/market/HDFCBANK_NSE_raw.csv

## Data preprocessing and feature generation

```bash
python -m src.data.preprocessing
```

This builds a processed dataset under:

- data/processed/

## Train model

```bash
python -m src.models.train
```

## Evaluate model

```bash
python -m src.models.evaluate
```

## Run backtest

```bash
python -m src.backtesting.backtester
```

## Start backend API

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8001
```

## Start frontend

The frontend is plain HTML/CSS/JavaScript. No npm installation is required. In a second terminal run:

```bash
cd frontend
python -m http.server 5500
```

Open http://localhost:5500. Backend documentation is available at http://127.0.0.1:8001/docs.

## Core API endpoints

- GET /api/health
- GET /api/market/hdfcbank
- GET /api/market/hdfcbank/chart
- GET /api/market/hdfcbank/indicators
- GET /api/news/hdfcbank
- GET /api/news/prediction/hdfcbank
- GET /api/prediction/hdfcbank
- GET /api/signal/hdfcbank
- GET /api/backtest/hdfcbank
- GET /api/data/status
- POST /api/refresh
- GET /api/market/latest
- GET /api/market/history
- GET /api/prediction/latest
- GET /api/signal/latest
- GET /api/risk/latest
- GET /api/backtest
- GET /api/model/metrics
- GET /api/explanation
- POST /api/train
- POST /api/predict

## Important safeguards

- No future data leakage: all features use only information known at the time
- All displayed metrics are computed from actual data
- Validation layer prevents impossible or missing outputs
- Model is decision support only and does not implement autonomous trading

## Known limitations

- Phase 1 MVP uses NSE India historical daily price data through `nsepython` only
- Sentiment features are optional and may be lightweight or disabled if unavailable
- Performance should be interpreted as research-oriented and not financial advice

## Phase 2 ideas

- Add more Indian banking tickers and cross-sectional models
- Add SHAP explanations and richer sentiment models
- Add database-backed storage with PostgreSQL
- Add advanced risk and portfolio optimization
- Add live alerts and strategy monitoring

## AI-assisted decision support disclaimer

AI-assisted decision support. Not financial advice.
