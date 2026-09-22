from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import ROOT_DIR, settings
from src.models.train import (
    CLASS_NAMES,
    CLASS_TO_INT,
    INT_TO_CLASS,
    add_features,
    create_target,
    get_feature_columns,
    load_data,
)
from src.news.model2 import _inputs

logger = logging.getLogger(__name__)

MARKET_PATH = ROOT_DIR / "data" / "raw" / "market" / "HDFCBANK_NSE_raw.csv"
NEWS_PATH = ROOT_DIR / "data" / "processed" / "news" / "HDFCBANK_news_event_labeled.csv"
MODEL1_PATH = ROOT_DIR / "models" / "trained" / "hdfcbank_xgboost.pkl"
MODEL2_PATH = ROOT_DIR / "models" / "trained" / "hdfcbank_news_event_model.pkl"
MODEL1_METADATA_PATH = ROOT_DIR / "models" / "metadata" / "hdfcbank_xgboost_metadata.json"
REPORT_DIR = ROOT_DIR / "reports" / "model"
REPORT_PATH = REPORT_DIR / "ensemble_report.json"
PREDICTIONS_PATH = REPORT_DIR / "ensemble_predictions.csv"

XGB_WEIGHT = 0.70
NEWS_WEIGHT = 0.30
RANDOM_STATE = int(getattr(settings, "seed", 42))
NEWS_TO_MARKET = {"POSITIVE": "UP", "NEGATIVE": "DOWN", "NEUTRAL": "NEUTRAL"}


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def load_model1() -> Any:
    _require_file(MODEL1_PATH, "Model 1")
    return joblib.load(MODEL1_PATH)


def load_model2() -> Dict[str, Any]:
    _require_file(MODEL2_PATH, "Model 2")
    artifact = joblib.load(MODEL2_PATH)
    required = ["model", "vectorizer", "scaler", "classes"]
    missing = [key for key in required if key not in artifact]
    if missing:
        raise ValueError(f"Model 2 artifact missing keys: {missing}")
    return artifact


def load_model1_metadata() -> Dict[str, Any]:
    if MODEL1_METADATA_PATH.exists():
        try:
            return json.loads(MODEL1_METADATA_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.warning("Model 1 metadata exists but is not valid JSON: %s", exc)
    return {}


def predict_market(model: Any) -> pd.DataFrame:
    market = load_data()
    if market.empty:
        raise RuntimeError("No market data available for Model 1 inference.")

    prepared = add_features(market)
    threshold = float(getattr(settings, "target_threshold", 0.003))
    prepared = create_target(prepared, threshold)

    metadata = load_model1_metadata()
    feature_columns = metadata.get("feature_names") or get_feature_columns(prepared)

    if not feature_columns:
        raise ValueError("No usable Model 1 feature columns were available.")

    missing = [column for column in feature_columns if column not in prepared.columns]
    if missing:
        raise ValueError(f"Model 1 metadata feature columns are missing from the prepared market frame: {missing[:10]}")

    prepared = prepared.dropna(subset=feature_columns).reset_index(drop=True)
    if prepared.empty:
        raise RuntimeError("Market frame became empty after feature construction and required-column filtering.")

    X = prepared[feature_columns]
    probabilities = model.predict_proba(X)
    model_classes = list(model.classes_)

    probability_frame = pd.DataFrame(probabilities, columns=model_classes, index=prepared.index)
    for class_name in CLASS_NAMES:
        class_id = CLASS_TO_INT.get(class_name)
        if class_id is not None and class_name in probability_frame.columns:
            prepared[f"xgb_probability_{class_name.lower()}"] = probability_frame[class_name].values
        else:
            prepared[f"xgb_probability_{class_name.lower()}"] = 0.0

    predictions = model.predict(X)
    prepared["xgb_prediction"] = [INT_TO_CLASS.get(int(prediction), str(prediction)) for prediction in predictions]
    prepared["market_target"] = prepared["Target"]
    return prepared


def predict_news(news: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    vectorizer = artifact["vectorizer"]
    scaler = artifact["scaler"]
    model = artifact["model"]

    text, numeric = _inputs(news)
    text_matrix = vectorizer.transform(text)
    numeric_matrix = scaler.transform(numeric)
    matrix = hstack([text_matrix, numeric_matrix])
    probabilities = model.predict_proba(matrix)
    classes = model.classes_

    probability_columns: Dict[str, np.ndarray] = {}
    for index, label in enumerate(classes):
        probability_columns[f"news_probability_{str(label).lower()}"] = probabilities[:, index]

    result = news[["news_date", "article_id"]].copy()
    for column, values in probability_columns.items():
        result[column] = values

    probability_columns_list = list(probability_columns.keys())
    daily = result.groupby("news_date")[probability_columns_list].mean().reset_index()
    daily["news_article_count"] = result.groupby("news_date").size().values
    return daily


def map_news_probabilities(news_daily: pd.DataFrame) -> pd.DataFrame:
    result = news_daily.copy()

    result["news_up_probability"] = result.get("news_probability_positive", 0.0)
    result["news_down_probability"] = result.get("news_probability_negative", 0.0)
    result["news_neutral_probability"] = result.get("news_probability_neutral", 0.0)

    probability_sum = result[["news_up_probability", "news_down_probability", "news_neutral_probability"]].sum(axis=1)
    probability_sum = probability_sum.replace(0, np.nan)

    for column in ["news_up_probability", "news_down_probability", "news_neutral_probability"]:
        result[column] = (result[column] / probability_sum).fillna(1 / 3)

    return result


def align_data(market_predictions: pd.DataFrame, news_predictions: pd.DataFrame) -> pd.DataFrame:
    market = market_predictions.copy()
    news = news_predictions.copy()

    if "Date" not in market.columns:
        raise ValueError("Market predictions do not contain a Date column.")
    if "news_date" not in news.columns:
        raise ValueError("News predictions do not contain a news_date column.")

    market["Date"] = pd.to_datetime(market["Date"], errors="coerce").dt.normalize()
    news["news_date"] = pd.to_datetime(news["news_date"], errors="coerce").dt.normalize()

    market = market.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    news = news.dropna(subset=["news_date"]).sort_values("news_date").reset_index(drop=True)

    aligned = pd.merge(market, news, left_on="Date", right_on="news_date", how="inner")
    aligned = aligned.sort_values("Date").reset_index(drop=True)

    if aligned.empty:
        raise RuntimeError("No common evaluation dates were available between market and news data.")

    return aligned


def combine_predictions(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()

    result["ensemble_up_probability"] = XGB_WEIGHT * result["xgb_probability_up"] + NEWS_WEIGHT * result["news_up_probability"]
    result["ensemble_down_probability"] = XGB_WEIGHT * result["xgb_probability_down"] + NEWS_WEIGHT * result["news_down_probability"]
    result["ensemble_neutral_probability"] = XGB_WEIGHT * result["xgb_probability_neutral"] + NEWS_WEIGHT * result["news_neutral_probability"]

    probability_columns = [
        "ensemble_up_probability",
        "ensemble_down_probability",
        "ensemble_neutral_probability",
    ]

    result["ensemble_prediction"] = (
        result[probability_columns].idxmax(axis=1)
        .str.replace("ensemble_", "", regex=False)
        .str.replace("_probability", "", regex=False)
        .str.upper()
    )

    news_probability_columns = [
        "news_up_probability",
        "news_down_probability",
        "news_neutral_probability",
    ]
    result["news_prediction"] = (
        result[news_probability_columns].idxmax(axis=1)
        .str.replace("news_", "", regex=False)
        .str.replace("_probability", "", regex=False)
        .str.upper()
    )

    return result


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    labels = CLASS_NAMES
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0),
        "rows": int(len(y_true)),
    }


def build_model2_news_frame() -> pd.DataFrame:
    _require_file(NEWS_PATH, "News dataset")
    data = pd.read_csv(NEWS_PATH)
    required = [
        "news_date",
        "article_id",
        "target_label",
        "headline",
        "clean_text",
        "text_positive_probability",
        "text_neutral_probability",
        "text_negative_probability",
        "timestamp_available",
    ]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"News dataset missing required columns: {missing}")

    data["news_date"] = pd.to_datetime(data["news_date"], errors="coerce")
    data = data.dropna(subset=["news_date"]).sort_values(["news_date", "article_id"]).reset_index(drop=True)
    if data.empty:
        raise RuntimeError("News dataset is empty after date filtering.")
    return data


def print_model_comparison(model1_metrics: Dict[str, Any], model2_metrics: Dict[str, Any], ensemble_metrics: Dict[str, Any], common_dates: int) -> None:
    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(f"Common evaluation dates: {common_dates}")

    def show(label: str, metrics: Dict[str, Any]) -> None:
        print(f"\n{label}")
        print("-" * 40)
        print(f"Accuracy            : {metrics['accuracy'] * 100:.2f}%")
        print(f"Balanced Accuracy   : {metrics['balanced_accuracy'] * 100:.2f}%")
        print(f"Macro Precision     : {metrics['precision_macro'] * 100:.2f}%")
        print(f"Macro Recall        : {metrics['recall_macro'] * 100:.2f}%")
        print(f"Macro F1            : {metrics['f1_macro'] * 100:.2f}%")
        print("Confusion Matrix:")
        print(np.array(metrics["confusion_matrix"]))

    show("Model 1 - XGBoost", model1_metrics)
    show("Model 2 - News", model2_metrics)
    show("Ensemble - Weighted Probability", ensemble_metrics)

    better_model = "Model 1" if model1_metrics["accuracy"] >= model2_metrics["accuracy"] else "Model 2"
    print(f"\nNumerically better single model on accuracy: {better_model}")
    if ensemble_metrics["accuracy"] >= max(model1_metrics["accuracy"], model2_metrics["accuracy"]):
        print("Ensemble accuracy is at least as good as the best individual model.")
    else:
        print("Ensemble accuracy is below the best individual model.")


def train_ensemble() -> Dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    print("\n" + "=" * 72)
    print("HDFCBANK MODEL ENSEMBLE")
    print("=" * 72)

    print("\nLoading Model 1...")
    model1 = load_model1()
    print(f"Loaded Model 1: {MODEL1_PATH}")

    print("\nLoading Model 2...")
    model2 = load_model2()
    print(f"Loaded Model 2: {MODEL2_PATH}")

    print("\nLoading and preparing market data...")
    market = load_data()
    market_predictions = predict_market(model1)
    print(f"Model 1 prediction rows: {len(market_predictions)}")

    print("\nLoading and preparing news data...")
    news = build_model2_news_frame()
    news_daily = predict_news(news, model2)
    news_daily = map_news_probabilities(news_daily)
    print(f"News daily rows: {len(news_daily)}")

    print("\nAligning market and news dates...")
    aligned = align_data(market_predictions, news_daily)
    print(f"Common aligned rows: {len(aligned)}")

    print("\nCombining model probabilities...")
    aligned = combine_predictions(aligned)

    y_true = aligned["market_target"].astype(str)

    model1_metrics = calculate_metrics(y_true, aligned["xgb_prediction"].astype(str))
    model2_labels = aligned["news_prediction"].astype(str)
    model2_metrics = calculate_metrics(y_true, model2_labels)
    ensemble_metrics = calculate_metrics(y_true, aligned["ensemble_prediction"].astype(str))

    print_model_comparison(model1_metrics, model2_metrics, ensemble_metrics, len(aligned))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_output = aligned[
        [
            "Date",
            "Close",
            "market_target",
            "xgb_prediction",
            "xgb_probability_up",
            "xgb_probability_down",
            "xgb_probability_neutral",
            "news_prediction",
            "news_up_probability",
            "news_down_probability",
            "news_neutral_probability",
            "ensemble_prediction",
            "ensemble_up_probability",
            "ensemble_down_probability",
            "ensemble_neutral_probability",
            "news_article_count",
        ]
    ]
    aligned_output.to_csv(PREDICTIONS_PATH, index=False)

    report = {
        "model": "HDFCBANK XGBoost + News event ensemble",
        "model_1": {
            "name": "HDFCBANK XGBoost",
            "artifact": str(MODEL1_PATH),
            "metrics": model1_metrics,
        },
        "model_2": {
            "name": "HDFCBANK News Model",
            "artifact": str(MODEL2_PATH),
            "label_mapping": {"POSITIVE": "UP", "NEGATIVE": "DOWN", "NEUTRAL": "NEUTRAL"},
            "metrics": model2_metrics,
        },
        "ensemble": {
            "algorithm": "Weighted probability ensemble",
            "xgboost_weight": XGB_WEIGHT,
            "news_weight": NEWS_WEIGHT,
            "metrics": ensemble_metrics,
        },
        "data": {
            "market_path": str(MARKET_PATH),
            "news_path": str(NEWS_PATH),
            "market_rows": int(len(market)),
            "news_rows": int(len(news)),
            "common_dates": int(len(aligned)),
            "first_common_date": str(aligned["Date"].min().date()),
            "last_common_date": str(aligned["Date"].max().date()),
        },
        "outputs": {
            "predictions": str(PREDICTIONS_PATH),
            "report": str(REPORT_PATH),
        },
        "random_state": RANDOM_STATE,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n" + "=" * 72)
    print("ENSEMBLE ARTIFACTS")
    print("=" * 72)
    print(f"\nPredictions: {PREDICTIONS_PATH}")
    print(f"Report: {REPORT_PATH}")
    print("\nEnsemble completed successfully.")
    return report


if __name__ == "__main__":
    train_ensemble()
