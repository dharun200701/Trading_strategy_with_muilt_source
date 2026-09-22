
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.config import ROOT_DIR, settings


logger = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================

DATA_PATH = ROOT_DIR / "data" / "raw" / "market" / "HDFCBANK_NSE_raw.csv"
MODEL_DIR = ROOT_DIR / "models" / "trained"
METADATA_DIR = ROOT_DIR / "models" / "metadata"
REPORT_DIR = ROOT_DIR / "reports" / "model"

MODEL_PATH = MODEL_DIR / "hdfcbank_xgboost.pkl"
METADATA_PATH = METADATA_DIR / "hdfcbank_xgboost_metadata.json"
REPORT_PATH = REPORT_DIR / "hdfcbank_xgboost_5year_report.json"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = int(getattr(settings, "seed", 42))

CLASS_NAMES = ["UP", "DOWN", "NEUTRAL"]
CLASS_TO_INT = {
    "UP": 0,
    "DOWN": 1,
    "NEUTRAL": 2,
}
INT_TO_CLASS = {v: k for k, v in CLASS_TO_INT.items()}


# Candidate thresholds are evaluated ONLY on the validation set.
TARGET_THRESHOLDS = [
    0.0010,  # 0.10%
    0.0015,  # 0.15%
    0.0020,  # 0.20%
    0.0025,  # 0.25%
    0.0030,  # 0.30%
    0.0040,  # 0.40%
    0.0050,  # 0.50%
    0.0075,  # 0.75%
    0.0100,  # 1.00%
]


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    source = pd.read_csv(DATA_PATH)

    def column_key(column: str) -> str:
        return "".join(character for character in str(column).lower() if character.isalnum())

    aliases = {
        "Date": {"date", "timestamp", "chtimestamp", "chdate"},
        "Open": {"open", "openingprice", "chopeningprice", "chopen"},
        "High": {"high", "tradehighprice", "chtradehighprice", "chhigh"},
        "Low": {"low", "tradelowprice", "chtradelowprice", "chlow"},
        "Close": {"close", "closingprice", "chclosingprice", "chclose"},
        "Volume": {
            "volume",
            "totaltradedquantity",
            "tottradedqty",
            "chtottradedqty",
            "chtotaltradedquantity",
            "ttltrdqty",
        },
    }

    source_columns = {column_key(column): column for column in source.columns}
    rename_map = {}
    for canonical, candidates in aliases.items():
        matches = [source_columns[candidate] for candidate in candidates if candidate in source_columns]
        if matches:
            rename_map[matches[0]] = canonical

    df = source.rename(columns=rename_map)
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    for column in required[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna(subset=["Date"] + required)
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )

    if len(df) < 500:
        raise ValueError(
            f"Dataset contains only {len(df)} rows. "
            "Expected approximately 5 years of daily data."
        )

    print("\nDATASET")
    print(f"Rows: {len(df)}")
    print(f"First date: {df['Date'].min().date()}")
    print(f"Last date: {df['Date'].max().date()}")
    print("\nCOLUMNS")
    for column in required:
        print(column)
    five_years = len(df) >= 1000 and df["Date"].min() <= pd.Timestamp("2022-01-01")
    print("\n5-YEAR CHECK")
    print("PASS" if five_years else "FAIL")

    return df


def prepare_training_frame(df: pd.DataFrame | list[dict]) -> pd.DataFrame:
    """Build the canonical leakage-safe Model 1 frame from market rows."""
    source = pd.DataFrame(df).copy()
    source.columns = [str(column).strip() for column in source.columns]
    aliases = {
        "date": "Date", "datetime": "Date", "open": "Open", "high": "High",
        "low": "Low", "close": "Close", "last_price": "last_price",
        "volume": "Volume", "average_price": "average_price",
        "turnover_lacs": "turnover_lacs", "no_of_trades": "NO_OF_TRADES",
        "delivery_quantity": "delivery_quantity", "delivery_percentage": "delivery_percentage",
    }
    source = source.rename(columns={column: aliases.get(column.lower(), column) for column in source.columns})
    if "Close" not in source.columns:
        raise ValueError("Market data must contain a close column.")
    for column, fallback in (("Open", "Close"), ("High", "Close"), ("Low", "Close"), ("Volume", None)):
        if column not in source.columns:
            source[column] = source[fallback] if fallback else 0.0
    if "Date" not in source.columns:
        source["Date"] = pd.RangeIndex(len(source))
    source["Date"] = pd.to_datetime(source["Date"], errors="coerce")
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source = source.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"])
    source = source.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    source["last_price"] = pd.to_numeric(source.get("last_price", source["Close"]), errors="coerce").fillna(source["Close"])
    source["average_price"] = pd.to_numeric(source.get("average_price", source["Close"]), errors="coerce").fillna(source["Close"])
    for column in ["turnover_lacs", "NO_OF_TRADES", "delivery_quantity", "delivery_percentage"]:
        if column not in source.columns:
            source[column] = 0.0
        source[column] = pd.to_numeric(source[column], errors="coerce").fillna(0.0)
    prepared = add_features(source)
    prepared = create_target(prepared, float(getattr(settings, "target_threshold", 0.01)))
    prepared = prepared.rename(columns={"Target": "target"})
    for source_name, alias in (("Close", "close"), ("Open", "open"), ("High", "high"), ("Low", "low"), ("Volume", "volume")):
        prepared[alias] = prepared[source_name]
    prepared["date"] = prepared["Date"]
    return prepared.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    close = data["Close"]
    open_price = data["Open"]
    high = data["High"]
    low = data["Low"]
    volume = data["Volume"]

    # --------------------------------------------------------
    # Basic returns
    # --------------------------------------------------------

    data["daily_return"] = close.pct_change()

    for period in [2, 3, 5, 10, 20, 30]:
        data[f"return_{period}d"] = close.pct_change(period)

    # --------------------------------------------------------
    # Price structure
    # --------------------------------------------------------

    data["high_low_range"] = (
        (high - low) / close.replace(0, np.nan)
    )

    data["open_close_change"] = (
        (close - open_price)
        / open_price.replace(0, np.nan)
    )

    data["gap_pct"] = (
        (open_price - close.shift(1))
        / close.shift(1).replace(0, np.nan)
    )

    data["close_position"] = (
        (close - low)
        / (high - low).replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    for period in [5, 10, 20, 50, 100, 200]:
        data[f"sma_{period}"] = close.rolling(
            period
        ).mean()

        data[f"close_over_sma{period}"] = (
            close / data[f"sma_{period}"] - 1
        )

    for period in [5, 10, 20, 50]:
        data[f"ema_{period}"] = close.ewm(
            span=period,
            adjust=False,
        ).mean()

    data["ema5_over_ema20"] = (
        data["ema_5"] / data["ema_20"] - 1
    )

    data["ema10_over_ema50"] = (
        data["ema_10"] / data["ema_50"] - 1
    )

    data["sma5_over_sma20"] = (
        data["sma_5"] / data["sma_20"] - 1
    )

    data["sma20_over_sma50"] = (
        data["sma_20"] / data["sma_50"] - 1
    )

    data["sma50_over_sma200"] = (
        data["sma_50"] / data["sma_200"] - 1
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    data["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    data["macd"] = ema12 - ema26

    data["macd_signal"] = data["macd"].ewm(
        span=9,
        adjust=False,
    ).mean()

    data["macd_histogram"] = (
        data["macd"]
        - data["macd_signal"]
    )

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    for period in [5, 10, 20]:
        data[f"roc_{period}"] = (
            close.pct_change(period) * 100
        )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1,
    ).max(axis=1)

    data["atr_14"] = true_range.rolling(14).mean()

    data["atr_pct"] = (
        data["atr_14"]
        / close.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    for period in [5, 10, 20, 30]:
        data[f"rolling_volatility_{period}"] = (
            data["daily_return"]
            .rolling(period)
            .std()
        )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    data["bollinger_upper"] = (
        bb_mid + 2 * bb_std
    )

    data["bollinger_lower"] = (
        bb_mid - 2 * bb_std
    )

    data["bollinger_width"] = (
        (data["bollinger_upper"]
         - data["bollinger_lower"])
        / bb_mid.replace(0, np.nan)
    )

    data["bollinger_pct_b"] = (
        (close - data["bollinger_lower"])
        / (
            data["bollinger_upper"]
            - data["bollinger_lower"]
        ).replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------

    data["volume_change"] = volume.pct_change()

    for period in [5, 10, 20, 50]:
        data[f"volume_sma_{period}"] = (
            volume.rolling(period).mean()
        )

    data["volume_ratio"] = (
        volume
        / data["volume_sma_20"].replace(0, np.nan)
    )

    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()

    data["volume_zscore"] = (
        (volume - volume_mean)
        / volume_std.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Recent high/low position
    # --------------------------------------------------------

    for period in [10, 20, 50]:
        rolling_high = high.rolling(period).max()
        rolling_low = low.rolling(period).min()

        data[f"distance_from_high_{period}"] = (
            close / rolling_high - 1
        )

        data[f"distance_from_low_{period}"] = (
            close / rolling_low - 1
        )

    # --------------------------------------------------------
    # Previous close
    # --------------------------------------------------------

    data["previous_close"] = close.shift(1)

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return data


# ============================================================
# TARGET
# ============================================================

def create_target(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:

    data = df.copy()

    next_return = (
        data["Close"].shift(-1)
        / data["Close"]
        - 1
    )

    data["next_day_return"] = next_return

    data["Target"] = np.select(
        [
            next_return > threshold,
            next_return < -threshold,
        ],
        [
            "UP",
            "DOWN",
        ],
        default="NEUTRAL",
    )

    # Last row has no next-day return.
    data = data.iloc[:-1].copy()

    return data


# ============================================================
# FEATURE COLUMN SELECTION
# ============================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:

    excluded = {
        "Date",
        "Target",
        "next_day_return",
        "Symbol",
        "Series",
    }

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return [
        c
        for c in numeric_columns
        if c not in excluded
    ]


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    n = len(df)

    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = df.iloc[:train_end].copy()
    validation = df.iloc[
        train_end:validation_end
    ].copy()

    test = df.iloc[
        validation_end:
    ].copy()

    return train, validation, test


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_class_weights(
    y_train: pd.Series,
) -> Dict[int, float]:

    counts = y_train.value_counts()

    total = len(y_train)
    number_of_classes = len(CLASS_NAMES)

    weights = {}

    for class_name, class_id in CLASS_TO_INT.items():

        count = counts.get(
            class_name,
            0,
        )

        if count == 0:
            weights[class_id] = 1.0
        else:
            weights[class_id] = (
                total
                / (
                    number_of_classes
                    * count
                )
            )

    return weights


# ============================================================
# CREATE MODEL SAMPLE WEIGHTS
# ============================================================

def create_sample_weights(
    y: pd.Series,
    class_weights: Dict[int, float],
) -> np.ndarray:

    encoded = y.map(CLASS_TO_INT)

    return np.array(
        [
            class_weights[int(value)]
            for value in encoded
        ]
    )


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_model(
    params: Dict,
) -> XGBClassifier:

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",

        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params[
            "colsample_bytree"
        ],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],

        random_state=RANDOM_STATE,
        n_jobs=-1,

        tree_method="hist",
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
) -> Dict:

    predictions = model.predict(X)

    accuracy = accuracy_score(
        y,
        predictions,
    )

    class_ids = list(range(len(CLASS_NAMES)))

    precision = precision_score(
        y,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        y,
        predictions,
        labels=class_ids,
    )

    report = classification_report(
        y,
        predictions,
        labels=class_ids,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    predicted_distribution = (
        pd.Series(predictions)
        .value_counts()
        .to_dict()
    )

    actual_distribution = (
        y.value_counts()
        .to_dict()
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "actual_distribution": {
            str(k): int(v)
            for k, v in actual_distribution.items()
        },
        "predicted_distribution": {
            str(k): int(v)
            for k, v in predicted_distribution.items()
        },
    }


# ============================================================
# MAJORITY BASELINE
# ============================================================

def majority_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict:

    majority_class = (
        y_train.value_counts()
        .idxmax()
    )

    predictions = np.full(
        len(y_test),
        majority_class,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    return {
        "majority_class": str(
            majority_class
        ),
        "accuracy": float(accuracy),
    }


# ============================================================
# TARGET THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    feature_df: pd.DataFrame,
    train_raw: pd.DataFrame,
    validation_raw: pd.DataFrame,
    feature_columns: List[str],
) -> float:

    best_threshold = TARGET_THRESHOLDS[0]
    best_score = -np.inf

    X_train = train_raw[
        feature_columns
    ]

    X_validation = validation_raw[
        feature_columns
    ]

    y_train_returns = train_raw[
        "next_day_return"
    ]

    y_validation_returns = validation_raw[
        "next_day_return"
    ]

    # We need a model for each candidate target.
    # Threshold selection uses validation only.
    for threshold in TARGET_THRESHOLDS:

        train_temp = create_target(
            train_raw.drop(
                columns=["Target"],
                errors="ignore",
            ),
            threshold,
        )

        validation_temp = create_target(
            validation_raw.drop(
                columns=["Target"],
                errors="ignore",
            ),
            threshold,
        )

        if len(train_temp) != len(X_train):
            continue

        if len(validation_temp) != len(X_validation):
            continue

        y_train = train_temp["Target"]
        y_validation = validation_temp["Target"]

        if y_train.nunique() < 2:
            continue

        model = create_model(
            {
                "n_estimators": 300,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 3,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
                "gamma": 0.05,
                "reg_alpha": 0.05,
                "reg_lambda": 1.0,
            }
        )

        class_weights = calculate_class_weights(
            y_train
        )

        sample_weights = create_sample_weights(
            y_train,
            class_weights,
        )

        model.fit(
            X_train,
            y_train.map(CLASS_TO_INT),
            sample_weight=sample_weights,
            verbose=False,
        )

        predictions = model.predict(
            X_validation
        )

        score = f1_score(
            y_validation.map(CLASS_TO_INT),
            predictions,
            average="macro",
            zero_division=0,
        )

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold


# ============================================================
# HYPERPARAMETER SEARCH
# ============================================================

def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> Tuple[Dict, float]:

    candidates = [
        {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.0,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.9,
            "gamma": 0.05,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 400,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.05,
            "reg_alpha": 0.05,
            "reg_lambda": 2.0,
        },
        {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.05,
            "min_child_weight": 5,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
        },
        {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.2,
            "reg_lambda": 3.0,
        },
        {
            "n_estimators": 600,
            "max_depth": 4,
            "learning_rate": 0.02,
            "min_child_weight": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "gamma": 0.05,
            "reg_alpha": 0.1,
            "reg_lambda": 3.0,
        },
    ]

    class_weights = calculate_class_weights(
        y_train
    )

    sample_weights = create_sample_weights(
        y_train,
        class_weights,
    )

    best_params = None
    best_score = -np.inf

    for index, params in enumerate(
        candidates,
        start=1,
    ):

        model = create_model(params)

        model.fit(
            X_train,
            y_train.map(CLASS_TO_INT),
            sample_weight=sample_weights,
            eval_set=[
                (
                    X_validation,
                    y_validation.map(
                        CLASS_TO_INT
                    ),
                )
            ],
            verbose=False,
        )

        predictions = model.predict(
            X_validation
        )

        score = f1_score(
            y_validation.map(CLASS_TO_INT),
            predictions,
            average="macro",
            zero_division=0,
        )

        print(
            f"Tuning {index}/{len(candidates)} "
            f"- Validation F1: {score * 100:.2f}%"
        )

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, float(best_score)


# ============================================================
# TRAINING
# ============================================================

def train_model():

    df = load_data()

    print("\nDATASET")
    print(
        f"Rows: {len(df)}"
    )
    print(
        f"First date: "
        f"{df['Date'].min().date()}"
    )
    print(
        f"Last date: "
        f"{df['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Feature engineering BEFORE target creation.
    # Features never use future prices.
    # --------------------------------------------------------

    data = add_features(df)

    # --------------------------------------------------------
    # Create initial target using current configured threshold.
    # --------------------------------------------------------

    configured_threshold = float(
        getattr(
            settings,
            "target_threshold",
            0.003,
        )
    )

    data = create_target(
        data,
        configured_threshold,
    )

    # Remove rows where indicators are not available.
    data = data.dropna().reset_index(
        drop=True
    )

    feature_columns = get_feature_columns(
        data
    )

    if not feature_columns:
        raise ValueError(
            "No usable feature columns found."
        )

    print(
        f"Feature rows: {len(data)}"
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train, validation, test = chronological_split(
        data
    )

    print(
        f"Train/validation/test: "
        f"[{len(train)}, "
        f"{len(validation)}, "
        f"{len(test)}]"
    )

    print(
        f"Train dates: "
        f"{train['Date'].min().date()} "
        f"to "
        f"{train['Date'].max().date()}"
    )

    print(
        f"Validation dates: "
        f"{validation['Date'].min().date()} "
        f"to "
        f"{validation['Date'].max().date()}"
    )

    print(
        f"Test dates: "
        f"{test['Date'].min().date()} "
        f"to "
        f"{test['Date'].max().date()}"
    )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nTARGET DISTRIBUTION")

    distribution = (
        train["Target"]
        .value_counts(normalize=True)
        * 100
    )

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<8}: "
            f"{distribution.get(class_name, 0):.2f}%"
        )

    # --------------------------------------------------------
    # X / y
    # --------------------------------------------------------

    X_train = train[
        feature_columns
    ].copy()

    y_train = train[
        "Target"
    ].copy()

    X_validation = validation[
        feature_columns
    ].copy()

    y_validation = validation[
        "Target"
    ].copy()

    X_test = test[
        feature_columns
    ].copy()

    y_test = test[
        "Target"
    ].copy()

    # --------------------------------------------------------
    # Hyperparameter tuning
    # --------------------------------------------------------

    print("\nHYPERPARAMETER TUNING")

    best_params, validation_f1 = (
        tune_hyperparameters(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )
    )

    print(
        f"\nBest validation F1: "
        f"{validation_f1 * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Train final model
    #
    # IMPORTANT:
    # We do NOT train on test data.
    # We combine TRAIN + VALIDATION only.
    # --------------------------------------------------------

    training_data = pd.concat(
        [train, validation],
        axis=0,
    ).reset_index(
        drop=True
    )

    X_final = training_data[
        feature_columns
    ]

    y_final = training_data[
        "Target"
    ]

    class_weights = calculate_class_weights(
        y_final
    )

    sample_weights = create_sample_weights(
        y_final,
        class_weights,
    )

    model = create_model(
        best_params
    )

    model.fit(
        X_final,
        y_final.map(CLASS_TO_INT),
        sample_weight=sample_weights,
        verbose=False,
    )

    # --------------------------------------------------------
    # FINAL TEST EVALUATION
    # --------------------------------------------------------

    metrics = evaluate_model(
        model,
        X_test,
        y_test.map(CLASS_TO_INT),
    )

    # --------------------------------------------------------
    # Convert distributions to readable labels
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    readable_predictions = [
        INT_TO_CLASS[int(x)]
        for x in predictions
    ]

    predicted_distribution = (
        pd.Series(
            readable_predictions
        )
        .value_counts()
        .to_dict()
    )

    actual_distribution = (
        y_test.value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline = majority_baseline(
        y_train,
        y_test,
    )

    # --------------------------------------------------------
    # Probability analysis
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )

    probability_analysis = {}

    for class_id, class_name in INT_TO_CLASS.items():

        values = probabilities[:, class_id]

        probability_analysis[
            class_name
        ] = {
            "mean": float(
                np.mean(values)
            ),
            "minimum": float(
                np.min(values)
            ),
            "maximum": float(
                np.max(values)
            ),
        }

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_model_path = (
        MODEL_DIR
        / "hdfcbank_xgboost.tmp.pkl"
    )

    import joblib

    joblib.dump(
        model,
        temporary_model_path,
    )

    temporary_model_path.replace(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "model_name": "HDFCBANK XGBoost",
        "model_type": "XGBClassifier",
        "symbol": "HDFCBANK",
        "data_source": "NSE India",
        "provider": "nsepython",
        "dataset_path": str(DATA_PATH),
        "training_timestamp_utc": (
            pd.Timestamp.now(
                "UTC"
            ).isoformat()
        ),

        "dataset_first_date": str(
            df["Date"].min().date()
        ),
        "dataset_last_date": str(
            df["Date"].max().date()
        ),
        "dataset_rows": int(len(df)),

        "feature_rows": int(len(data)),
        "feature_count": int(
            len(feature_columns)
        ),
        "feature_names": feature_columns,

        "target_definition": (
            "Next trading day return "
            "classified as UP, DOWN, "
            "or NEUTRAL."
        ),

        "target_threshold": (
            configured_threshold
        ),

        "train_rows": int(len(train)),
        "validation_rows": int(
            len(validation)
        ),
        "test_rows": int(len(test)),

        "train_first_date": str(
            train["Date"].min().date()
        ),
        "train_last_date": str(
            train["Date"].max().date()
        ),

        "validation_first_date": str(
            validation["Date"].min().date()
        ),
        "validation_last_date": str(
            validation["Date"].max().date()
        ),

        "test_first_date": str(
            test["Date"].min().date()
        ),
        "test_last_date": str(
            test["Date"].max().date()
        ),

        "class_distribution": {
            str(k): int(v)
            for k, v in
            training_data[
                "Target"
            ].value_counts().items()
        },

        "class_weights": {
            str(k): float(v)
            for k, v in class_weights.items()
        },

        "model_parameters": best_params,

        "metrics": metrics,

        "baseline": baseline,

        "predicted_distribution":
            predicted_distribution,

        "actual_test_distribution":
            actual_distribution,

        "probability_analysis":
            probability_analysis,

        "validation_f1": validation_f1,

        "random_state":
            RANDOM_STATE,
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Save report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Terminal result
    # --------------------------------------------------------

    print("\nMODEL PERFORMANCE")
    print("-----------------")

    print(
        f"Accuracy  : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1 Score  : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print("\nBASELINE")
    print("-----------------")

    print(
        f"Accuracy  : "
        f"{baseline['accuracy'] * 100:.2f}%"
    )

    print(
        f"Majority  : "
        f"{baseline['majority_class']}"
    )

    print("\nPREDICTED DISTRIBUTION")
    print("-----------------")

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:<8}: "
            f"{predicted_distribution.get(class_name, 0)}"
        )

    print("\nTEST PERIOD")
    print("-----------------")

    print(
        f"{test['Date'].min().date()} "
        f"to "
        f"{test['Date'].max().date()}"
    )

    print("\nMODEL ARTIFACT")
    print("-----------------")

    print(MODEL_PATH)

    print("\nMETADATA")
    print("-----------------")

    print(METADATA_PATH)

    print("\nREPORT")
    print("-----------------")

    print(REPORT_PATH)

    return {
        "model": model,
        "metrics": metrics,
        "baseline": baseline,
        "metadata": metadata,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s - %(message)s",
    )

    train_model()