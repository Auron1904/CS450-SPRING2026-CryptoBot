from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from src.model.train import FEATURE_COLUMNS
except ModuleNotFoundError:
    from train import FEATURE_COLUMNS


def load_latest_feature_row(input_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed BTC feature dataset and return the newest scorable row.

    This is designed for forward-looking inference, so it does not require the
    row to have a Target label. The latest row may be today's market data with
    Target still missing because tomorrow has not happened yet.
    """
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Processed feature dataset not found at '{dataset_path}'. "
            "Run src/features/indicators.py first."
        )

    df = pd.read_csv(dataset_path)

    required_columns = ["timestamp"] + FEATURE_COLUMNS
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Prediction dataset is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_frame = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    scorable_mask = feature_frame.notna().all(axis=1)
    if not scorable_mask.any():
        raise ValueError(
            "No scorable feature rows were found in the processed dataset."
        )

    latest_index = feature_frame[scorable_mask].index[-1]
    latest_row = df.iloc[latest_index].copy()
    latest_features = feature_frame.iloc[[latest_index]]

    return latest_features, latest_row


def _load_xgboost_classifier():
    """
    Import XGBoost lazily so runtime issues surface with a clear message.
    """
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "xgboost could not be imported. Verify the project environment is synced "
            "and the native XGBoost runtime is available."
        ) from exc

    return XGBClassifier


def load_model(model_path: str | Path):
    """
    Load the trained XGBoost classifier from disk.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{path}'. Run src/model/train.py first."
        )

    XGBClassifier = _load_xgboost_classifier()
    model = XGBClassifier()
    model.load_model(path)
    return model


def predict_next_move(model, latest_features: pd.DataFrame) -> tuple[int, float]:
    """
    Predict whether tomorrow's BTC price will go up or down.
    """
    prediction = int(model.predict(latest_features)[0])

    probability_up = 0.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(latest_features)[0]
        if len(probabilities) > 1:
            probability_up = float(probabilities[1])
        else:
            probability_up = float(probabilities[0])

    return prediction, probability_up


def get_latest_prediction(
    model_path: str | Path,
    data_path: str | Path,
) -> dict:
    """
    Load the latest engineered row, score it with the trained model, and
    return a compact prediction payload for app integration.
    """
    latest_features, latest_row = load_latest_feature_row(data_path)
    model = load_model(model_path)
    prediction, probability_up = predict_next_move(model, latest_features)
    latest_timestamp = pd.to_datetime(latest_row["timestamp"])
    target_timestamp = latest_timestamp + pd.Timedelta(days=1)
    predicted_probability = probability_up if prediction == 1 else 1 - probability_up

    return {
        "timestamp": latest_row["timestamp"],
        "target_timestamp": str(target_timestamp),
        "target_date": target_timestamp.strftime("%b %d, %Y").replace(" 0", " "),
        "prediction": "UP" if prediction == 1 else "DOWN",
        "probability_up": probability_up,
        "probability": predicted_probability,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    model_path = project_root / "models" / "btc_model.json"
    data_path = project_root / "data" / "processed" / "btc_features.csv"

    prediction_payload = get_latest_prediction(model_path=model_path, data_path=data_path)
    latest_timestamp = prediction_payload["timestamp"]
    direction = prediction_payload["prediction"]
    probability_up = prediction_payload["probability_up"]

    print("BTC AI Prediction")
    print("-----------------")
    print(f"Latest feature date: {latest_timestamp}")
    print(f"Model features: {', '.join(FEATURE_COLUMNS)}")

    if direction == "UP":
        print(
            f"\nBullish signal unlocked: the model predicts BTC will go {direction} tomorrow."
        )
    else:
        print(
            f"\nDefensive signal detected: the model predicts BTC will go {direction} tomorrow."
        )

    print(f"Probability of UP move: {probability_up:.2%}")
