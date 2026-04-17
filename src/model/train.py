from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "SMA_20",
    "SMA_50",
    "EMA_9",
    "ATR_14",
]
TARGET_COLUMN = "Target"


def load_training_data(input_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the engineered BTC feature dataset and return cleaned features/labels.
    """
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at '{dataset_path}'. "
            "Run src/features/indicators.py first to generate btc_features.csv."
        )

    df = pd.read_csv(dataset_path)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Training dataset is missing required columns: {missing}")

    model_df = df[required_columns].copy()
    model_df = model_df.apply(pd.to_numeric, errors="coerce")
    model_df = model_df.dropna().reset_index(drop=True)

    if model_df.empty:
        raise ValueError("No usable rows remain after dropping NaN values.")

    X = model_df[FEATURE_COLUMNS]
    y = model_df[TARGET_COLUMN].astype(int)
    return X, y


def _load_ml_dependencies():
    """
    Import runtime ML dependencies with clearer failure messages.
    """
    try:
        from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
        from sklearn.model_selection import TimeSeriesSplit
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required for Step 4 but is not installed in the environment. "
            "Run `uv sync` after adding the dependency."
        ) from exc

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "xgboost could not be imported. On macOS this usually means the OpenMP "
            "runtime is missing. Install `libomp` and then retry."
        ) from exc

    return (
        TimeSeriesSplit,
        XGBClassifier,
        accuracy_score,
        classification_report,
        precision_score,
        recall_score,
    )


def train_xgb_with_timeseries_cv(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
):
    """
    Train an XGBClassifier using chronological TimeSeriesSplit folds.

    Returns the fitted model from the final fold plus the last-fold evaluation
    artifacts needed for reporting.
    """
    (
        TimeSeriesSplit,
        XGBClassifier,
        accuracy_score,
        classification_report,
        precision_score,
        recall_score,
    ) = _load_ml_dependencies()

    if len(X) <= n_splits:
        raise ValueError(
            f"Need more than {n_splits} rows for TimeSeriesSplit, but got {len(X)} rows."
        )

    splitter = TimeSeriesSplit(n_splits=n_splits)
    last_fold_details = None

    for fold_number, (train_index, test_index) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        last_fold_details = {
            "fold_number": fold_number,
            "model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "classification_report": classification_report(
                y_test,
                y_pred,
                zero_division=0,
            ),
        }

    if last_fold_details is None:
        raise RuntimeError("TimeSeriesSplit did not produce any folds.")

    return last_fold_details


def build_feature_importance_table(model) -> pd.DataFrame:
    """
    Return a sorted feature-importance table for the trained XGBoost model.
    """
    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_,
        }
    )
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)


def save_model(model, output_path: str | Path) -> Path:
    """
    Save the trained XGBoost model to disk as JSON.
    """
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(destination)
    return destination


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "btc_features.csv"
    output_path = project_root / "models" / "btc_model.json"

    X, y = load_training_data(input_path)
    results = train_xgb_with_timeseries_cv(X, y, n_splits=5)
    feature_importance_df = build_feature_importance_table(results["model"])
    saved_model_path = save_model(results["model"], output_path)

    print(f"Rows used for training: {len(X)}")
    print(f"Feature columns: {', '.join(FEATURE_COLUMNS)}")
    print(f"Evaluated on final TimeSeriesSplit fold: {results['fold_number']}")

    print("\nLast Fold Metrics")
    print("-----------------")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")

    print("\nClassification Report")
    print("---------------------")
    print(results["classification_report"])

    print("Feature Importance")
    print("------------------")
    print(feature_importance_df.to_string(index=False))

    print(f"\nSaved trained model to: {saved_model_path}")
