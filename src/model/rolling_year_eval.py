from __future__ import annotations

from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "SMA_20",
    "SMA_50",
    "EMA_9",
    "ATR_14",
    "ret_1",
    "ret_3",
    "ret_7",
    "close_to_sma20",
    "close_to_sma50",
    "sma20_minus_sma50",
    "ema9_minus_sma20",
    "macd_cross_gap",
    "atr_pct",
    "ret_1_std_7",
    "ret_1_std_14",
    "high_low_range_pct",
    "vol_ret_1",
    "vol_ratio_20",
    "range_pos_20",
]
TARGET_COLUMN = "Target"


def _load_ml_dependencies():
    try:
        from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "scikit-learn is required but is not installed."
        ) from exc

    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError(
            "xgboost could not be imported."
        ) from exc

    return (
        XGBClassifier,
        accuracy_score,
        classification_report,
        precision_score,
        recall_score,
    )


def load_feature_dataset(input_path: str | Path) -> pd.DataFrame:
    """
    Load the processed feature dataset and keep only rows usable for
    train/test evaluation.
    """
    dataset_path = Path(input_path)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found at '{dataset_path}'. "
            "Run the raw-data and indicator pipeline first."
        )

    df = pd.read_csv(dataset_path)

    required_columns = ["timestamp"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {', '.join(missing_columns)}"
        )

    df = df[required_columns].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing timestamp/features/target.
    # This excludes the final unlabeled forecasting row, which is correct for evaluation.
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def train_and_score_one_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Train on one date window and evaluate on the next date window.
    """
    (
        XGBClassifier,
        accuracy_score,
        classification_report,
        precision_score,
        recall_score,
    ) = _load_ml_dependencies()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

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
    prob_up = model.predict_proba(X_test)[:, 1]

    results = {
        "model": model,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "predictions": pd.DataFrame(
            {
                "timestamp": test_df["timestamp"].values,
                "y_true": y_test.values,
                "y_pred": y_pred,
                "prob_up": prob_up,
            }
        ),
    }
    return results


def rolling_year_evaluation(
    df: pd.DataFrame,
    train_years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rolling yearly evaluation.

    Example with train_years=2:
      train 2020-2021 -> test 2022
      train 2021-2022 -> test 2023
      train 2022-2023 -> test 2024
    """
    available_years = sorted(df["timestamp"].dt.year.unique())
    if len(available_years) < train_years + 1:
        raise ValueError(
            f"Need at least {train_years + 1} distinct years of data, "
            f"but found only {len(available_years)}."
        )

    summary_rows = []
    prediction_frames = []

    for i in range(train_years, len(available_years)):
        train_year_block = available_years[i - train_years:i]
        test_year = available_years[i]

        train_df = df[df["timestamp"].dt.year.isin(train_year_block)].copy()
        test_df = df[df["timestamp"].dt.year == test_year].copy()

        if train_df.empty or test_df.empty:
            continue

        results = train_and_score_one_split(train_df, test_df)

        predictions = results["predictions"].copy()
        predictions["train_years"] = ",".join(str(y) for y in train_year_block)
        predictions["test_year"] = test_year
        prediction_frames.append(predictions)

        summary_rows.append(
            {
                "train_start_year": min(train_year_block),
                "train_end_year": max(train_year_block),
                "test_year": test_year,
                "train_rows": len(train_df),
                "test_rows": len(test_df),
                "accuracy": round(results["accuracy"], 4),
                "precision": round(results["precision"], 4),
                "recall": round(results["recall"], 4),
            }
        )

        print(
            f"\n=== Train {min(train_year_block)}-{max(train_year_block)} -> Test {test_year} ==="
        )
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print("\nClassification Report")
        print("---------------------")
        print(results["classification_report"])

    if not summary_rows:
        raise RuntimeError("No rolling yearly splits were produced.")

    summary_df = pd.DataFrame(summary_rows)
    all_predictions_df = pd.concat(prediction_frames, ignore_index=True)

    return summary_df, all_predictions_df


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    input_path = project_root / "data" / "processed" / "btc_features.csv"
    summary_output_path = project_root / "data" / "processed" / "rolling_year_summary.csv"
    predictions_output_path = project_root / "data" / "processed" / "rolling_year_predictions.csv"

    df = load_feature_dataset(input_path)

    # Change this to 1 if you want:
    # train 2022 -> test 2023
    # train 2023 -> test 2024
    #
    # With 2:
    # train 2022-2023 -> test 2024
    train_years = 2

    summary_df, predictions_df = rolling_year_evaluation(df, train_years=train_years)

    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_output_path, index=False)
    predictions_df.to_csv(predictions_output_path, index=False)

    print("\n=== Rolling Year Summary ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved summary to: {summary_output_path}")
    print(f"Saved predictions to: {predictions_output_path}")