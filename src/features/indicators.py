from __future__ import annotations

from pathlib import Path

import pandas as pd

_PANDAS_TA_MODULE = None

FEATURE_DATA_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "timeframe",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "SMA_20",
    "SMA_50",
    "EMA_9",
    "ATR_14",
    "Target",
]


def _get_pandas_ta():
    """
    Import pandas_ta lazily and avoid the numba cache crash seen in this env.

    In the current virtualenv, importing pandas_ta at module load time can fail
    because one of its utility functions is decorated with `@njit(cache=True)`.
    We patch numba's `njit` helper to default `cache=False` before importing the
    library so the indicators module remains executable.
    """
    global _PANDAS_TA_MODULE

    if _PANDAS_TA_MODULE is not None:
        return _PANDAS_TA_MODULE

    import numba

    original_njit = numba.njit

    def safe_njit(*args, **kwargs):
        kwargs.setdefault("cache", False)
        return original_njit(*args, **kwargs)

    numba.njit = safe_njit

    try:
        import pandas_ta as ta
        _PANDAS_TA_MODULE = ta
        return ta
    except Exception as exc:
        raise RuntimeError(
            "Failed to import pandas_ta even after disabling numba cache. "
            "Check the project environment and dependency versions."
        ) from exc
    finally:
        numba.njit = original_njit


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard technical indicators to an OHLCV dataframe.

    Expected input columns:
    `timestamp`, `open`, `high`, `low`, `close`, `volume`
    """
    enriched_df = df.copy()
    ta = _get_pandas_ta()

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = sorted(required_columns.difference(enriched_df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    # Convert market columns to numeric so indicator calculations are reliable
    # even if the CSV was loaded with mixed types.
    for column in ["open", "high", "low", "close", "volume"]:
        enriched_df[column] = pd.to_numeric(enriched_df[column], errors="coerce")

    # Momentum: RSI helps measure whether price has recently moved too far
    # in one direction.
    enriched_df["RSI_14"] = ta.rsi(enriched_df["close"], length=14)

    # Trend: MACD is commonly used to compare short- and long-term momentum.
    macd_df = ta.macd(enriched_df["close"], fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        enriched_df["MACD"] = pd.NA
        enriched_df["MACD_signal"] = pd.NA
        enriched_df["MACD_hist"] = pd.NA
    else:
        enriched_df["MACD"] = macd_df.iloc[:, 0]
        enriched_df["MACD_signal"] = macd_df.iloc[:, 1]
        enriched_df["MACD_hist"] = macd_df.iloc[:, 2]

    # Trend-following moving averages at short and medium windows.
    enriched_df["SMA_20"] = ta.sma(enriched_df["close"], length=20)
    enriched_df["SMA_50"] = ta.sma(enriched_df["close"], length=50)
    enriched_df["EMA_9"] = ta.ema(enriched_df["close"], length=9)

    # Volatility: ATR estimates average movement range using OHLC data.
    enriched_df["ATR_14"] = ta.atr(
        high=enriched_df["high"],
        low=enriched_df["low"],
        close=enriched_df["close"],
        length=14,
    )

    return enriched_df


def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary target column for next-period direction.

    Target = 1 if the next closing price is higher than the current closing
    price, otherwise 0. The final row has no future label and is left empty.
    """
    labeled_df = df.copy()

    if "close" not in labeled_df.columns:
        raise ValueError("Missing required OHLCV column: close")

    labeled_df["close"] = pd.to_numeric(labeled_df["close"], errors="coerce")
    next_close = labeled_df["close"].shift(-1)

    # Use a nullable integer dtype so the final unlabeled row can stay empty
    # until the caller decides whether to drop it.
    labeled_df["Target"] = pd.Series(pd.NA, index=labeled_df.index, dtype="Int64")
    valid_rows = labeled_df["close"].notna() & next_close.notna()
    labeled_df.loc[valid_rows, "Target"] = (
        next_close[valid_rows] > labeled_df.loc[valid_rows, "close"]
    ).astype(int)

    return labeled_df


def build_feature_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature dataset while preserving the newest unlabeled row.

    Rows with incomplete indicator features are removed, but the final row is
    allowed to keep a missing Target because that is exactly the row we want
    to score for the next-day forecast.
    """
    feature_df = add_technical_indicators(df)
    feature_df = add_target_labels(feature_df)

    feature_columns_without_target = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "symbol",
        "timeframe",
        "RSI_14",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "SMA_20",
        "SMA_50",
        "EMA_9",
        "ATR_14",
    ]
    feature_df = feature_df.dropna(subset=feature_columns_without_target).reset_index(
        drop=True
    )

    return feature_df[FEATURE_DATA_COLUMNS].copy()


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    preferred_input_path = project_root / "data" / "raw" / "btc_ohlcv.csv"
    fallback_input_path = project_root / "data" / "raw" / "BTC-USD_1d.csv"

    if preferred_input_path.exists():
        input_path = preferred_input_path
    elif fallback_input_path.exists():
        input_path = fallback_input_path
    else:
        raise FileNotFoundError(
            "Could not find a raw OHLCV file. Expected either "
            f"'{preferred_input_path}' or '{fallback_input_path}'."
        )

    output_path = project_root / "data" / "processed" / "btc_features.csv"

    raw_df = pd.read_csv(input_path)
    feature_df = build_feature_dataset(raw_df)

    columns_to_show = [
        "timestamp",
        "close",
        "RSI_14",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "SMA_20",
        "SMA_50",
        "EMA_9",
        "ATR_14",
        "Target",
    ]

    print("Last 5 rows with technical indicators and target labels:")
    print(feature_df[columns_to_show].tail().to_string(index=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, index=False)
    print(f"\nSaved feature dataset to: {output_path}")
