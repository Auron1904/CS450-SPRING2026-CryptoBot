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

    # original indicators
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "SMA_20",
    "SMA_50",
    "EMA_9",
    "ATR_14",

    # new features
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

    # target
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




def add_target_labels(
    df: pd.DataFrame,
    horizon: int = 7,
    threshold: float = 0.0001,
) -> pd.DataFrame:
    """
    Add a binary target column based on future return over a multi-day horizon.

    Target = 1 if the future return over `horizon` days is greater than
    `threshold`, else 0.

    Small or negative moves are labeled 0 here, which keeps this as a binary
    classification problem.

    Example:
      horizon=3, threshold=0.005 means:
      label as 1 only if price is up more than 0.5% after 3 days.
    """
    labeled_df = df.copy()

    if "close" not in labeled_df.columns:
        raise ValueError("Missing required OHLCV column: close")

    labeled_df["close"] = pd.to_numeric(labeled_df["close"], errors="coerce")

    future_close = labeled_df["close"].shift(-horizon)
    future_return = (future_close - labeled_df["close"]) / labeled_df["close"]

    labeled_df["Target"] = pd.Series(pd.NA, index=labeled_df.index, dtype="Int64")

    valid_rows = labeled_df["close"].notna() & future_close.notna()
    labeled_df.loc[valid_rows, "Target"] = (
        future_return[valid_rows] > threshold
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
        # new features
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
    feature_df = feature_df.dropna(subset=feature_columns_without_target).reset_index(drop=True)

    return feature_df[FEATURE_DATA_COLUMNS].copy()

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add standard technical indicators plus richer engineered features
    to an OHLCV dataframe.
    """
    enriched_df = df.copy()
    ta = _get_pandas_ta()

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = sorted(required_columns.difference(enriched_df.columns))
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    for column in ["open", "high", "low", "close", "volume"]:
        enriched_df[column] = pd.to_numeric(enriched_df[column], errors="coerce")

    # Existing indicators
    enriched_df["RSI_14"] = ta.rsi(enriched_df["close"], length=14)

    macd_df = ta.macd(enriched_df["close"], fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        enriched_df["MACD"] = pd.NA
        enriched_df["MACD_signal"] = pd.NA
        enriched_df["MACD_hist"] = pd.NA
    else:
        enriched_df["MACD"] = macd_df.iloc[:, 0]
        enriched_df["MACD_signal"] = macd_df.iloc[:, 1]
        enriched_df["MACD_hist"] = macd_df.iloc[:, 2]

    enriched_df["SMA_20"] = ta.sma(enriched_df["close"], length=20)
    enriched_df["SMA_50"] = ta.sma(enriched_df["close"], length=50)
    enriched_df["EMA_9"] = ta.ema(enriched_df["close"], length=9)

    enriched_df["ATR_14"] = ta.atr(
        high=enriched_df["high"],
        low=enriched_df["low"],
        close=enriched_df["close"],
        length=14,
    )

    # New return features
    enriched_df["ret_1"] = enriched_df["close"].pct_change(1)
    enriched_df["ret_3"] = enriched_df["close"].pct_change(3)
    enriched_df["ret_7"] = enriched_df["close"].pct_change(7)

    # Trend relationship features
    enriched_df["close_to_sma20"] = (
        (enriched_df["close"] - enriched_df["SMA_20"]) / enriched_df["SMA_20"]
    )
    enriched_df["close_to_sma50"] = (
        (enriched_df["close"] - enriched_df["SMA_50"]) / enriched_df["SMA_50"]
    )
    enriched_df["sma20_minus_sma50"] = (
        (enriched_df["SMA_20"] - enriched_df["SMA_50"]) / enriched_df["SMA_50"]
    )
    enriched_df["ema9_minus_sma20"] = (
        (enriched_df["EMA_9"] - enriched_df["SMA_20"]) / enriched_df["SMA_20"]
    )
    enriched_df["macd_cross_gap"] = (
        enriched_df["MACD"] - enriched_df["MACD_signal"]
    )

    # Volatility features
    enriched_df["atr_pct"] = enriched_df["ATR_14"] / enriched_df["close"]
    enriched_df["ret_1_std_7"] = enriched_df["ret_1"].rolling(7).std()
    enriched_df["ret_1_std_14"] = enriched_df["ret_1"].rolling(14).std()
    enriched_df["high_low_range_pct"] = (
        (enriched_df["high"] - enriched_df["low"]) / enriched_df["close"]
    )

    # Volume features
    enriched_df["vol_ret_1"] = enriched_df["volume"].pct_change(1)
    enriched_df["vol_sma_20"] = enriched_df["volume"].rolling(20).mean()
    enriched_df["vol_ratio_20"] = enriched_df["volume"] / enriched_df["vol_sma_20"]

    # Rolling range / price position
    enriched_df["rolling_high_20"] = enriched_df["high"].rolling(20).max()
    enriched_df["rolling_low_20"] = enriched_df["low"].rolling(20).min()
    range_denominator = (
         enriched_df["rolling_high_20"] - enriched_df["rolling_low_20"]
    )

    enriched_df["range_pos_20"] = (
        (enriched_df["close"] - enriched_df["rolling_low_20"]) /
        range_denominator.replace(0, pd.NA)
        )   
        

    return enriched_df

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
    print(feature_df["Target"].value_counts(normalize=True))