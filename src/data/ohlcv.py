from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

# Yahoo Finance expects symbols like BTC-USD for crypto pairs.
# We keep the accepted intervals explicit so invalid requests fail fast.
SUPPORTED_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

# Friendly aliases make the API a bit easier to use elsewhere in the project.
INTERVAL_ALIASES = {
    "1min": "1m",
    "2min": "2m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "60min": "60m",
    "hourly": "1h",
    "daily": "1d",
    "weekly": "1wk",
    "monthly": "1mo",
}

# Yahoo limits how far back some intraday intervals can go, so we choose
# safe defaults when the caller does not provide a start/end date.
DEFAULT_PERIOD_BY_INTERVAL = {
    "1m": "7d",
    "2m": "60d",
    "5m": "60d",
    "15m": "60d",
    "30m": "60d",
    "60m": "2y",
    "90m": "60d",
    "1h": "2y",
    "1d": "max",
    "5d": "max",
    "1wk": "max",
    "1mo": "max",
    "3mo": "max",
}

STANDARD_COLUMN_NAMES = {
    "Date": "timestamp",
    "Datetime": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def normalize_interval(timeframe: str) -> str:
    """Return a Yahoo-compatible interval string."""
    normalized = INTERVAL_ALIASES.get(timeframe.strip().lower(), timeframe.strip().lower())
    if normalized not in SUPPORTED_INTERVALS:
        supported = ", ".join(sorted(SUPPORTED_INTERVALS))
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Supported values: {supported}")
    return normalized


def build_output_path(
    symbol: str,
    timeframe: str,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Create a stable filename for a symbol/timeframe dataset."""
    normalized_timeframe = normalize_interval(timeframe)
    safe_symbol = (
        symbol.strip()
        .replace("/", "-")
        .replace(" ", "_")
        .replace("^", "")
        .replace("=", "")
    )
    safe_timeframe = normalized_timeframe
    return Path(output_dir) / f"{safe_symbol}_{safe_timeframe}.csv"


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    auto_adjust: bool = False,
    prepost: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Yahoo Finance and return a normalized dataframe.

    The returned dataframe always uses the columns:
    `timestamp`, `open`, `high`, `low`, `close`, `volume`, `symbol`, `timeframe`
    """
    interval = normalize_interval(timeframe)

    if period is not None and (start is not None or end is not None):
        raise ValueError("Pass either `period` or `start`/`end`, not both.")

    # If the caller does not specify a date range, choose a default lookback
    # that is valid for the requested interval.
    if period is None and start is None and end is None:
        period = DEFAULT_PERIOD_BY_INTERVAL[interval]

    download_kwargs = {
        "tickers": symbol,
        "interval": interval,
        "auto_adjust": auto_adjust,
        "progress": False,
        "threads": False,
        "prepost": prepost,
        "group_by": "column",
    }

    if period is not None:
        download_kwargs["period"] = period
    else:
        if start is not None:
            download_kwargs["start"] = start
        if end is not None:
            download_kwargs["end"] = end

    df = yf.download(**download_kwargs)
    if df.empty:
        raise ValueError(
            f"No OHLCV data returned for symbol '{symbol}' at timeframe '{interval}'."
        )

    # yfinance can occasionally return a MultiIndex on columns; flattening
    # keeps the rest of the normalization logic simple and predictable.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(column[0]) for column in df.columns.to_flat_index()]

    df = df.reset_index().rename(columns=STANDARD_COLUMN_NAMES)

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Downloaded data is missing required OHLCV columns: {missing}")

    # Keep only the standardized market fields plus a little metadata that
    # helps downstream code trace where the dataset came from.
    df = df[required_columns].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    timestamp_tz = getattr(df["timestamp"].dt, "tz", None)
    if timestamp_tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    for numeric_column in ["open", "high", "low", "close", "volume"]:
        df[numeric_column] = pd.to_numeric(df[numeric_column], errors="coerce")

    df = df.dropna(subset=required_columns).drop_duplicates(subset=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    df["timeframe"] = interval

    return df


def save_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    output_dir: str | Path = "data/raw",
) -> Path:
    """Persist a normalized OHLCV dataframe to CSV."""
    output_path = build_output_path(symbol=symbol, timeframe=timeframe, output_dir=output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def fetch_and_save_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    period: str | None = None,
    output_dir: str | Path = "data/raw",
    auto_adjust: bool = False,
    prepost: bool = False,
) -> tuple[pd.DataFrame, Path]:
    """Convenience wrapper for fetching data and saving it in one call."""
    df = fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        period=period,
        auto_adjust=auto_adjust,
        prepost=prepost,
    )
    output_path = save_ohlcv(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        output_dir=output_dir,
    )
    return df, output_path
if __name__ == "__main__":
    df, output_path = fetch_and_save_ohlcv(
        symbol="BTC-USD",
        timeframe="1d",    )

    print(f"Saved OHLCV data to: {output_path}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
