import requests
import pandas as pd
import os


def fetch_and_process_data(days=90):
    API_KEY = os.getenv("CG_API_KEY", "your_default_key")  # Use .env for safety
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    headers = {"x-cg-demo-api-key": API_KEY}
    params = {"vs_currency": "usd", "days": days}

    response = requests.get(url, params=params, headers=headers, timeout=30)
    if response.status_code != 200:
        return None

    data = response.json()

    # 2) Build raw dataframe
    # -----------------------------
    prices_df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    volumes_df = pd.DataFrame(data["total_volumes"], columns=["timestamp", "volume"])
    market_caps_df = pd.DataFrame(
        data["market_caps"], columns=["timestamp", "market_cap"]
    )

    df = prices_df.merge(volumes_df, on="timestamp").merge(
        market_caps_df, on="timestamp"
    )

    # -----------------------------
    # 3) Clean dataframe
    # -----------------------------
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # 4) Build ML features
    # -----------------------------

    # Price returns
    df["return_1"] = df["price"].pct_change(1)
    df["return_5"] = df["price"].pct_change(5)
    df["return_10"] = df["price"].pct_change(10)

    # Moving averages
    df["ma_5"] = df["price"].rolling(window=5).mean()
    df["ma_10"] = df["price"].rolling(window=10).mean()

    # Moving average relationship
    df["ma_ratio"] = df["ma_5"] / df["ma_10"]

    # Rolling volatility
    df["volatility_5"] = df["return_1"].rolling(window=5).std()

    # Volume features
    df["volume_change"] = df["volume"].pct_change()
    df["volume_ma_5"] = df["volume"].rolling(window=5).mean()

    # Optional: market cap change
    df["market_cap_change"] = df["market_cap"].pct_change()

    # -----------------------------
    # 5) Optional target column
    # -----------------------------
    # 1 if next price is higher, else 0
    df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)

    # -----------------------------
    # 6) Remove rows with missing values
    # -----------------------------
    df = df.dropna().reset_index(drop=True)
    return df
