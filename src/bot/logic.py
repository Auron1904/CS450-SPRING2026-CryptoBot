import os
from statistics import mean
from time import time

import requests
from dotenv import load_dotenv


class MATrendStrategy:
    def __init__(self) -> None:
        load_dotenv()
        self.api_key = os.getenv("CG_API_KEY")
        self.symbol = "bitcoin"
        self.vs_currency = "usd"

    def fetch_last_5_minute_prices(self) -> list[float]:
        url = f"https://api.coingecko.com/api/v3/coins/{self.symbol}/market_chart"
        params = {
            "vs_currency": self.vs_currency,
            "days": "1",
            "interval": "minutely",
        }

        headers = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        prices = payload.get("prices", [])
        if not prices:
            return []

        cutoff_ms = int((time() - 5 * 60) * 1000)
        recent_prices = [point[1] for point in prices if point[0] >= cutoff_ms]

        if len(recent_prices) < 5:
            recent_prices = [point[1] for point in prices[-5:]]

        return [float(price) for price in recent_prices[-5:]]

    def generate_signal(self) -> dict:
        prices = self.fetch_last_5_minute_prices()
        if len(prices) < 2:
            return {
                "signal": "HOLD",
                "current_price": None,
                "ma_5": None,
                "reason": "Not enough recent price points",
            }

        current_price = prices[-1]
        ma_5 = self._calculate_sma(prices)

        if current_price > ma_5:
            signal = "BUY"
        elif current_price < ma_5:
            signal = "SELL"
        else:
            signal = "HOLD"

        return {
            "signal": signal,
            "current_price": current_price,
            "ma_5": ma_5,
            "sample_size": len(prices),
        }

    @staticmethod
    def _calculate_sma(prices: list[float]) -> float:
        return float(mean(prices))
