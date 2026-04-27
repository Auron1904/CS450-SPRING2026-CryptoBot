"""
Comprehensive test suite for all CryptoBot components.
"""
import os
import sys
from pathlib import Path

import pytest

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoinGeckoIntegration:
    """Test CoinGecko data downloader integration"""

    def test_fetch_and_process_data(self):
        """Test CoinGecko API fetch and data processing"""
        from src.data.downloader import fetch_and_process_data

        df = fetch_and_process_data(days=7)

        assert df is not None, "Failed to fetch data from CoinGecko"
        assert len(df) > 0, "Returned dataframe is empty"
        assert "timestamp" in df.columns, "Missing timestamp column"
        assert "price" in df.columns, "Missing price column"
        assert "volume" in df.columns, "Missing volume column"
        assert "target" in df.columns, "Missing target column"
        assert len(df.columns) > 5, "Missing feature columns"


class TestAlpacaClient:
    """Test Alpaca trading client"""

    def test_alpaca_client_initialization(self):
        """Test AlpacaClient can be initialized"""
        from src.bot.alpaca_client import AlpacaClient

        client = AlpacaClient()
        assert client is not None, "Failed to initialize AlpacaClient"
        assert hasattr(client, 'client'), "AlpacaClient missing trading client"

    def test_get_account_details(self):
        """Test retrieving account details"""
        from src.bot.alpaca_client import AlpacaClient

        client = AlpacaClient()
        account = client.get_account_details()

        assert isinstance(account, dict), "Account details should be a dict"
        assert "cash" in account or "error" in account, "Missing cash or error field"

        if "cash" in account:
            assert "buying_power" in account, "Missing buying_power"
            assert "equity" in account, "Missing equity"

    def test_get_open_positions(self):
        """Test retrieving open positions"""
        from src.bot.alpaca_client import AlpacaClient

        client = AlpacaClient()
        positions = client.get_open_positions()

        assert isinstance(positions, dict), "Positions should be a dict"
        assert "has_btc_position" in positions or "error" in positions

    def test_get_recent_trades(self):
        """Test retrieving recent trades"""
        from src.bot.alpaca_client import AlpacaClient

        client = AlpacaClient()
        trades = client.get_recent_trades(limit=5)

        assert isinstance(trades, (list, dict)), "Trades should be list or dict"

    def test_parse_side(self):
        """Test OrderSide parsing"""
        from src.bot.alpaca_client import AlpacaClient
        from alpaca.trading.enums import OrderSide

        assert AlpacaClient._parse_side("BUY") == OrderSide.BUY
        assert AlpacaClient._parse_side("SELL") == OrderSide.SELL
        assert AlpacaClient._parse_side("HOLD") is None
        assert AlpacaClient._parse_side(OrderSide.BUY) == OrderSide.BUY


class TestMATrendStrategy:
    """Test MA_5 Trend Following strategy"""

    def test_strategy_initialization(self):
        """Test MATrendStrategy initialization"""
        from src.bot.logic import MATrendStrategy

        strategy = MATrendStrategy()
        assert strategy is not None, "Failed to initialize MATrendStrategy"
        assert strategy.symbol == "bitcoin", "Incorrect symbol"
        assert strategy.vs_currency == "usd", "Incorrect currency"

    def test_generate_signal(self):
        """Test signal generation"""
        from src.bot.logic import MATrendStrategy

        strategy = MATrendStrategy()
        signal = strategy.generate_signal()

        assert isinstance(signal, dict), "Signal should be a dict"
        assert "signal" in signal, "Missing signal field"
        assert signal["signal"] in ["BUY", "SELL", "HOLD"], f"Invalid signal: {signal['signal']}"

    def test_calculate_sma(self):
        """Test SMA calculation"""
        from src.bot.logic import MATrendStrategy

        prices = [100.0, 102.0, 101.0, 103.0, 99.0]
        sma = MATrendStrategy._calculate_sma(prices)

        expected = sum(prices) / len(prices)
        assert abs(sma - expected) < 0.01, f"SMA calculation incorrect: {sma} vs {expected}"


class TestDataProcessing:
    """Test data processing pipelines"""

    def test_ohlcv_structure(self):
        """Test OHLCV data structure"""
        from src.data.ohlcv import fetch_and_save_ohlcv
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            df, _ = fetch_and_save_ohlcv(symbol="BTC-USD", timeframe="1d", output_dir=tmpdir)

            if df is not None:
                assert len(df) > 0, "Empty OHLCV dataframe"
                assert "Open" in df.columns or "open" in df.columns, "Missing open price"
                assert "Close" in df.columns or "close" in df.columns, "Missing close price"
                assert "Volume" in df.columns or "volume" in df.columns, "Missing volume"

    def test_indicators_calculation(self):
        """Test technical indicators with proper OHLCV data"""
        from src.features.indicators import build_feature_dataset
        from src.data.ohlcv import fetch_and_save_ohlcv
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            df, _ = fetch_and_save_ohlcv(symbol="BTC-USD", timeframe="1d", output_dir=tmpdir)
            if df is not None and len(df) > 20:
                features = build_feature_dataset(df)

                assert features is not None, "Failed to build features"
                assert len(features) > 0, "Empty features dataframe"
                # Should have technical indicators added
                assert len(features.columns) > 4, "Features should have indicators"


class TestModelIntegration:
    """Test model loading and prediction"""

    def test_model_file_exists(self):
        """Test model file exists and is valid"""
        from pathlib import Path
        import json

        model_path = Path(__file__).parent.parent / "models" / "btc_model.json"

        assert model_path.exists(), f"Model file not found: {model_path}"

        with open(model_path) as f:
            model_data = json.load(f)
            assert model_data is not None, "Model file is empty or invalid JSON"

    def test_predict_function(self):
        """Test prediction function with proper feature data"""
        from src.model.predict import get_latest_prediction
        from src.data.ohlcv import fetch_and_save_ohlcv
        from src.features.indicators import build_feature_dataset
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            df, _ = fetch_and_save_ohlcv(symbol="BTC-USD", timeframe="1d", output_dir=tmpdir)
            if df is not None and len(df) > 20:
                features = build_feature_dataset(df)
                csv_path = Path(tmpdir) / "features.csv"
                features.to_csv(csv_path, index=False)

                model_path = Path(__file__).parent.parent / "models" / "btc_model.json"
                prediction = get_latest_prediction(model_path=model_path, data_path=csv_path)

                assert prediction is not None, "Prediction returned None"
                assert isinstance(prediction, dict), "Prediction should be a dict"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
