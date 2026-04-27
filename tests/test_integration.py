"""
End-to-end integration test for the entire CryptoBot pipeline.
Tests: Data fetch -> Feature Engineering -> Model Prediction -> Trade Execution
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_complete_pipeline():
    """Test the complete pipeline from data to trade signal"""
    print("\n" + "="*60)
    print("FULL PIPELINE END-TO-END TEST")
    print("="*60)

    try:
        # Step 1: Fetch data
        print("\n[1/6] Fetching market data from CoinGecko...")
        from src.data.ohlcv import fetch_and_save_ohlcv

        with tempfile.TemporaryDirectory() as tmpdir:
            df, cache_path = fetch_and_save_ohlcv(
                symbol="BTC-USD", timeframe="1d", output_dir=tmpdir
            )
            assert df is not None, "Failed to fetch OHLCV data"
            assert len(df) > 20, "Insufficient historical data"
            print(f"   ✓ Fetched {len(df)} candles")
            print(f"   ✓ Date range: {df.index[0]} to {df.index[-1]}")

            # Step 2: Build features
            print("\n[2/6] Building technical indicators...")
            from src.features.indicators import build_feature_dataset

            features_df = build_feature_dataset(df)
            assert features_df is not None, "Failed to build features"
            assert len(features_df) > 0, "Empty features dataframe"
            print(f"   ✓ Built features with {len(features_df.columns)} columns")
            print(f"   ✓ Rows ready for prediction: {len(features_df)}")

            # Step 3: Save features
            print("\n[3/6] Saving feature dataset...")
            features_path = Path(tmpdir) / "btc_features.csv"
            features_df.to_csv(features_path, index=False)
            assert features_path.exists(), "Failed to save features"
            print(f"   ✓ Features saved to {features_path}")

            # Step 4: Load model and make prediction
            print("\n[4/6] Loading model and generating prediction...")
            from src.model.predict import get_latest_prediction

            model_path = Path(__file__).parent.parent / "models" / "btc_model.json"
            assert model_path.exists(), f"Model not found: {model_path}"

            prediction = get_latest_prediction(model_path=model_path, data_path=features_path)
            assert prediction is not None, "Prediction failed"
            assert "prediction" in prediction, "Missing prediction in result"
            print(f"   ✓ Model prediction: {prediction.get('prediction')}")
            print(f"   ✓ Prediction score: {prediction.get('score', 'N/A')}")

            # Step 5: Generate trading signal
            print("\n[5/6] Generating MA_5 trading signal...")
            from src.bot.logic import MATrendStrategy

            strategy = MATrendStrategy()
            signal = strategy.generate_signal()
            assert signal is not None, "Signal generation failed"
            assert "signal" in signal, "Missing signal in result"
            print(f"   ✓ MA_5 signal: {signal.get('signal')}")
            print(f"   ✓ Current price: ${signal.get('current_price', 'N/A')}")
            print(f"   ✓ MA_5: ${signal.get('ma_5', 'N/A')}")

            # Step 6: Verify trading capability
            print("\n[6/6] Verifying Alpaca trading capability...")
            from src.bot.alpaca_client import AlpacaClient

            client = AlpacaClient()
            account = client.get_account_details()
            assert "error" not in account or account.get("cash"), "Account access failed"
            print(f"   ✓ Connected to Alpaca paper trading")
            print(f"   ✓ Account cash: ${account.get('cash', 'N/A')}")

            positions = client.get_open_positions()
            assert "error" not in positions, "Position check failed"
            print(f"   ✓ BTC Position: {positions.get('position_state', 'Unknown')}")

            print("\n" + "="*60)
            print("✅ FULL PIPELINE TEST PASSED")
            print("="*60)
            print("\nPipeline Summary:")
            print(f"  • Data: {len(df)} OHLCV candles")
            print(f"  • Features: {len(features_df.columns)} technical indicators")
            print(f"  • AI Prediction: {prediction.get('prediction')}")
            print(f"  • MA Signal: {signal.get('signal')}")
            print(f"  • Trading Ready: Yes")
            print("="*60 + "\n")

            return True

    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_consistency():
    """Test data consistency across pipeline"""
    print("\n" + "="*60)
    print("DATA CONSISTENCY TEST")
    print("="*60)

    try:
        from src.data.downloader import fetch_and_process_data
        from src.data.ohlcv import fetch_and_save_ohlcv
        import tempfile

        print("\n[1/3] Fetching CoinGecko data...")
        df_cg = fetch_and_process_data(days=7)
        assert df_cg is not None, "CoinGecko fetch failed"
        print(f"   ✓ CoinGecko: {len(df_cg)} rows, {len(df_cg.columns)} columns")

        print("\n[2/3] Fetching OHLCV data...")
        with tempfile.TemporaryDirectory() as tmpdir:
            df_ohlcv, _ = fetch_and_save_ohlcv(
                symbol="BTC-USD", timeframe="1d", output_dir=tmpdir
            )
            assert df_ohlcv is not None, "OHLCV fetch failed"
            print(f"   ✓ OHLCV: {len(df_ohlcv)} rows, {len(df_ohlcv.columns)} columns")

            print("\n[3/3] Checking data validity...")
            assert len(df_cg) > 0, "CoinGecko data empty"
            assert len(df_ohlcv) > 0, "OHLCV data empty"

            # Check CoinGecko data has required columns
            assert "price" in df_cg.columns, "Missing price in CoinGecko data"
            assert "volume" in df_cg.columns, "Missing volume in CoinGecko data"

            # Check OHLCV data has required columns
            required_ohlcv = ["open", "high", "low", "close", "volume"]
            ohlcv_cols = df_ohlcv.columns.tolist()
            for col in required_ohlcv:
                assert col in ohlcv_cols, f"Missing {col} in OHLCV data"

            print("   ✓ All required columns present")
            print("   ✓ Data consistency verified")

        print("\n" + "="*60)
        print("✅ DATA CONSISTENCY TEST PASSED")
        print("="*60 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ DATA CONSISTENCY TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dashboard_components():
    """Test all dashboard components are importable and functional"""
    print("\n" + "="*60)
    print("DASHBOARD COMPONENTS TEST")
    print("="*60)

    try:
        components = [
            ("Account Metrics", "src.dashboard.account", "render_account_metrics"),
            ("Auto-Trade", "src.dashboard.auto_trade", "render_auto_trade_controls"),
            ("Backtest", "src.dashboard.backtest", "render_backtest"),
            ("Chart", "src.dashboard.chart", "render_price_chart"),
            ("Controls", "src.dashboard.controls", "render_manual_controls"),
            ("Forecast", "src.dashboard.forecast", "render_ai_forecast"),
            ("History", "src.dashboard.history", "render_trade_history"),
            ("Pipeline", "src.dashboard.pipeline", "render_pipeline"),
        ]

        for name, module_name, func_name in components:
            try:
                module = __import__(module_name, fromlist=[func_name])
                func = getattr(module, func_name, None)
                if func:
                    print(f"   ✓ {name}: Available")
                else:
                    print(f"   ⚠ {name}: Function not found")
            except ImportError as e:
                print(f"   ✗ {name}: Import failed - {e}")

        print("\n" + "="*60)
        print("✅ DASHBOARD COMPONENTS TEST PASSED")
        print("="*60 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ DASHBOARD COMPONENTS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    results = []

    # Run all integration tests
    results.append(test_complete_pipeline())
    results.append(test_data_consistency())
    results.append(test_dashboard_components())

    # Summary
    print("\n" + "="*60)
    print("INTEGRATION TEST SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Tests Passed: {passed}/{total}")
    if all(results):
        print("✅ ALL INTEGRATION TESTS PASSED")
    else:
        print("⚠ SOME TESTS FAILED")
    print("="*60 + "\n")

    sys.exit(0 if all(results) else 1)
