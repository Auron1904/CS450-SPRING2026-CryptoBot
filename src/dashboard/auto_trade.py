from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from src.bot.alpaca_client import AlpacaClient
from src.bot.logic import MATrendStrategy
from src.data.ohlcv import fetch_and_save_ohlcv
from src.features.indicators import build_feature_dataset
from src.model.predict import get_latest_prediction


def render_auto_trade_controls(alpaca_client: AlpacaClient | None) -> None:
    """
    Render auto-trade sidebar controls and logic.
    """
    # Initialize session state
    if "auto_trade_mode" not in st.session_state:
        st.session_state["auto_trade_mode"] = False
    if "auto_trade_confirmed" not in st.session_state:
        st.session_state["auto_trade_confirmed"] = False
    if "auto_trade_notional" not in st.session_state:
        st.session_state["auto_trade_notional"] = 50.0
    if "last_auto_trade_time" not in st.session_state:
        st.session_state["last_auto_trade_time"] = None
    if "trades_today_count" not in st.session_state:
        st.session_state["trades_today_count"] = 0
    if "trades_today_date" not in st.session_state:
        st.session_state["trades_today_date"] = None

    # ========== AUTO-TRADE SECTION ==========
    st.sidebar.header("🤖 Auto-Trade (24hr)")

    # Toggle switch
    auto_trade_mode = st.sidebar.toggle(
        "Enable Auto-Trade",
        value=st.session_state["auto_trade_mode"],
        key="auto_trade_toggle",
    )
    st.session_state["auto_trade_mode"] = auto_trade_mode

    # Status indicator
    if auto_trade_mode:
        st.sidebar.success("🟢 Auto-Trade: ON (24hr mode)")
    else:
        st.sidebar.info("⚪ Auto-Trade: OFF")

    # Confirmation checkbox
    confirm = st.sidebar.checkbox(
        "I confirm auto-trade",
        value=st.session_state["auto_trade_confirmed"],
        key="auto_trade_confirm",
    )
    st.session_state["auto_trade_confirmed"] = confirm

    # Notional input
    notional = st.sidebar.number_input(
        "Auto-Trade Notional (USD)",
        min_value=1.0,
        max_value=10000.0,
        value=st.session_state["auto_trade_notional"],
        step=1.0,
        key="auto_trade_notional_input",
    )
    st.session_state["auto_trade_notional"] = notional

    # Status display when ON
    if auto_trade_mode:
        st.sidebar.divider()

        last_trade_time = st.session_state.get("last_auto_trade_time")
        if last_trade_time:
            st.sidebar.caption(f"Last trade: {last_trade_time}")

        today = pd.Timestamp.today().date()
        trades_today_date = st.session_state.get("trades_today_date")
        if trades_today_date != today:
            st.session_state["trades_today_count"] = 0
            st.session_state["trades_today_date"] = today

        st.sidebar.caption(f"Trades today: {st.session_state['trades_today_count']}")

        st.sidebar.divider()

        # Run auto-trade logic
        _execute_auto_trade_logic(alpaca_client)


def _execute_auto_trade_logic(alpaca_client: AlpacaClient | None) -> None:
    """
    Main auto-trade execution logic.
    """
    try:
        if alpaca_client is None:
            return

        auto_trade_mode = st.session_state.get("auto_trade_mode", False)
        confirm = st.session_state.get("auto_trade_confirmed", False)
        notional = st.session_state.get("auto_trade_notional", 50.0)

        if not auto_trade_mode or not confirm:
            return

        # Check cooldown (1 hour minimum between trades)
        last_trade_time = st.session_state.get("last_auto_trade_time")
        if last_trade_time:
            time_since_last = datetime.now() - datetime.fromisoformat(last_trade_time)
            if time_since_last < timedelta(hours=1):
                st.sidebar.info(
                    f"⏳ Trade cooldown active. Next trade in {timedelta(hours=1) - time_since_last}"
                )
                return

        project_root = Path(__file__).resolve().parents[2]

        # Fetch live data
        with st.spinner("🔄 Auto-trade: Fetching data..."):
            raw_output_dir = project_root / "data" / "raw"
            raw_output_dir.mkdir(parents=True, exist_ok=True)

            raw_df, _ = fetch_and_save_ohlcv(
                symbol="BTC-USD",
                timeframe="1d",
                output_dir=raw_output_dir,
            )

            # Build features
            feature_df = build_feature_dataset(raw_df)
            processed_path = project_root / "data" / "processed"
            processed_path.mkdir(parents=True, exist_ok=True)
            feature_csv = processed_path / "btc_features.csv"
            feature_df.to_csv(feature_csv, index=False)

            # Get AI prediction
            prediction = get_latest_prediction(
                model_path=project_root / "models" / "btc_model.json",
                data_path=feature_csv,
            )

            # Get MA_5 signal
            strategy = MATrendStrategy()
            ma_signal = strategy.generate_signal()

            # Combine signals
            ai_prediction = prediction.get("prediction", "DOWN")
            ma_signal_value = ma_signal.get("signal", "HOLD")

            # Execute if both signals agree or AI is strong
            should_trade = False
            trade_side = None

            if ai_prediction == "UP" and ma_signal_value == "BUY":
                should_trade = True
                trade_side = "BUY"
            elif ai_prediction == "DOWN" and ma_signal_value == "SELL":
                should_trade = True
                trade_side = "SELL"

            if should_trade and trade_side:
                # Check for open positions
                positions = alpaca_client.get_open_positions()
                if "error" in positions:
                    st.sidebar.error(f"❌ Position check failed: {positions['error']}")
                    return

                # Execute trade
                if trade_side == "SELL" and not positions.get("has_btc_position", False):
                    st.sidebar.warning("⚠️ No BTC position to sell")
                    return

                order_result = alpaca_client.execute_market_order(
                    symbol="BTC/USD",
                    notional=float(notional),
                    side=trade_side,
                )

                if "error" not in order_result:
                    st.sidebar.success(
                        f"✅ Auto-trade executed: {trade_side} ({order_result['status']})"
                    )
                    st.sidebar.info(
                        f"Prediction: {ai_prediction} | MA Signal: {ma_signal_value}"
                    )

                    # Update last trade time and count
                    st.session_state["last_auto_trade_time"] = datetime.now().isoformat()
                    st.session_state["trades_today_count"] += 1

                    # Show toast
                    st.toast(f"🤖 Auto-trade: {trade_side} executed")
                else:
                    st.sidebar.error(f"❌ Trade execution failed: {order_result['error']}")

    except Exception as e:
        st.sidebar.error(f"❌ Auto-trade error: {e}")
