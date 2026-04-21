import streamlit as st

from src.bot.alpaca_client import AlpacaClient
from src.bot.logic import MATrendStrategy
from src.dashboard.account import render_account_metrics
from src.dashboard.chart import render_price_chart
from src.dashboard.controls import render_manual_controls
from src.dashboard.forecast import render_ai_forecast
from src.dashboard.history import render_trade_history
from src.data.downloader import fetch_and_process_data

st.set_page_config(page_title="Crypto Bot Dashboard", layout="wide")

st.title("🚀 CS450 CryptoTradeBot Dashboard")

if "strategy" not in st.session_state:
    st.session_state["strategy"] = MATrendStrategy()

if "latest_signal" not in st.session_state:
    st.session_state["latest_signal"] = None

if "latest_order" not in st.session_state:
    st.session_state["latest_order"] = None

if "force_signal" not in st.session_state:
    st.session_state["force_signal"] = "None"

if "previous_account_snapshot" not in st.session_state:
    st.session_state["previous_account_snapshot"] = None

alpaca_client = None
try:
    alpaca_client = AlpacaClient()
except ValueError:
    pass

st.sidebar.header("Data Settings")
days_to_fetch = st.sidebar.slider("Days of History", 7, 365, 90)

if st.sidebar.button("Fetch Fresh Data"):
    with st.spinner("Updating dataset..."):
        dataset = fetch_and_process_data(days=days_to_fetch)
        if dataset is not None:
            dataset.to_csv("data/raw/bitcoin_dataset.csv", index=False)
            st.sidebar.success("CSV Updated!")
        else:
            st.sidebar.error("Failed to fetch data from CoinGecko.")

st.sidebar.header("Trade Signal")
if st.sidebar.button("Generate Trade Signal"):
    with st.spinner("Generating MA_5 signal..."):
        st.session_state["latest_signal"] = st.session_state[
            "strategy"
        ].generate_signal()

latest_signal = st.session_state.get("latest_signal")
if latest_signal:
    signal_value = latest_signal.get("signal", "HOLD")
    if signal_value == "BUY":
        st.sidebar.success(f"Signal: {signal_value}")
    elif signal_value == "SELL":
        st.sidebar.error(f"Signal: {signal_value}")
    else:
        st.sidebar.warning(f"Signal: {signal_value}")

render_manual_controls(alpaca_client)

df = render_price_chart()
render_ai_forecast()
render_account_metrics(alpaca_client)

if latest_signal:
    st.subheader("Latest Trade Signal")
    col1, col2, col3 = st.columns(3)
    current_price = latest_signal.get("current_price")
    ma_5 = latest_signal.get("ma_5")
    current_price_label = (
        "N/A" if current_price is None else f"{float(current_price):,.2f}"
    )
    ma_5_label = "N/A" if ma_5 is None else f"{float(ma_5):,.2f}"
    col1.metric("Signal", latest_signal.get("signal", "HOLD"))
    col2.metric("Current Price", current_price_label)
    col3.metric("MA_5", ma_5_label)

render_trade_history(alpaca_client)

if df is not None and st.checkbox("Show Raw Data Table"):
    st.dataframe(df)
