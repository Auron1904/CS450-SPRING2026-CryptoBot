import streamlit as st
import pandas as pd
import plotly.express as px
import ccxt

st.set_page_config(page_title="CryptoBot Dashboard", layout="wide")

st.title("🚀 CS450 CryptoTradeBot")
st.sidebar.header("Controls")

# Test CCXT Connection
exchange = ccxt.binance()
symbol = st.sidebar.selectbox("Select Coin", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])

if st.button("Fetch Live Price"):
    ticker = exchange.fetch_ticker(symbol)
    st.metric(label=f"Current {symbol} Price", value=f"${ticker['last']:,}")
    st.success("Connection to Binance API Successful!")