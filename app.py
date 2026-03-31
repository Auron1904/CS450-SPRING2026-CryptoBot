import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.data.downloader import fetch_and_process_data

st.set_page_config(page_title="Crypto Bot Dashboard", layout="wide")

st.title("🚀 CS450 CryptoTradeBot Dashboard")

# --- Sidebar Controls ---
st.sidebar.header("Data Settings")
days_to_fetch = st.sidebar.slider("Days of History", 7, 365, 90)

if st.sidebar.button("Fetch Fresh Data"):
    with st.spinner("Updating dataset..."):
        df = fetch_and_process_data(days=days_to_fetch)
        if df is not None:
            df.to_csv("data/raw/bitcoin_dataset.csv", index=False)
            st.sidebar.success("CSV Updated!")

# --- Main Dashboard ---
try:
    # Load the data that was saved by your fetch script
    df = pd.read_csv("data/raw/bitcoin_dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Create an Interactive Chart
    fig = go.Figure()

    # Add Price Line
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['price'], 
        name='BTC Price', line=dict(color='#1f77b4')
    ))

    # Add Moving Average Line (from your fetch_data.py logic)
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ma_5'], 
        name='5-Day Moving Average', line=dict(color='#ff7f0e', dash='dot')
    ))

    fig.update_layout(
        title="Bitcoin Price vs Moving Average",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show Raw Data Table
    if st.checkbox("Show Raw Data Table"):
        st.dataframe(df)

except FileNotFoundError:
    st.warning("No data found. Please click 'Fetch Fresh Data' in the sidebar to start.")