import pandas as pd
import streamlit as st


def render_trade_history(client) -> None:
    try:
        st.subheader("Recent Trades")

        if client is None:
            st.info("Trade history unavailable until Alpaca client is configured.")
            return

        trades = client.get_recent_trades(limit=10)
        if isinstance(trades, dict) and "error" in trades:
            st.error(trades["error"])
            return

        if not trades:
            st.info("No recent trades found.")
            return

        history_df = pd.DataFrame(trades)
        st.dataframe(history_df, use_container_width=True)
    except Exception as exc:
        st.error(f"Trade history component failed: {exc}")
