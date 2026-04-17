import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def render_price_chart(
    data_path: str = "data/raw/bitcoin_dataset.csv",
) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["price"],
                name="BTC Price",
                line=dict(color="#1f77b4"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["ma_5"],
                name="5-Day Moving Average",
                line=dict(color="#ff7f0e", dash="dot"),
            )
        )
        fig.update_layout(
            title="Bitcoin Price vs Moving Average",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            hovermode="x unified",
        )

        st.plotly_chart(fig, use_container_width=True)
        return df
    except FileNotFoundError:
        st.warning(
            "No data found. Please click 'Fetch Fresh Data' in the sidebar to start."
        )
        return None
    except Exception as exc:
        st.error(f"Chart component failed: {exc}")
        return None
