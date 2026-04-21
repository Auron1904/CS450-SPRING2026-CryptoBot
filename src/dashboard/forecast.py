from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.data.ohlcv import fetch_and_save_ohlcv
from src.features.indicators import build_feature_dataset
from src.model.predict import get_latest_prediction


def _run_live_pipeline(project_root: Path) -> dict:
    raw_output_dir = project_root / "data" / "raw"
    processed_path = project_root / "data" / "processed" / "btc_features.csv"

    raw_df, _ = fetch_and_save_ohlcv(
        symbol="BTC-USD",
        timeframe="1d",
        output_dir=raw_output_dir,
    )

    feature_df = build_feature_dataset(raw_df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(processed_path, index=False)

    return get_latest_prediction(
        model_path=project_root / "models" / "btc_model.json",
        data_path=processed_path,
    )


def _load_cached_prediction(project_root: Path) -> dict:
    return get_latest_prediction(
        model_path=project_root / "models" / "btc_model.json",
        data_path=project_root / "data" / "processed" / "btc_features.csv",
    )


def render_ai_forecast() -> None:
    """
    Render a small forecast card that matches the existing Streamlit styling.
    """
    try:
        project_root = Path(__file__).resolve().parents[2]

        if "ai_forecast_prediction" not in st.session_state:
            st.session_state["ai_forecast_prediction"] = _load_cached_prediction(
                project_root
            )

        if st.button("Refresh Data & Predict", key="refresh_data_predict"):
            with st.spinner("Fetching live market data..."):
                st.session_state["ai_forecast_prediction"] = _run_live_pipeline(
                    project_root
                )

        prediction = st.session_state["ai_forecast_prediction"]

        st.subheader(f"AI Forecast (Date: {prediction['target_date']})")

        prediction_value = prediction["prediction"]
        if prediction_value == "UP":
            st.markdown("**Prediction:** :green[UP]")
        else:
            st.markdown("**Prediction:** :red[DOWN]")

        st.markdown(f"**Probability:** {prediction['probability']:.2%}")
        st.progress(float(prediction["probability"]))
    except FileNotFoundError as exc:
        st.info(f"AI forecast unavailable: {exc}")
    except Exception as exc:
        st.error(f"AI forecast failed: {exc}")
